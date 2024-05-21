import torch
import torch.nn as nn

import utils
from utils import negative_sampling
from torch_geometric.datasets import Planetoid
from utils import get_sparse_norm_adj
from torch_geometric.utils import dense_to_sparse,remove_self_loops
from utils import get_sparse_norm_adj,mi_loss,reconstruct_loss
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self,input_dims,output_dims,bias=False,use_act=True):
        super(GCN, self).__init__()
        self.weights=torch.nn.Parameter(torch.FloatTensor(input_dims,output_dims))
        self.use_bias=bias
        if bias:
            self.bias=torch.nn.Parameter(torch.FloatTensor(1,output_dims))
        else:
            self.register_parameter('bias',None)
        self.reset_parameter()
        self.use_act=use_act

        self.act=nn.PReLU()
    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            torch.nn.init.xavier_uniform_(self.bias)

    def forward(self,node_x,adj):
        support=torch.matmul(node_x,self.weights)
        output=torch.sparse.mm(adj,support)
        if self.use_bias:
            output+=self.bias
        if self.use_act==False:
            return output,support
        else:
            return self.act(output),support



class Discriminator(nn.Module):
    def __init__(self,input_dims1,input_dims2):
        super(Discriminator, self).__init__()
        self.static_network=nn.Bilinear(in1_features=input_dims1,in2_features=input_dims2,out_features=1)
        self.act=nn.Sigmoid()
        # 初始化各模块的参数
        for parameter in self.modules():
            self.reset_parameter(parameter)

    def reset_parameter(self,parameter):
        if isinstance(parameter,nn.Bilinear):
            torch.nn.init.xavier_uniform_(parameter.weight.data)
            if parameter.bias is not None:
                parameter.bias.data.fill_(0.0)

    def forward(self,node_x,neighbor_x,sample_list):
        pos_scores=self.static_network(node_x,neighbor_x)
        pos_scores=self.act(pos_scores)
        neg_scores_list=[]
        for i in range(len(sample_list)):
            neg_sample=neighbor_x[sample_list[i]]
            neg_scores=self.static_network(node_x,neg_sample)
            neg_scores_list.append(neg_scores)
        neg_scores=torch.stack(neg_scores_list,dim=1).squeeze(dim=-1)
        neg_scores=self.act(neg_scores)

        return pos_scores,neg_scores

# 估计节点和其邻域的互信息大小，返回每个节点关于其邻域的互信息
class MI_Estimator_POOL(nn.Module):
    def __init__(self,input_dims1,input_dims2,ratio,lamb):
        super(MI_Estimator_POOL, self).__init__()
        self.disc=Discriminator(input_dims1=input_dims1,input_dims2=input_dims2)
        self.act=nn.PReLU()
        self.ratio=ratio
        self.lamb=lamb


    def Agg_neighbor(self,h_w,adj):
        adj_ori,_=remove_self_loops(adj.coalesce())
        return torch.spmm(adj_ori,h_w)

    # 图重构部分就要重新写，弄成多个图的情况如何构建图
    def get_coarsen_Graph(self,scores,node_x,adj):
        num_nodes=node_x.shape[0]
        top_count=int(num_nodes*self.ratio)

        # 将关键节点选择出来  提取这些节点的特征
        values,perm=torch.topk(scores.squeeze(dim=-1),k=top_count)
        node_x=node_x[perm,:]
        node_x=node_x*values.view(-1,1)

        # 6.656933784484863  11.08916187286377
        # 得到保留的图结构 adj是输入的归一化的邻接矩阵
        adj_ori=adj.bool().float().to_dense()
        #num_nodes=adj_ori.shape[0]      #考虑是根据节点的总数还是邻域个数

        # 计算任意两个节点的邻域总和  15.52151870727539
        num_neighbors=adj_ori.sum(dim=1).unsqueeze(1)
        sum_neighbor=num_neighbors+num_neighbors.t()

        # 计算任意两个节点之间公共邻域节点个数
        common_neighbor=torch.matmul(adj_ori,adj_ori)

        # 后续考虑完全利用互信息的方式来构造图结构
        # adj_info=common_neighbor.float()/num_nodes
        #adj_ori=2.0*adj_ori/(num_neighbors+num_neighbors.t())
        # torch.cuda.memory_allocated()/(1024**3)
        #print("计算adj_info之前使用显存{:.4f}".format(torch.cuda.memory_allocated()/(1024**3)))
        adj_info=2*common_neighbor/sum_neighbor

        struct_adj=adj_info[perm,:]
        struct_adj=struct_adj[:,perm]
        # 归一化
        #struct_adj_edge,struct_adj_weights=dense_to_sparse(struct_adj)
        #struct_adj=get_sparse_norm_adj(struct_adj_edge, struct_adj_weights, top_count).to_dense()
        #struct_adj.fill_diagonal_(0)

        # 粗化后图结构
        adj=adj.to_dense()
        c_adj=adj[perm,:]
        c_adj=c_adj[:,perm]
        # 结构加权
        c_adj=c_adj+self.lamb*struct_adj
        edge_index,edge_weights=dense_to_sparse(c_adj)
        adj_norm=get_sparse_norm_adj(edge_index,edge_weights,top_count)
        #adj_norm=torch.sparse.FloatTensor(edge_index,edge_weights,size=(top_count,top_count))
        return node_x,adj_norm,perm


    def get_coarsen_graph(self,scores,node_x,adj):
        """
        输入  节点得分  节点嵌入表示  邻接矩阵
        输出  粗化图节点特征  归一化的图结构  粗图节点的下标
        生成粗化的图：  节点特征和图结构
        :param scores:
        :param adj:这个位置传入的是归一化的图邻接矩阵   但是我希望能够得到的是非归一化的邻接矩阵  然后对其做归一化
        :param node_x:
        :param ratio:
        :return:
        返回粗化后节点特征   未归一化的图结构  节点的下标
        """
        num_nodes=node_x.shape[0]
        top_count=int(num_nodes*self.ratio)
        values,perm=torch.topk(scores.squeeze(dim=-1),k=top_count)
        # 生成粗化节点特征
        c_node_x=node_x[perm,:]
        c_node_x=c_node_x*values.view(-1,1)

        # 生成粗化图结构
        adj=adj.bool().float().to_dense()
        S=adj[perm,:]
        coarsen_graph=torch.matmul(S,torch.matmul(adj,S.transpose(0,1)))

        #coarsen_graph=adj.bool().float().to_dense()
        # 这样得到的邻接矩阵太过稠密了
        #coarsen_graph=torch.matmul(coarsen_graph,coarsen_graph).bool().float()
        #coarsen_graph=coarsen_graph[perm,:]
        #coarsen_graph=coarsen_graph[:,perm]
        #torch.sparse.FloatTensor(torch.nonzero(dense_matrix), dense_matrix[dense_matrix != 0])
        coarsen_graph=F.dropout(coarsen_graph,p=0.1,training=self.training)
        edge_index,_=dense_to_sparse(coarsen_graph)
        edge_index,_=remove_self_loops(edge_index)
        # 尽可能将边的权重也返回回去  看一下有没有上85的
        return c_node_x,edge_index,perm


    def forward(self,node_x,h_neighbor,adj,neg_num):
        """
        :param node_x:
        :param h_neighbor:
        :param adj:      添加自循环的归一化的矩阵
        :param neg_num:
        :return:
        返回粗化后的节点特征 图结构 idx和当前层的互信息损失
        """
        # 问题：h_neighbor 和node_x是一样的
        h_neighbor=self.act(self.Agg_neighbor(h_neighbor,adj))
        neg_samples=negative_sampling(node_x.shape[0],neg_num)
        pos_scores,neg_scores=self.disc(node_x,h_neighbor,neg_samples)
        c_node_x,c_graph,perm=self.get_coarsen_Graph(pos_scores,node_x,adj)
        adj_rebuilt=torch.sigmoid(torch.matmul(node_x,node_x.transpose(0,1)))
        struct_loss = reconstruct_loss(adj_rebuilt, adj)
        return c_node_x,c_graph,perm,mi_loss(pos_score=pos_scores,neg_score=neg_scores),struct_loss


if __name__=="__main__":
    print('test')
    dataset=Planetoid(root='data/Cora',name="Cora")
    data=dataset[0]
    num_nodes=data.num_nodes
    input_dims=data.num_features
    hidden_dims=512
    # 对图数据进行处理   一个要归一化然后进行
    norm_adj=get_sparse_norm_adj(edge_index=data.edge_index,num_nodes=num_nodes)
    #ori_adj=get_sparse_adj(edge_index=data.edge_index,num_nodes=num_nodes)
    print(data)




