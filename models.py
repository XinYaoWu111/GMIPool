import torch
import torch.nn as nn
from layers import  GCN,MI_Estimator_POOL
from utils import get_sparse_norm_adj,get_sparse_norm_adj_add_self_loop
from utils import get_sparse_norm_adj,mi_loss,reconstruct_loss
import torch.nn.functional as F

class NodeClassificationModel(nn.Module):
    def __init__(self,args):
        super(NodeClassificationModel, self).__init__()
        # depth代表池化次数
        assert args.depth>=1
        self.depth=args.depth
        self.in_channels=args.in_channels
        self.out_channels=args.out_channels
        # hidden_channels 应该改一下，尽量使得MI_Estimator 里面 比普通的GCN大一些
        self.hidden_channels=args.hidden_channels
        self.ratios=[args.ratio1,args.ratio2,args.ratio3,args.ratio4,args.ratio5]
        self.act=nn.ReLU(inplace=True)
        self.dropout=args.dropout
        # 构造编码器
        self.encoder=torch.nn.ModuleList()
        self.encoder.append(GCN(input_dims=self.in_channels,output_dims=self.hidden_channels))

        self.pool=torch.nn.ModuleList()
        for l in range(self.depth):
            self.pool.append(MI_Estimator_POOL(input_dims1=self.hidden_channels,input_dims2=self.hidden_channels,
                                               ratio=self.ratios[l],lamb=args.lamb))
            self.encoder.append(GCN(input_dims=self.hidden_channels,output_dims=self.hidden_channels))

        # 解码部分
        self.decoder=torch.nn.ModuleList()
        for i in range(self.depth):
            self.decoder.append(GCN(self.hidden_channels,self.hidden_channels))
        self.decoder.append(GCN(self.hidden_channels,self.out_channels,use_act=False))

    def unpool(self,adj,node_x,idx):
        new_node_x=node_x.new_zeros([adj.shape[0],node_x.shape[1]])
        new_node_x[idx]=node_x
        return new_node_x


    def forward(self,node_x,edge_index,neg_nums):
        adj_list=[]             # 保留每个粒层的图结构
        down_out_list=[]        # 保留每个粒层得到的图特征
        indices_list=[]         # 保留每个粒层池化的节点的下标
        mi_loss_list=[]         # 保留每个粒层计算的互信息
        struct_loss_list=[]
        node_x_list = []        #
        adj_norm = get_sparse_norm_adj_add_self_loop(edge_index, node_x.shape[0])
        node_x=F.dropout(node_x,p=self.dropout,training=self.training)
        for l in range(self.depth):

            #adj_ori = get_sparse_norm_adj(edge_index, edge_weights=None,num_nodes=node_x.shape[0])
            #adj_norm = get_sparse_norm_adj_add_self_loop(edge_index, node_x.shape[0])

            node_x,h_neighbor=self.encoder[l](node_x,adj_norm)
            adj_list.append(adj_norm)
            down_out_list.append(node_x)

            node_x,adj_norm,perm,mi_loss,struct_loss=self.pool[l](node_x,h_neighbor,adj_norm,neg_nums)

            #node_x, edge_index, perm, mi_loss = self.pool[l](node_x, h_neighbor, adj_norm, neg_nums)

            indices_list.append(perm)
            mi_loss_list.append(mi_loss)
            struct_loss_list.append(struct_loss)

        # 在最后一个粒层上做一次编码
        node_x, _ = self.encoder[-1](node_x, adj_norm)
        #node_x,_=self.encoder[-1](node_x,get_sparse_norm_adj_add_self_loop(edge_index,node_x.shape[0]))

        # 解码部分   解码过程导致梯度不可导
        for l in range(self.depth):
            up_idx=self.depth-l-1
            adj,perm=adj_list[up_idx],indices_list[up_idx]
            # 做一个上采样
            node_x=self.unpool(adj,node_x,perm)

            node_x,_=self.decoder[l](node_x,adj)
            node_x=node_x+down_out_list[up_idx]
        adj_rebuilt=torch.sigmoid(torch.matmul(node_x,node_x.transpose(0,1)))
        struct_loss=reconstruct_loss(adj_rebuilt,adj)
        node_x,_=self.decoder[-1](node_x,adj)
        return node_x,mi_loss_list,struct_loss





