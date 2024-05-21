import torch
import numpy as np
import random
import os
from torch_geometric.utils import add_self_loops
# 添加自循环
def get_sparse_norm_adj(edge_index,edge_weights,num_nodes):
    if edge_weights==None:
        edge_weights=torch.ones((edge_index.size(1),),device=edge_index.device)
    row,col=edge_index
    deg=torch.sparse_coo_tensor(indices=edge_index,values=edge_weights.float(),
                                size=torch.Size([num_nodes,num_nodes])).to_dense().sum(dim=1)
    deg_inv_sqrt=deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt==float('inf')]=0

    edge_weights=deg_inv_sqrt[row]*edge_weights*deg_inv_sqrt[col]
    # 创建稀疏矩阵
    #norm_adj=torch.sparse.FloatTensor(edge_index,edge_weights,size=(num_nodes,num_nodes))
    norm_adj = torch.sparse_coo_tensor(edge_index, edge_weights, size=(num_nodes, num_nodes))

    return norm_adj.coalesce()


def get_sparse_norm_adj_add_self_loop(edge_index,num_nodes):
    edge_index,edge_weights=add_self_loops(edge_index,num_nodes=num_nodes)
    norm_adj=get_sparse_norm_adj(edge_index,edge_weights,num_nodes)
    return norm_adj



def get_sparse_adj_ori(edge_index,num_nodes):
    edge_weight=torch.ones((edge_index.size(1),),device=edge_index.device)

    #adj=torch.sparse.FloatTensor(edge_index,edge_weight,size=(num_nodes,num_nodes))
    adj=torch.sparse_coo_tensor(edge_index,edge_weight,size=(num_nodes,num_nodes))
    return adj

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def negative_sampling(num_nodes,sample_times):
    sample_list=[]
    for i in range(sample_times):
        samples=[]
        j=0
        # 采样
        while True:
            randnum=np.random.randint(0,num_nodes)
            if randnum!=i:
                samples.append(randnum)
                j=j+1
            if len(samples)==num_nodes:
                break
        sample_list.append(samples)
    return sample_list



def soft_plus(x):
    return torch.log(1+torch.exp(x))

def mi_loss(pos_score,neg_score):
    pos_score=soft_plus(-pos_score)
    neg_score=torch.mean(soft_plus(neg_score),dim=-1)
    loss=(pos_score+neg_score)
    return torch.mean(loss)

def KL_Divergence(rebuilt_adj,ture_adj):
    epsilon=1e-8
    rebuilt_adj=torch.where(rebuilt_adj==0,torch.tensor(epsilon),rebuilt_adj)
    ture_adj=torch.where(ture_adj==0,torch.tensor(epsilon),ture_adj)
    return torch.sum(rebuilt_adj * (torch.log(rebuilt_adj) - torch.log(ture_adj)), dim=1)
def reconstruct_loss(pre, gnd):
    #gnd=gnd.bool().float().to_dense()
    gnd=gnd.to_dense()
    nodes_n = gnd.shape[0]
    edges_n=torch.sum(gnd!=0)/2
    #edges_n = np.sum(gnd)/2
    weight1 = (nodes_n*nodes_n-edges_n)*1.0/edges_n
    weight2 = nodes_n*nodes_n*1.0/(nodes_n*nodes_n-edges_n)
    #gnd = torch.FloatTensor(gnd).cuda()
    temp1 = gnd*torch.log(pre+(1e-10))*(-weight1)
    temp2 = (1-gnd)*torch.log(1-pre+(1e-10))
    return torch.mean(temp1-temp2)*weight2

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # * the rest for testing
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

















