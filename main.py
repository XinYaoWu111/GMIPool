import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import NodeClassificationModel
from utils import setup_seed
import time
import argparse
torch.autograd.set_detect_anomaly(True)
parser=argparse.ArgumentParser()

parser.add_argument('--seed',type=int,default=4144,help='random seed')
parser.add_argument('--hidden_channels',type=int,default=32,help='hidden layer dims')
parser.add_argument('--depth',type=int,default=4,help='pool depth')
parser.add_argument('--neg_num',type=int,default=1,help='number of negtive samples')
parser.add_argument('--lr',type=float,default=0.002,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.01,help='weight decay')
parser.add_argument('--dropout',type=float,default=0.76,help='dropout rate')
parser.add_argument('--lamb',type=float,default=0.2,help='weighted parameters')
parser.add_argument('--alpha',type=float,default=2.0,help='balancing parameters')
parser.add_argument('--beta',type=float,default=0.000,help='balancing parameters')
parser.add_argument('--dataset',type=str,default='Cora',help='Cora/Citeseer/Pubmed')
parser.add_argument('--device',type=str,default='cuda:0',help='cuda device')
parser.add_argument('--epochs',type=int,default=200,help='maximum number of epoch')
parser.add_argument('--ratio1', type=float, default=0.5, help='pool1 parameter')
parser.add_argument('--ratio2', type=float, default=0.8, help='pool2 parameter')
parser.add_argument('--ratio3', type=float, default=0.8, help='pool3 parameter')
parser.add_argument('--ratio4', type=float, default=0.7, help='pool4 parameter')
parser.add_argument('--ratio5', type=float, default=0.999, help='pool5 parameter')
args = parser.parse_args()


def compute_test(model,mask):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        loss_test = 0.0
        out,_,_ = model(data.x, data.edge_index,neg_nums=args.neg_num)
        out = F.log_softmax(out, dim=1)
        pred = out[mask].max(dim=1)[1]
        correct += pred.eq(data.y[mask]).sum().item()
        loss_test += F.nll_loss(out[mask], data.y[mask]).item()
        return correct / mask.sum().item(), loss_test

def train(model,optimizer):
    best_val_loss=1e9
    best_val_acc=0.0
    best_test_acc=0.0
    best_epoch = 0
    t=time.time()

    for epoch in range(args.epochs):
        loss_train=0.0
        model.train()
        optimizer.zero_grad()

        output,mi_loss_list,struct_loss=model(data.x,data.edge_index,neg_nums=args.neg_num)

        output=F.log_softmax(output,dim=1)
        mi_loss = torch.mean(torch.stack(mi_loss_list, dim=0))+args.beta*struct_loss
        #loss=F.nll_loss(output[data.train_mask], data.y[data.train_mask]) + args.alpha* mi_loss
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask]) + max(0.0, args.alpha - 0.01 * epoch) * mi_loss

        loss.backward()
        optimizer.step()

        loss_train+=loss.item()
        pred=output[data.train_mask].max(dim=1)[1]
        correct=pred.eq(data.y[data.train_mask]).sum().item()
        acc_train=correct/data.train_mask.sum().item()


        # 模型评估阶段
        acc_val,loss_val=compute_test(model,data.val_mask)
        acc_test,loss_test=compute_test(model,data.test_mask)

        if acc_test>best_test_acc:
            best_test_acc=acc_test

        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train),
              'acc_train: {:.4f}'.format(acc_train), 'loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(acc_val), 'best_acc: {:.4f}'.format(best_test_acc),
                'time:{:.4f}'.format(time.time() - t))

        # 保留最优的model
        if loss_val<best_val_loss:
            best_val_loss=loss_val
            best_epoch=epoch+1
            torch.save(model.state_dict(), f"best_model.pth")

    return best_epoch


if __name__=='__main__':
    test_acc_list=[]
    setup_seed(args.seed)
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0].to(device=args.device)
    args.num_nodes = data.num_nodes
    args.in_channels = data.num_features
    args.out_channels = dataset.num_classes
    print(args)
    model = NodeClassificationModel(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model=train(model,optimizer)

    model.load_state_dict(torch.load('best_model.pth'))

    test_acc, test_loss = compute_test(model,mask=data.test_mask)
    test_acc_list.append(test_acc)
    print('Test set results, best_epoch={:04d} loss = {:.4f}, accuracy = {:.4f}'.format(best_model,test_loss, test_acc))
