import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from CMIGNet import CMIGNet, GNN, LDGCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40, ModelNet40RPMNet
from util import npmat2euler, IOStream
import numpy as np
from tqdm import tqdm
import random
import scipy.io as io
import time


torch.backends.cudnn.enabled = False  # fix cudnn non-contiguous error

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def test_one_epoch(args, net, test_loader):
    net.eval()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []
    count = 0
    for src, target, R, t, euler, views in tqdm(test_loader):

        src = src.cuda()
        target = target.cuda()


        R_pred, t_pred, *_ = net(src, target, views)
        src2 = torch.matmul(R_pred, src) + t_pred.unsqueeze(-1)

        count += 1
        R_list.append(R.numpy())
        t_list.append(t.numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    for src, target, R, t, euler, views in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        R = R.cuda()
        t = t.cuda()
        views = views.cuda()
        # print(views.shape) # torch.Size([batchsize, 20, 3, 224, 224])
        # print(type(views)) # <class 'torch.Tensor'>

        opt.zero_grad()
        R_pred, t_pred, loss, *_ = net(src, target, views, R, t)

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

        loss.backward()
        opt.step()

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae, loss


def train(args, net, train_loader, test_loader, io):
    opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = MultiStepLR(opt, milestones=[50, 75], gamma=0.5)

    train_stats = test_one_epoch(args, net, test_loader)
    for epoch in range(args.epochs):
        test = False
        train_stats = train_one_epoch(args, net, train_loader, opt)

        test = True
        test_stats = test_one_epoch(args, net, test_loader)


        print('=====  EPOCH %d  =====' % (epoch+1))
        print('TRAIN, rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f, loss: %f' % train_stats)
        if test:
            print('TEST,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % test_stats)
        # io.cprint('=====  EPOCH %d  =====' % (epoch + 1))
        io.cprint('TRAIN epoch %d, rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f, loss: %f' % (
                (epoch + 1), train_stats[0], train_stats[1], train_stats[2], train_stats[3], train_stats[4]))
        if test:
            io.cprint('TEST epoch %d,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % (
                (epoch + 1), test_stats[0], test_stats[1], test_stats[2], test_stats[3]))

        torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))

        scheduler.step()


def main():
    arg_bool = lambda x: x.lower() in ['true', 't', '1']
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num_iter', type=int, default=3, metavar='N',
                        help='Number of iteration inside the network')
    parser.add_argument('--emb_nn', type=str, default='GNN', metavar='N',
                        help='Feature extraction method. [GNN]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings') 
    parser.add_argument('--batch_size', type=int, default=6, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=6, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--unseen', type=arg_bool, default='True',
                        help='Test on unseen categories')
    parser.add_argument('--gaussian_noise', type=arg_bool, default='False',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--alpha', type=float, default=0.75, metavar='N',
                        help='Fraction of points when sampling partial point cloud')  
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')  
    parser.add_argument('--pretrained', type=arg_bool, default='False',
                        help='Load pretrained weight') 
    parser.add_argument('--K_test', type=int, default=256,
                        help='The number of key points preserved during testing')
    parser.add_argument('--overlap_cl', type=arg_bool, default='False',
                        help='Wheter to use overlap cl')
    parser.add_argument('--n_emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')

    args = parser.parse_args()
    print(args)

    ##### make checkpoint directory and backup #####
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    ##### make checkpoint directory and backup #####

    io = IOStream('checkpoints/' + args.exp_name + '/log.txt')
    io.cprint(str(args))

    ##### load data #####
    train_loader = DataLoader(
        ModelNet40(partition='train', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=10)
    test_loader = DataLoader(
        ModelNet40(partition='test', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=10)
    

    ##### load model #####
    net = CMIGNet(LDGCNN(), args).cuda()


    ##### train #####
    train(args, net, train_loader, test_loader, io)
    io.close()

if __name__ == '__main__':
    seed_everything(3407)
    main()
