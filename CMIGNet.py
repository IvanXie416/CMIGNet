import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import knn, batch_choice
import open3d as o3d
import scipy.io as io
import torchvision.models as models
from lightly.loss.ntx_ent_loss import NTXentLoss


def get_graph_feature(x, idx=None, k=20):
    # x = x.squeeze(-1)
    x = x.view(*x.size()[:3])
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    rel_pos = feature - x
    feature = torch.cat((rel_pos, x), dim=3).permute(0, 3, 1, 2)
    return feature


def knn(x, k=20):
    x = x.view(*x.size()[:3])
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = distance.sort(descending=True, dim=-1)[1]  # (batch_size, num_points, k)
    return idx[:, :, :k]


def drop_last_layer(model):
    new_model = nn.Sequential(*list(model.children())[:-1])
    return new_model


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def pdist(A, B, dist_type='L2'):
  if dist_type == 'L2':
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
  else:
    raise NotImplementedError('Not implemented')


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Propagate(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)  # [B, C, N, k]
        x = nn_feat - x.unsqueeze(-1)  # [B, c, N, k] - [B, c, N, 1] = [B, c, N, k]
        x = self.conv2d(x)  # [B, emb_dims, N, k]
        x = x.max(-1)[0]  # [B, C, N]
        x = self.conv1d(x)
        return x


class GNN(nn.Module):
    def __init__(self, emb_dims=64):
        super(GNN, self).__init__()
        self.propogate1 = Propagate(3, 64)  # Conv2DBNReLU(3,64)->Conv2DBNReLU(64,64)->Conv1DBNReLU(64,64)
        self.propogate2 = Propagate(64, 64)
        self.propogate3 = Propagate(64, 64)
        self.propogate4 = Propagate(64, 64)
        self.propogate5 = Propagate(64, emb_dims)

    def forward(self, x):
        # [B, 3, N]
        nn_idx = knn(x, k=12) 

        x = self.propogate1(x, nn_idx)
        x = self.propogate2(x, nn_idx)
        x = self.propogate3(x, nn_idx)
        x = self.propogate4(x, nn_idx)
        x = self.propogate5(x, nn_idx)  # [B, emb_dims, N]

        return x
    

class LDGCNN(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(LDGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(134, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(262, 64, kernel_size=1, bias=False)
        #self.conv3 = nn.Conv2d(390, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(390, 128, kernel_size=1, bias=False)
        #self.conv4 = nn.Conv2d(518, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(323, n_emb_dims, kernel_size=1, bias=False)
        #self.conv5 = nn.Conv2d(515, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x): # (2B, dim, num)
        batch_size, num_dims, num_points = x.size()
        x = x.unsqueeze(-1)
        edge_feature = get_graph_feature(x)
        net = F.relu(self.bn1(self.conv1(edge_feature)))
        # (bs, 64, num_points, 1)
        net = net.max(dim=-1, keepdim=True)[0]
        net1 = net
        idx = knn(net)
        # (bs, 67, num_points, 1)
        net = torch.cat([x, net1], dim=1)
        # (bs, 134, np, k)
        edge_feature = get_graph_feature(net, idx)
        # (bs, 64, num_points, k)
        net = F.relu(self.bn2(self.conv2(edge_feature)))
        # (bs, 64, num_points, 1)
        net = net.max(dim=-1, keepdim=True)[0]

        net2 = net
        idx = knn(net)
        # (bs, 131, num_points, 1)
        net = torch.cat([x, net1, net2], dim=1) # (2*B, 195, num_points, 1)
        # (bs, 262, np, k)
        edge_feature = get_graph_feature(net, idx) # (2*B, 390, num_points, k)
        net = F.relu(self.bn3(self.conv3(edge_feature))) # (2*B, 64, num_points, k)
        net = net.max(dim=-1, keepdim=True)[0] # (2*B, 64, num_points, 1)
        net3 = net
        idx = knn(net)
        # (bs, 195, np, 1)
        net = torch.cat([x, net1, net2, net3], dim=1) # (2*B, 259, num_points, 1)
        # (bs, 390, np, k)
        edge_feature = get_graph_feature(net, idx) # (2*B, 518, num_points, k)
        # (bs, 128, np, k)
        net = F.relu(self.bn4(self.conv4(edge_feature))) # (2*B, 128, num_points, k)    
        # (bs, 128, np, 1)
        net = net.max(dim=-1, keepdim=True)[0]

        net4 = net
        # (bs, 323, np, 1)
        net = torch.cat([x, net1, net2, net3, net4], dim=1) # (2*B, 515, num_points, 1)
        # (bs, 512, np, 1)
        net = F.relu(self.bn5(self.conv5(net)))
        # (bs, 512, np)
        net = net.squeeze(-1)

        src_feat, tgt_feat = torch.chunk(net, 2, dim=0) # (B, 512, num_points)
        #return net
        return src_feat, tgt_feat


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + (weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)



class MVCNN(nn.Module):

    def __init__(self, args):
        super(MVCNN, self).__init__()   
        self.net = models.resnet18()
        self.net = drop_last_layer(self.net)
        self.fc = nn.Sequential(nn.Linear(512, 40))
            

    def forward(self, views, fc_only=False):
        if fc_only:
            pc_feature = views
            logit = self.fc(pc_feature)
            out = {'logit': logit}
            return out

        x = views.float()  # [B, V, C, W, H]
        b, v = x.shape[:2]
        x = x.flatten(0, 1)  # [B*V, C, W, H]
        x = x.cuda()

        #print(x.shape)
        x = self.net(x)
        
        x = x.view(b, -1)
        x = x.view(b, v, -1)  # [B, V, D]
        x = x.max(1)[0]  # [B, 512]
        # logit = self.fc(x)  # [B, C]
        # out = {'logit': logit}

        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))
    

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
    

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.n_ff_dims = args.n_ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout),
                                            self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding
    

class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()


class CMIGNet(nn.Module):
    def __init__(self, emb_nn, args):
        super(CMIGNet, self).__init__()
        self.emb_dims = args.emb_dims  # 64
        self.num_iter = args.num_iter  # 4
        self.emb_nn = emb_nn  # GNN,[B, 64, N]
        self.K_test = args.K_test
        self.overlap_cl = args.overlap_cl
        self.significance_fc = Conv1DBlock((self.emb_dims, 256, 128, 64, 32, 1), 1)
        self.head = SVDHead(args=args)
        self.sim_mat_Fconv = nn.ModuleList(
            [Conv2DBlock((self.emb_dims + 1, 256, 128, 64, 32, 32, 32, 1), 1) for _ in range(self.num_iter)])
        self.sim_mat_Sconv = nn.ModuleList(
            [Conv2DBlock((7, 32, 32, 32, 1), 1) for _ in range(self.num_iter)])

        self.weight_fc = nn.ModuleList([Conv1DBlock((32, 32, 1), 1) for _ in range(self.num_iter)])
        self.preweight = nn.ModuleList([Conv2DBlock((2, 32), 1) for _ in range(self.num_iter)])

        self.MVCNN = MVCNN(args)
        self.attn = Transformer(args)
        self.pooling = Pooling()
        self.h3 = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.ReLU(),
								nn.Conv1d(512, 256, 1), nn.ReLU(),
								nn.Conv1d(256, 128, 1), nn.ReLU(),
								nn.Conv1d(128,   1, 1), nn.Sigmoid())


    def find_mask(self, x, t_out_h1):
        batch_size, _ , num_points = t_out_h1.size()
        x = x.unsqueeze(2)
        x = x.repeat(1,1,num_points)
        x = torch.cat([t_out_h1, x], dim=1)
        x = self.h3(x)
        return x.view(batch_size, -1)
    

    def forward(self, src, tgt, views, R_gt=None, t_gt=None):
        """
        :param src: [B, 3, 768]
        :param tgt: [B, 3, 768]
        :param R_gt: [B, 3, 3]
        :param t_gt: [B, 3]
        :return:
        """
             
        ##### only pass ground truth while training #####
        if not (self.training or (R_gt is None and t_gt is None)):
            raise Exception('Passing ground truth while testing')

        ##### getting ground truth correspondences #####
        if self.training:
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)  # [B, 3, 768, 768]
            min_dist, min_idx = (dist ** 2).sum(1).min(-1) # [B, npoint], [B, npoint]
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy() # drop to cpu for numpy
            match_labels = (min_dist < 0.05).float()  # [B, 768]
            indicator = match_labels.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)  # [B, N]
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)  # get the proability whether the point in src has a correspondences in tgt


        ##### Keypoints' Feature Extraction #####
        enc_input = torch.cat((src.detach(), tgt), dim=0)  # (2B, dim, num)
        src_embedding,tgt_embedding = self.emb_nn(enc_input)  # [B, 512, N]

        batch_size = src_embedding.size(0)
        src_embedding_maxpool = F.adaptive_max_pool1d(src_embedding, 1).view(batch_size, -1) # [B, 512]
        tgt_embedding_maxpool = F.adaptive_max_pool1d(tgt_embedding, 1).view(batch_size, -1) # [B, 512]
        
        views_embedding = self.MVCNN(views) # [B, 512]
        views_embedding_transformer = views_embedding.unsqueeze(2).repeat(1,1, src_embedding.size(2)) # [B, 512, N]
        
                        
        # cross-modal contrastive learning
        device = torch.device("cuda")
        criterion = NTXentLoss(temperature = 0.1).to(device)
        loss = 0.

        point_embedding_maxpool_avg = torch.stack([src_embedding_maxpool,tgt_embedding_maxpool]).mean(dim=0) 
        loss_cmid = criterion(point_embedding_maxpool_avg, views_embedding)

        loss = loss + loss_cmid


        if self.overlap_cl:
            threshold = 0.01
            mp = 70
            mn = 3
            if self.training:
                src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1) # [B, 3, 768]
                batchsize = src_gt.size(0)
                
                loss_overlap = 0
                for i in range(batchsize):
                    
                    src_gt_i = src_gt[i].transpose(0,1) # [768, 3]
                    tgt_i = tgt[i].transpose(0,1) # [768, 3]
                    
                    dist = torch.cdist(src_gt_i, tgt_i) # [768, 768]
                    overlap_mask = dist < threshold # [768, 768]
                
                    overlap_indices_X_i = torch.nonzero(overlap_mask.sum(dim=1) > 0).squeeze(dim=1) # [N]
                    overlap_indices_Y_i = torch.nonzero(overlap_mask.sum(dim=0) > 0).squeeze(dim=1) # [M]
                    non_overlap_indices_X_i = torch.nonzero(overlap_mask.sum(dim=1) == 0).squeeze(dim=1) 
                    non_overlap_indices_Y_i = torch.nonzero(overlap_mask.sum(dim=0) == 0).squeeze(dim=1) 
                    
                    src_embedding_overlap = src_embedding[i,:,overlap_indices_X_i].permute(1, 0) # [N, 64]
                    tgt_embedding_overlap = tgt_embedding[i,:,overlap_indices_Y_i].permute(1, 0) # [M, 64]
                    src_embedding_non_overlap = src_embedding[i,:,non_overlap_indices_X_i].permute(1, 0) 
                    tgt_embedding_non_overlap = tgt_embedding[i,:,non_overlap_indices_Y_i].permute(1, 0) 
                    
                    pos_dist = pdist(src_embedding_overlap, tgt_embedding_overlap) # [N, M]
                    neg1_dist = pdist(src_embedding_overlap, tgt_embedding_non_overlap) # [768-N, M]
                    neg2_dist = pdist(tgt_embedding_overlap, src_embedding_non_overlap) # [768-N, 768-M]
                    
                    mask_pos = pos_dist > mp
                    mask_neg1 = neg1_dist < mn
                    mask_neg2 = neg2_dist < mn
                    
                    loss_pos = (F.relu(pos_dist[mask_pos] - mp)).pow(2).mean()
                    loss_neg1 = (F.relu(mn - neg1_dist[mask_neg1])).pow(2).mean()
                    loss_neg2 = (F.relu(mn - neg2_dist[mask_neg2])).pow(2).mean()
                    
                    if torch.isnan(loss_pos):
                        loss_pos = torch.tensor(0.0, requires_grad=True, device='cuda:0')
                                        
                    if torch.isnan(loss_neg1):
                        loss_neg1 = torch.tensor(0.0, requires_grad=True, device='cuda:0')
                        
                    if torch.isnan(loss_neg2):
                        loss_neg2 = torch.tensor(0.0, requires_grad=True, device='cuda:0')
                                
                    loss_overlap += loss_pos + loss_neg1 + loss_neg2
                       
                loss_overlap = loss_overlap/batchsize
                loss = loss + loss_overlap      
        

        # Transformer Fusion
        src_embedding_p, tgt_embedding_p = self.attn(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p  # [B, 512, N]
        tgt_embedding = tgt_embedding + tgt_embedding_p

        src_embedding_p2, _ = self.attn(src_embedding, views_embedding_transformer)
        src_embedding = src_embedding + src_embedding_p2
        tgt_embedding_p2, _ = self.attn(tgt_embedding, views_embedding_transformer)
        tgt_embedding = tgt_embedding + tgt_embedding_p2


        src_features_pooling = self.pooling(src_embedding) # (B,512)
        tgt_features_pooling = self.pooling(tgt_embedding)
        src_sig_score = self.find_mask(src_features_pooling, tgt_embedding) # (B,768)
        tgt_sig_score = self.find_mask(tgt_features_pooling, src_embedding)

        
        num_point_preserved = 256

        if not self.training:
            num_point_preserved = self.K_test

        if self.training:
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_point_preserved//2, p=pos_probs)
            neg_idx = batch_choice(candidates, num_point_preserved-num_point_preserved//2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        else:
            src_idx = src_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()   # (batchsize,num_point_preserved)
            tgt_idx = tgt_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy() 


        batch_idx = np.arange(src.size(0))[:, np.newaxis]
        if self.training:
            match_labels = match_labels[batch_idx, src_idx]
        src = src[batch_idx, :, src_idx].transpose(1, 2)  # [B, 3, N]
        src_embedding = src_embedding[batch_idx, :, src_idx].transpose(1, 2)  # [B, 3, C]
        src_sig_score = src_sig_score[batch_idx, src_idx]
        tgt = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_embedding = tgt_embedding[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_sig_score = tgt_sig_score[batch_idx, tgt_idx]

        ##### transformation initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        #loss = 0.

        for i in range(self.num_iter):
            batch_size, num_dims, num_points = src_embedding.size()
            _src_emb = src_embedding.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt_emb = tgt_embedding.unsqueeze(-2).repeat(1, 1, num_points, 1)  # [B, C, N, N]

            #### Feature Matching Matrix Computation ####
            diff_f = _tgt_emb - _src_emb
            dist_f = torch.sqrt((diff_f ** 2).sum(1, keepdim=True))
            diff_f = diff_f / (dist_f + 1e-8)
            similarity_matrix_F = torch.cat([dist_f, diff_f], 1)
            similarity_matrix_F = self.sim_mat_Fconv[i](similarity_matrix_F)  # [B, 1, N, N]

            ##### Coordinate Matching Matrix Computation #####
            diff = src.unsqueeze(-1) - tgt.unsqueeze(-2)
            dist = (diff ** 2).sum(1, keepdim=True)
            dist = torch.sqrt(dist)
            diff = diff / (dist + 1e-8)
            similarity_matrix_S = torch.cat([dist, diff, src.unsqueeze(-1).repeat(1, 1, 1, tgt.size(2))], 1)  # [B, 7, N, N]
            similarity_matrix_S = self.sim_mat_Sconv[i](similarity_matrix_S)  # [B, 1, N, N]

            ##### Final Matching Matrix Computation #####
            similarity_matrix = similarity_matrix_F + similarity_matrix_S

            ##### Correspondences Credibility Computation #####
            preweight = torch.cat([similarity_matrix_F, similarity_matrix_S], 1) # [B, 2, N, N]
            preweight = self.preweight[i](preweight)
            weights = preweight.max(-1)[0]
            weights = self.weight_fc[i](weights).squeeze(1)  # [B, N]

            ##### Obtain  Final Matching Matrix #####
            similarity_matrix = similarity_matrix.squeeze(1)
            similarity_matrix = similarity_matrix.clamp(min=-20, max=20)  # [B, N, N] in -20 ~ 20
            ##### similarity matrix convolution to get similarities #####

            ###############################################      Loss     ################################################################
            ##### keypoints selection loss #####
            if self.training and i==0:
                src_neg_ent = torch.softmax(similarity_matrix.squeeze(1), dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(similarity_matrix.squeeze(1), dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                loss = loss + 0.5*(F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score, tgt_neg_ent.detach()))
                #print("keypoints selection loss",F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score, tgt_neg_ent.detach()))

            ###### correspondence matching loss #####
            if self.training:
                temp = torch.softmax(similarity_matrix, dim=-1)  # [B, N. N]
                temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
                temp = - torch.log(temp)
                match_loss = (temp * match_labels).sum() / match_labels.sum()
                loss = loss +  match_loss
                #print("match_loss:",match_loss)

            ##### finding correspondences #####
            corr_idx = similarity_matrix.max(-1)[1]
            src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)

            ##### correspondences credibility computation loss #####
            if self.training:
                weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).cuda().unsqueeze(0)).float()
                weight_loss = F.binary_cross_entropy_with_logits(weights, weight_labels)
                loss = loss + weight_loss


            ##### Unreliable correspondence elimination #####
            weights = torch.sigmoid(weights)  # [B,N]
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

            ##### get R and t #####
            rotation_ab, translation_ab = self.head(src, src_corr, weights)
            rotation_ab = rotation_ab.detach() # prevent backprop through svd
            translation_ab = translation_ab.detach() # prevent backprop through svd
            src = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(-1)
            R = torch.matmul(rotation_ab, R)
            t = torch.matmul(rotation_ab, t.unsqueeze(-1)).squeeze() + translation_ab

            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)


        return R, t, loss, rotation_ab, translation_ab, rotation_ba, translation_ba