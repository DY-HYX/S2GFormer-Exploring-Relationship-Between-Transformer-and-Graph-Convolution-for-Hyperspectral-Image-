import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from einops import rearrange, repeat


def gain_neighborhood_band(x_train, band, band_patch, patch_all):
    nn = band_patch // 2
    pp = (patch_all) // 2
    x_train_band = torch.zeros((x_train.shape[0], patch_all*band_patch, band),dtype=float)#64*27*200
    # 中心区域
    x_train_band[:,nn*patch_all:(nn+1)*patch_all,:] = x_train
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch_all:(i+1)*patch_all,:i+1] = x_train[:,:,band-i-1:]
            x_train_band[:,i*patch_all:(i+1)*patch_all,i+1:] = x_train[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch_all:(nn+i+2)*patch_all,:band-i-1] = x_train[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch_all:(nn+i+2)*patch_all,band-i-1:] = x_train[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train[:,0:1,(i+1):]
    return x_train_band


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_token, heads, dim_head, dropout,dis,D2,edge):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dis = dis
        self.D2 = D2
        self.edge=edge
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.degree_encoder = nn.Embedding(10, dim, padding_idx=0)#根据度矩阵排序 0-80，然后映射 ，维度和token相同
        self.spatial_pos_encoder = nn.Embedding(8, heads, padding_idx=0)#划定5*5的区域计算欧式距离然后映射，维度和head相同
        self.edge_dis_encoder = nn.Embedding(4, heads, padding_idx=0)#将每个边都进行编码，生成对应的权重，然后利用生成的权重乘以距离
        self.edge_weight = nn.Embedding(4, heads, padding_idx=0)  # 将每个边都进行编码，生成对应的权重，然后利用生成的权重乘以距离
    def forward(self, x, degree,mask = None):
        b, n, _, h = *x.shape, self.heads
        # edge= torch.tensor([1,2,3,4,5,6,7,8]).cuda()
        # edge = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        edge = torch.tensor([0, 1, 2, 3]).cuda()
        # edge = torch.tensor([0, 1, 2, 3]).cuda()
        # c = self.degree_encoder(self.D2)  # 中心编码
        # cc=self.degree_encoder(self.D2).unsqueeze(0)#中心编码
        # x=x+self.degree_encoder( .D2)#中心编码
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale#[64,4,81,81]
        spatial_pos_encoder=self.spatial_pos_encoder(self.dis).unsqueeze(0).permute(0, 3, 1, 2)
        edge_dis_encoder=torch.mul(self.edge_dis_encoder(edge),self.edge_weight(edge))#8,8,4
        edge_dis_encoder=torch.matmul(self.edge,edge_dis_encoder )
        edge_dis_encoder=edge_dis_encoder.unsqueeze(0).permute(0, 3, 1, 2)
        # dots = dots + spatial_pos_encoder#距离编码
        # dots = dots + edge_dis_encoder  # 边编码
        mask_value = -torch.finfo(dots.dtype).max
        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_token, mode, dis,D2,edge):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim,num_token, heads = heads, dim_head = dim_head, dropout = dropout, dis=dis,D2=D2,edge=edge))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_token, num_token, [1, 2], 1, 0))

    def forward(self, x, degree, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, degree=degree, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, degree=degree, mask = mask)
                x = ff(x)
                nl += 1

        return x

class ViT(nn.Module):
    def __init__(self, band, num_token, num_classes, dim, depth, heads, mlp_dim, dis,D2,edge, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        self.num_classes=num_classes
        self.pos_embedding = nn.Parameter(torch.randn(1, num_token, dim))#1,201,64
        self.patch_to_embedding = nn.Linear(band, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_token, mode, dis,D2,edge)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, center_pos, degree,mask = None):

        x=x.to(torch.float32)
        x = self.patch_to_embedding(x)
        batch, n, _ = x.shape
        pos=self.pos_embedding[:, :n]
        x += pos
        x = self.dropout(x)
        x = self.transformer(x, degree, mask)
        x = self.mlp_head(x) #[64,81,16]
        x_out=torch.zeros((batch, self.num_classes),dtype=float).to(device)
        for i in range(batch):
            x_out[i]=x[i,center_pos[i],:]
        return x_out


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
 # 这个函数主要是为了生成对角线全1，其余部分全0的二维数组

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(2)
        batch,l=D.shape
        D1=torch.reshape(D, (batch * l,1))
        D1=D1.squeeze(1)
        D2=torch.pow(D1, -0.5)
        D2=torch.reshape(D2,(batch,l))
        D_hat=torch.zeros([batch,l,l],dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cuda()

    def forward(self, H, A ):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # 方案一：一阶切比雪夫
        (batch, l, c) = H.shape
        H1 = torch.reshape(H,(batch*l, c)) 
        H2 = self.BN(H1)
        H=torch.reshape(H2,(batch,l, c)) 
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))#点乘
        A_hat = I + A_hat
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))#矩阵相乘
        output = self.Activition(output)
        return output
        # 方案一：二阶切比雪夫
        # H = H.to(torch.float16)
        # H = self.BN(H).to(torch.float16)
        # A1 = self.A1.to(torch.float16)
        # A2 = self.A2.to(torch.float16)
        # D1_hat = self.A_to_D_inv(A1).to(torch.float16)
        # A1_hat = torch.matmul(D1_hat, torch.matmul(A1, D1_hat))  # 点乘
        # M = self.I + A1_hat + torch.matmul(A1_hat.to(torch.float16), A1_hat.to(torch.float16))
        # W = math.exp(-1) / (math.exp(-1) + math.exp(-4)) * A1 + math.exp(-4) / (math.exp(-1) + math.exp(-4)) * (
        #             A2 - A1) + self.I
        # M = M.mul(W)  # 逐点相乘
        # output = torch.mm(M.to(torch.float16), self.GCN_liner_out_1(H.to(torch.float32)).to(torch.float16))  # 矩阵相乘
        # output = self.Activition(output)
        # return output, A1

