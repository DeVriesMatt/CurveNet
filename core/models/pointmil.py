import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(K, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, K]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, K, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class TransformerBlock(nn.Module):
    def __init__(self, C_in, C_out, n_samples=None, K=20, dim_k=32, heads=8, ch_raise=64, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.d = dim_k
        assert (C_out % heads) == 0, 'values dimension must be integer'
        dim_v = C_out // heads

        self.n_samples = n_samples
        self.K = K
        self.heads = heads

        C_in = C_in * 2 + dim_v
        self.mlp = nn.Sequential(
            nn.Conv2d(C_in, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True),
            nn.Conv2d(ch_raise, ch_raise, 1, bias=False),
            nn.BatchNorm2d(ch_raise),
            nn.ReLU(True))

        self.mlp_v = nn.Conv1d(C_in, dim_v, 1, bias=False)
        self.mlp_k = nn.Conv1d(C_in, dim_k, 1, bias=False)
        self.mlp_q = nn.Conv1d(ch_raise, heads * dim_k, 1, bias=False)
        self.mlp_h = nn.Conv2d(3, dim_v, 1, bias=False)

        self.bn_value = nn.BatchNorm1d(dim_v)
        self.bn_query = nn.BatchNorm1d(heads * dim_k)

    def forward(self, x):
        xyz = x[..., :3]
        if not self.use_norm:
            feature = xyz
        else:
            feature = x[..., 3:]

        bs = xyz.shape[0]

        knn_idx = knn_point(self.K, xyz, xyz)  # [B, S, K]
        neighbor_xyz = index_points(xyz, knn_idx)  # [B, S, K, 3]
        grouped_features = index_points(feature, knn_idx)  # [B, S, K, C]
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # [B, C, S, K]
        grouped_points_norm = grouped_features - feature.transpose(2, 1).unsqueeze(-1).contiguous()  # [B, C, S, K]
        # relative spatial coordinates
        relative_pos = neighbor_xyz - xyz.unsqueeze(-2).repeat(1, 1, self.K, 1)  # [B, S, K, 3]
        relative_pos = relative_pos.permute(0, 3, 1, 2).contiguous()  # [B, 3, S, K]

        pos_encoder = self.mlp_h(relative_pos)
        feature = torch.cat([grouped_points_norm,
                             feature.transpose(2, 1).unsqueeze(-1).expand(-1, -1, -1, self.K),
                             pos_encoder], dim=1)  # [B, 2C_in + d, S, K]

        feature_q = self.mlp(feature).max(-1)[0]  # [B, C, S]
        query = F.relu(self.bn_query(self.mlp_q(feature_q)))  # [B, head * d, S]
        query = rearrange(query, 'b (h d) n -> b h d n', b=bs, h=self.heads, d=self.d)  # [B, head, d, S]

        feature = feature.permute(0, 2, 1, 3).contiguous()  # [B, S, 2C, K]
        feature = feature.view(bs * self.n_samples, -1, self.K)  # [B*S, 2C, K]
        value = self.bn_value(self.mlp_v(feature))  # [B*S, v, K]
        value = value.view(bs, self.n_samples, -1, self.K)  # [B, S, v, K]
        key = self.mlp_k(feature).softmax(dim=-1)  # [B*S, d, K]
        key = key.view(bs, self.n_samples, -1, self.K)  # [B, S, d, K]
        k_v_attn = einsum('b n d k, b n v k -> b d v n', key, value)  # [bs, d, v, N]
        out = einsum('b h d n, b d v n -> b h v n', query, k_v_attn.contiguous())  # [B, S, head, v]
        out = rearrange(out.contiguous(), 'b h v n -> b (h v) n')  # [B, C_out, S]

        return xyz, out


class MedPTFeatureExtractor(nn.Module):
    def __init__(self, trans_block=TransformerBlock,
                 output_channels=2,
                 use_norm=True,
                 num_K=None,
                 dropout=0.1,
                 dim_k=32,
                 head=8,
                 channel_dim=None,
                 num_points=1024,
                 channel_raise=None):
        super().__init__()
        if channel_raise is None:
            channel_raise = [64, 256]
        if num_K is None:
            num_K = [32, 64]
        if channel_dim is None:
            channel_dim = [128, 256]
        self.output_channels = output_channels
        self.use_norm = use_norm
        self.num_K = num_K
        self.dim_k = dim_k
        self.head = head
        self.channel_dim = channel_dim
        self.num_points = num_points
        self.channel_raise = channel_raise
        self.dropout = dropout
        self.non_linear_cls = True

        # transformer layer
        self.tf1 = trans_block(3,
                               channel_dim[0],
                               n_samples=self.num_points,
                               K=num_K[0],
                               dim_k=dim_k,
                               heads=head,
                               ch_raise=channel_raise[0],
                               use_norm=self.use_norm)
        self.tf2 = trans_block(channel_dim[0],
                               channel_dim[1],
                               n_samples=self.num_points,
                               K=num_K[1],
                               dim_k=dim_k,
                               heads=head,
                               ch_raise=channel_raise[1],
                               use_norm=True)
    def forward(self, x):

        xyz, feature = self.tf1(x)
        xyz, feature = self.tf2(torch.cat([xyz, feature.transpose(2, 1)], dim=2))

        return xyz, feature


import torch
from torch import nn
import math

class InstancePooling(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes=2,
                 dropout=0.5,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(InstancePooling, self).__init__()
        self.non_linear = non_linear
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)


        if self.non_linear:
            self.node_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                # nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                # nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.node_out = nn.Linear(self.num_features, num_classes)

    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        isinstance_logits = self.node_out(features[1].transpose(2, 1))
        bag_logits = torch.mean(isinstance_logits, dim=1)

        return {
            'interpretation': isinstance_logits.transpose(2, 1),
            'bag_logits': bag_logits
        }

class AttentionPooling(nn.Module):
    def __init__(self, num_features,
                 num_classes=2,
                 dropout=0.5,
                 heads=8,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(AttentionPooling, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.non_linear = non_linear
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, heads),
            nn.Tanh(),
            nn.Linear(heads, 1),
            nn.Sigmoid(),
        )

        if self.non_linear:
            self.bag_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.BatchNorm1d(self.num_features * 2),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                nn.BatchNorm1d(self.num_features // 2),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.bag_out = nn.Linear(self.num_features, num_classes)

    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        attn_weights = self.attention_head(features[1].transpose(2, 1))
        bag_embedding = torch.mean(features[1].transpose(2, 1) * attn_weights, dim=1)
        bag_logits = self.bag_out(bag_embedding)

        return {
            'interpretation': attn_weights.repeat(1, 1, self.num_classes).transpose(2, 1),
            'bag_logits': bag_logits
        }

class AdditivePooling(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes=2,
                 dropout=0.5,
                 heads=8,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(AdditivePooling, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.non_linear = non_linear
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)

        if self.non_linear:
            self.node_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                # nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                # nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.node_out = nn.Linear(self.num_features, num_classes)

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, heads),
            nn.Tanh(),
            nn.Linear(heads, 1),
            nn.Sigmoid(),
        )


    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        attn_weights = self.attention_head(features[1].transpose(2, 1))
        weighted_instance_features = features[1].transpose(2, 1) * attn_weights
        instance_logits = self.node_out(weighted_instance_features)
        bag_logits = torch.mean(instance_logits, dim=1)


        return {
            'interpretation': (instance_logits * attn_weights).transpose(2, 1),
            'bag_logits': bag_logits,
            'instance_logits': instance_logits.transpose(2, 1),
            'attention': attn_weights

        }


class ConjunctivePooling(nn.Module):
    def __init__(self, num_features,
                 num_classes=2,
                 dropout=0.5,
                 heads=8,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(ConjunctivePooling, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.non_linear = non_linear
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)

        # self.conv_raise = nn.Sequential(
        #         nn.Conv1d(self.num_features, self.num_features * 2, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(self.num_features * 2),
        #         nn.ReLU(True),
        #         nn.Conv1d(self.num_features * 2, self.num_features * 4, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(self.num_features * 4),
        #         nn.ReLU(True))
        #
        # if self.non_linear:
        #     self.node_out = nn.Sequential(
        #         nn.Linear(self.num_features * 4, self.num_features * 2),
        #         nn.LayerNorm(self.num_features * 2),
        #         nn.ReLU(True),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(self.num_features * 2, self.num_features),
        #         nn.LayerNorm(self.num_features),
        #         nn.ReLU(True),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(self.num_features, num_classes)
        #     )
        if self.non_linear:
            self.node_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                # nn.Dropout(p=dropout),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.node_out = nn.Linear(self.num_features, num_classes)

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, heads),
            nn.Tanh(),
            nn.Linear(heads, 1),
            nn.Sigmoid(),
        )
        # self.attention_head = nn.Sequential(
        #     nn.Linear(num_features * 4, heads),
        #     nn.Tanh(),
        #     nn.Linear(heads, 1),
        #     nn.Sigmoid(),
        # )
    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        # features_raised = self.conv_raise(features[1])
        # print(features_raised.shape)
        attn_weights = self.attention_head(features[1].transpose(2, 1))
        # print(features_raised.shape)

        instance_logits = self.node_out(features[1].transpose(2, 1))
        # print(instance_logits.shape)
        weighted_instance_logits = instance_logits * attn_weights
        bag_logits = torch.mean(weighted_instance_logits, dim=1)

        return {
            'interpretation': (instance_logits * attn_weights).transpose(2, 1),
            'bag_logits': bag_logits,
            'instance_logits': instance_logits.transpose(2, 1),
            'attention': attn_weights
        }



class PositionalEncoding3D(nn.Module):
    """
    A PyTorch module that generates sinusoidal positional encodings for 3D coordinates.
    """

    def __init__(self, num_features: int):
        """
        Initialize the positional encoding module.

        Args:
            num_features: Number of features to encode (should match the original feature dimension).
        """
        super(PositionalEncoding3D, self).__init__()
        self.num_features = num_features
        div_term = torch.exp(torch.arange(0, num_features, 2) * -(math.log(10000.0) / num_features))
        self.register_buffer('div_term', div_term)

    def forward(self, coords: torch.Tensor, features) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings for 3D coordinates.

        Args:
            coords: Tensor of shape (batch_size, num_points, 3) representing (x, y, z).

        Returns:
            Tensor of shape (batch_size, num_points, num_features).
            :param coords:
            :param features:
        """
        batch_size, num_points, _ = coords.shape
        pe = torch.zeros(batch_size, num_points, self.num_features, device=coords.device)

        for i in range(3):  # Loop over x, y, z
            coord = coords[:, :, i].unsqueeze(-1)  # Shape: (batch_size, num_points, 1)
            pe[:, :, 0::2] += torch.sin(coord * self.div_term)
            pe[:, :, 1::2] += torch.cos(coord * self.div_term)



        return pe, features + pe.transpose(2, 1)


class PointMIL(nn.Module):
    def __init__(self,
                 feature_extractor=MedPTFeatureExtractor(use_norm=False),
                 pooling=ConjunctivePooling(num_features=256,
                                          num_classes=40,
                                          apply_pos_encoding=False),
                 ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooling = pooling
    def interpret(self, model_output):
        return model_output['interpretation']
    def forward(self, x):
        # print(x.shape)
        features = self.feature_extractor(x.transpose(2, 1))
        # print(features[1].shape)
        return self.pooling(features)['bag_logits']