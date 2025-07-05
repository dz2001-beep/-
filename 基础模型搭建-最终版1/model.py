import math
import torch
from torch import nn
import torch.nn.functional as F
from SFC_GAT import SFC_GAT
from SFC_GCN import SFC_GCN

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import DeepMVCLoss  # CVCL 中的簇分配对比损失
import numpy as np


class IRCVA(nn.Module):
    """
    同一区域跨视图对齐分支，使用多头注意力对齐四个视图中相同区域的嵌入。
    输入顺序：[n_emb, poi_emb, s_emb, d_emb]，每个形状 (N, d)。
    输出同样顺序的四个对齐后嵌入 (N, d)。
    """
    def __init__(self, embedding_size):
        super(IRCVA, self).__init__()
        # 各视图插值系数 α
        self.alpha_n   = nn.Parameter(torch.tensor(0.95))
        self.alpha_poi = nn.Parameter(torch.tensor(0.95))
        self.alpha_s   = nn.Parameter(torch.tensor(0.95))
        self.alpha_d   = nn.Parameter(torch.tensor(0.95))

        # 多头注意力 (embed_dim = d, num_heads = 4)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=4, batch_first=False
        )

    def forward(self, n_emb, poi_emb, s_emb, d_emb):
        # 将四个视图的 (N, d) 嵌入 stack 成 (4, N, d)
        stk = torch.stack((n_emb, poi_emb, s_emb, d_emb), dim=0)  # (4, N, d)
        fusion, _ = self.attn(stk, stk, stk)                      # (4, N, d)

        n_f   = fusion[0] * self.alpha_n   + (1 - self.alpha_n)   * n_emb
        poi_f = fusion[1] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        s_f   = fusion[2] * self.alpha_s   + (1 - self.alpha_s)   * s_emb
        d_f   = fusion[3] * self.alpha_d   + (1 - self.alpha_d)   * d_emb

        return n_f, poi_f, s_f, d_f


class CRCVA(nn.Module):
    """
    跨视图不同区域聚合分支：一次性向量化计算 Top-K 注意力。
    输入：
      - aligned_list: list of length 4，四个对齐后嵌入，每项 (N, d)
      - C_list: 4×4 嵌套 list，C_list[p][q] 形状 (N, N)，仅 p≠q 有效
    输出：
      - nbr_msgs_list: list of length 4，每项 (N, d) 聚合后消息
    """
    def __init__(self, embedding_size, topk):
        super(CRCVA, self).__init__()
        self.V = 4
        self.d = embedding_size
        self.topk = topk

        # Q/K/V 投影，各视图一组
        self.WQ = nn.ModuleList([nn.Linear(self.d, self.d, bias=False) for _ in range(self.V)])
        self.WK = nn.ModuleList([nn.Linear(self.d, self.d, bias=False) for _ in range(self.V)])
        self.WV = nn.ModuleList([nn.Linear(self.d, self.d, bias=False) for _ in range(self.V)])

    def forward(self, aligned_list, C_list):
        """
        Args:
          - aligned_list: list 长度=4，每项 (N, d) 对齐后嵌入
          - C_list: 4×4 嵌套 list，每项 (N, N) 的相似度矩阵
        Returns:
          - nbr_msgs_list: list 长度=4，每项 (N, d) 聚合后消息
        """
        V, d, K = self.V, self.d, self.topk
        device = aligned_list[0].device
        N = aligned_list[0].shape[0]

        # 1. 投影 Qn, Kn, Vn: 形状都 (V, N, d)
        Qn = torch.stack([self.WQ[p](aligned_list[p]) for p in range(V)], dim=0)
        Kn = torch.stack([self.WK[p](aligned_list[p]) for p in range(V)], dim=0)
        Vn = torch.stack([self.WV[p](aligned_list[p]) for p in range(V)], dim=0)

        # 2. 构建 C_hat_batch: (V, V, N, N)，仅 p≠q 有 Top-K 权重
        C_hat_batch = torch.zeros((V, V, N, N), device=device)
        for p in range(V):
            for q in range(V):
                if p == q:
                    continue
                Cpq = C_list[p][q].clone()  # (N, N)
                idx = torch.arange(N, device=device)
                # Cpq[idx, idx] = 0  # 对角置零
                topk_vals, topk_idx = torch.topk(Cpq, K, dim=1)  # (N, K)
                sparse = torch.zeros_like(Cpq)
                sparse.scatter_(1, topk_idx, topk_vals)
                row_sum = sparse.sum(dim=1, keepdim=True) + 1e-12
                C_hat_batch[p, q] = sparse / row_sum  # (N, N)

        # 3. 计算 scores_batch: (V, V, N, N)
        Qn_expand = Qn.unsqueeze(1).expand(V, V, N, d)   # (V, V, N, d)
        Kn_expand = Kn.unsqueeze(0).expand(V, V, N, d)   # (V, V, N, d)
        Kn_t = Kn_expand.transpose(2, 3)                 # (V, V, d, N)
        # 批量点积
        scores = torch.matmul(
            Qn_expand.reshape(-1, N, d),
            Kn_t.reshape(-1, d, N)
        ).view(V, V, N, N) / (d ** 0.5)  # (V, V, N, N)

        # 4. mask 非 Top-K 位置为 -1e9
        mask = (C_hat_batch > 0).float()  # (V, V, N, N)
        scores = scores * mask + (1.0 - mask) * (-1e9)

        # 5. Softmax (dim=3): 对每对 (p,q)、每个 i 归一化 j∈TopK
        alpha_nbr = F.softmax(scores, dim=3)  # (V, V, N, N)

        # 6. 聚合 Value：nbr_msgs[p] = ∑_{q≠p} alpha_nbr[p,q] @ Vn[q]
        nbr_msgs = torch.zeros((V, N, d), device=device)
        for q in range(V):
            # alpha_nbr[:, q] 形状 (V, N, N)
            # Vn[q] 形状 (N, d), expand 扩到 (V, N, d)
            Aq = alpha_nbr[:, q]                       # (V, N, N)
            Vq = Vn[q].unsqueeze(0).expand(V, -1, -1)   # (V, N, d)
            nbr_msgs += torch.matmul(Aq, Vq)           # (V, N, d)

        # 拆分成列表返回
        return [nbr_msgs[p] for p in range(V)]


class CrossViewInteraction(nn.Module):
    """
    整体跨视图交互模块：调用 AlignModule 与 TopKModule，然后融合并与原始 GAT 嵌入插值。
    输入：
      - H_list: list 长度=4，对应视图 [n, poi, s, d] 的原始 GAT 嵌入 (N, d)
      - C_list: 4×4 嵌套 list，每项形状 (N, N) 的跨视图相似度矩阵
    输出：
      - H_final_list: list 长度=4，每项 (N, d) 模块最终输出
    """
    def __init__(self, embedding_size, topk):
        super(CrossViewInteraction, self).__init__()
        self.V = 4
        self.d = embedding_size

        # 1. 同区域对齐子模块
        self.align_module = IRCVA(embedding_size)

        # 2. Top-K 跨视图跨区域聚合子模块
        self.topk_module = CRCVA(embedding_size, topk)

        # 4. 分支融合系数 α_align，α_nbr = 1 - α_align
        self.alpha_align = nn.Parameter(torch.tensor(0.5))

        # 5. 插值系数 β
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, H_list, C_list):
        """
        Args:
          - H_list: list of 4, 每项 (N, d) 的原始 GAT 嵌入
          - C_list: nested list 4×4, 每项 (N, N) 的预计算相似度矩阵
        Returns:
          - H_final_list: list of 4, 每项 (N, d) 的最终输出
        """
        n_emb, poi_emb, s_emb, d_emb = H_list
        N = n_emb.shape[0]
        device = n_emb.device

        # ── 1. 同区域跨视图对齐模块 ───────────────────
        n_align, poi_align, s_align, d_align = self.align_module(
            n_emb, poi_emb, s_emb, d_emb
        )
        aligned_list = [n_align, poi_align, s_align, d_align]

        # ── 2. 调用 Top-K 不同区域聚合模块 ───────────
        nbr_msgs_list = self.topk_module(aligned_list, C_list)
        # 返回列表长度 4，每项 (N, d)

        # ── 3. 融合与插值 ───────────────────────────
        H_final_list = []
        a_align = torch.sigmoid(self.alpha_align)  # 始终 ∈ (0,1)
        a_nbr = 1.0 - a_align

        for p in range(self.V):
            f_align = aligned_list[p]   # (N, d)
            f_nbr   = nbr_msgs_list[p]    # (N, d)

            # 融合
            f_fuse = F.relu(a_align * f_align + a_nbr * f_nbr)  # (N, d)

            # 与原始 GAT 嵌入插值
            Hp_orig = H_list[p]  # (N, d)
            Hp_final = self.beta * Hp_orig + (1.0 - self.beta) * f_fuse  # (N, d)

            H_final_list.append(Hp_final)

        return H_final_list

'''多视图门控融合模块'''
class FourViewGMUFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 每个视图的非线性投影
        self.view_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)
        ])
        self.view_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(4)
        ])
        # 门控生成参数
        self.gate_fc = nn.Linear(4 * d_model, 4)
        # self.gate_bias = nn.Parameter(torch.zeros(4))

    def forward(self, h_list):
        # h_list: [n_f, poi_f, s_f, d_f], each (B, d_model)
        z = torch.cat(h_list, dim=-1)                # (B, 4*d_model)
        # gate_logits = self.gate_fc(z) + self.gate_bias # (B, 4)
        gate_logits = self.gate_fc(z) # (B, 4)
        alpha = F.softmax(gate_logits, dim=-1)        # (B, 4)

        vs = []
        for i, hi in enumerate(h_list):
            vi = torch.tanh(self.view_transforms[i](hi) + self.view_biases[i])
            vs.append(vi)                             # 每个 (B, d_model)

        vs_stack = torch.stack(vs, dim=1)             # (B, 4, d_model)
        fused = torch.sum(alpha.unsqueeze(-1) * vs_stack, dim=1)  # (B, d_model)
        return fused
#
# class DeepFc(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DeepFc, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, input_dim * 2),
#             nn.Linear(input_dim * 2, input_dim * 2),
#             nn.LeakyReLU(negative_slope=0.3, inplace=True),
#             nn.Linear(input_dim * 2, output_dim),
#             nn.LeakyReLU(negative_slope=0.3, inplace=True),
#         )
#         self.output = None
#
#     def forward(self, x):
#         output = self.model(x)
#         self.output = output
#         return output
#
#     def out_feature(self):
#         return self.output
#
# class RegionFusionBlock(nn.Module):
#     def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
#         super(RegionFusionBlock, self).__init__()
#         self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout,
#                                                batch_first=True, bias=True)
#         self.linear1 = nn.Linear(input_dim, dim_feedforward)
#         self.linear2 = nn.Linear(dim_feedforward, input_dim)
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.norm2 = nn.LayerNorm(input_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = F.relu
#
#     def forward(self, src, attn_mask=None):
#         # src: (batch_size, num_regions, input_dim)
#         src2, _ = self.self_attn(src, src, src, attn_mask=attn_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.activation(self.linear1(src)))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src
#
# class RegionFusion(nn.Module):
#     def __init__(self, input_dim, num_blocks=3, nhead=4, dropout=0.1, dim_feedforward=2048):
#         super(RegionFusion, self).__init__()
#         self.blocks = nn.ModuleList([
#             RegionFusionBlock(input_dim=input_dim, nhead=nhead,
#                               dropout=dropout, dim_feedforward=dim_feedforward)
#             for _ in range(num_blocks)
#         ])
#         self.fc = DeepFc(input_dim, input_dim)
#
#     def forward(self, x, neighbors_list=None):
#         # x: (1, N, d) or (batch, N, d)
#         batch, N, _ = x.size()
#         attn_mask = None
#         if neighbors_list is not None:
#             mask = torch.ones(N, N, dtype=torch.bool, device=x.device)
#             for i, nbrs in enumerate(neighbors_list):
#                 mask[i, i] = False
#                 for j in nbrs:
#                     mask[i, j] = False
#             attn_mask = mask
#
#         out = x
#         for block in self.blocks:
#             out = block(out, attn_mask=attn_mask)
#         out = out.squeeze(0)
#         out = self.fc(out)
#         return out
#
# class ViewFusion(nn.Module):
#     def __init__(self, emb_dim, out_dim, hidden_dim=None):
#         super(ViewFusion, self).__init__()
#         # Pointwise MLP replacing Conv1d
#         self.W = nn.Linear(emb_dim, out_dim, bias=False)
#         hidden = hidden_dim or out_dim
#         self.f1 = nn.Sequential(
#             nn.Linear(out_dim, hidden),
#             nn.LeakyReLU(negative_slope=0.3, inplace=True),
#             nn.Linear(hidden, 1)
#         )
#         self.f2 = nn.Sequential(
#             nn.Linear(out_dim, hidden),
#             nn.LeakyReLU(negative_slope=0.3, inplace=True),
#             nn.Linear(hidden, 1)
#         )
#         self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)
#
#     def forward(self, src):
#         # src: (batch, emb_dim, seq_len)
#         # permute to (batch, seq_len, emb_dim)
#         x = src.permute(0, 2, 1)
#         # pointwise W
#         seq_fts = self.W(x)  # (batch, seq_len, out_dim)
#         # compute f1 and f2 on each position
#         f_1 = self.f1(seq_fts)  # (batch, seq_len, 1)
#         f_2 = self.f2(seq_fts)  # (batch, seq_len, 1)
#         # logits: broadcast to (batch, seq_len, seq_len)
#         logits = f_1 + f_2.transpose(1, 2)
#         coefs = torch.mean(self.act(logits), dim=-1)
#         coefs = torch.mean(coefs, dim=0)
#         coefs = F.softmax(coefs, dim=-1)
#         return coefs
#
# class Model(nn.Module):
#     def __init__(self, S_dict, C_list, embedding_size, gcn_layers, dropout,
#                  TopK, gamma):
#         super(Model, self).__init__()
#         self.S_dict = S_dict
#         self.C_list = C_list
#         self.TopK = TopK
#         self.sfc_gcns = SFC_GCN(embedding_size, dropout, gcn_layers, gamma)
#         self.cross_view_interaction = CrossViewInteraction(embedding_size, self.TopK)
#         self.viewFusionLayer = ViewFusion(180, embedding_size)
#         self.regionFusionLayer = RegionFusion(embedding_size)
#         self.decoder_s = nn.Linear(embedding_size, embedding_size)
#         self.decoder_d = nn.Linear(embedding_size, embedding_size)
#         self.decoder_p = nn.Linear(embedding_size, embedding_size)
#         self.decoder_n = nn.Linear(embedding_size, embedding_size)
#
#     def forward(self, feature, edge_index, S, neighbors_list=None):
#         # 1. SFC-GCN encoding
#         n_emb, poi_emb, s_emb, d_emb = self.sfc_gcns(feature, edge_index, S, is_training=True)
#         # 2. Cross-view interaction
#         n_f, poi_f, s_f, d_f = self.cross_view_interaction([n_emb, poi_emb, s_emb, d_emb], self.C_list)
#         # 3. View-level fusion
#         out = torch.stack([n_f, poi_f, s_f, d_f], dim=0).transpose(0, 2)
#         coef = self.viewFusionLayer(out)
#         temp_out = coef[0] * n_f + coef[1] * poi_f + coef[2] * s_f + coef[3] * d_f
#         # 4. Region-level fusion with neighbor mask
#         temp_out = temp_out.unsqueeze(0)  # (1, N, d)
#         fused_emb = self.regionFusionLayer(temp_out, neighbors_list=neighbors_list)
#         # 5. Decoding for reconstruction
#         recon_n = self.decoder_n(fused_emb)
#         recon_p = self.decoder_p(fused_emb)
#         recon_s = self.decoder_s(fused_emb)
#         recon_d = self.decoder_d(fused_emb)
#         return fused_emb, recon_n, recon_p, recon_s, recon_d

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output

class RegionFusionBlock(nn.Module):
    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super(RegionFusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout,
                                               batch_first=True, bias=True)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, attn_mask=None):
        # src: (batch_size, num_regions, input_dim)
        # attn_mask: (num_regions, num_regions) boolean mask, True to ignore
        src2, _ = self.self_attn(src, src, src, attn_mask=attn_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RegionFusion(nn.Module):
    def __init__(self, input_dim, num_blocks=3, nhead=4, dropout=0.1, dim_feedforward=2048):
        super(RegionFusion, self).__init__()
        self.blocks = nn.ModuleList([
            RegionFusionBlock(input_dim=input_dim, nhead=nhead,
                              dropout=dropout, dim_feedforward=dim_feedforward)
            for _ in range(num_blocks)
        ])
        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x, neighbors_list=None):
        # x: (1, N, d) or (batch, N, d)
        # neighbors_list: list of lists, neighbors_list[i] gives neighbor indices of region i
        batch, N, _ = x.size()
        attn_mask = None
        if neighbors_list is not None:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device)
            for i, nbrs in enumerate(neighbors_list):
                mask[i, i] = False
                for j in nbrs:
                    mask[i, j] = False
            attn_mask = mask

        out = x
        for block in self.blocks:
            out = block(out, attn_mask=attn_mask)
        out = out.squeeze(0)
        out = self.fc(out)
        return out

class ViewFusion(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(ViewFusion, self).__init__()
        self.W = nn.Conv1d(emb_dim, out_dim, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.f2 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, src):
        seq_fts = self.W(src)
        f_1 = self.f1(seq_fts)
        f_2 = self.f2(seq_fts)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = torch.mean(self.act(logits), dim=-1)
        coefs = torch.mean(coefs, dim=0)
        coefs = F.softmax(coefs, dim=-1)
        return coefs

class Model(nn.Module):
    def __init__(self, S_dict, C_list, embedding_size, gcn_layers, dropout,
                 TopK, gamma):
        super(Model, self).__init__()
        self.S_dict = S_dict
        self.C_list = C_list
        self.TopK = TopK
        self.sfc_gcns = SFC_GCN(embedding_size, dropout, gcn_layers, gamma)
        self.cross_view_interaction = CrossViewInteraction(embedding_size, self.TopK)
        self.viewFusionLayer = ViewFusion(180, embedding_size)
        self.regionFusionLayer = RegionFusion(embedding_size)
        self.decoder_s = nn.Linear(embedding_size, embedding_size)
        self.decoder_d = nn.Linear(embedding_size, embedding_size)
        self.decoder_p = nn.Linear(embedding_size, embedding_size)
        self.decoder_n = nn.Linear(embedding_size, embedding_size)

    def forward(self, feature, edge_index, S, neighbors_list=None):
        # 1. SFC-GCN encoding
        n_emb, poi_emb, s_emb, d_emb = self.sfc_gcns(feature, edge_index, S, is_training=True)
        # 2. Cross-view interaction
        n_f, poi_f, s_f, d_f = self.cross_view_interaction([n_emb, poi_emb, s_emb, d_emb], self.C_list)
        # 3. View-level fusion
        out = torch.stack([n_f, poi_f, s_f, d_f], dim=0).transpose(0, 2)
        coef = self.viewFusionLayer(out)
        temp_out = coef[0] * n_f + coef[1] * poi_f + coef[2] * s_f + coef[3] * d_f
        # 4. Region-level fusion with neighbor mask
        temp_out = temp_out.unsqueeze(0)  # (1, N, d)
        fused_emb = self.regionFusionLayer(temp_out, neighbors_list=neighbors_list)
        # 5. Decoding for reconstruction
        recon_n = self.decoder_n(fused_emb)
        recon_p = self.decoder_p(fused_emb)
        recon_s = self.decoder_s(fused_emb)
        recon_d = self.decoder_d(fused_emb)
        return fused_emb, recon_n, recon_p, recon_s, recon_d