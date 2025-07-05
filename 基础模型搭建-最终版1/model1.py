import math
import torch
from torch import nn
import torch.nn.functional as F
from SFC_GAT import SFC_GAT
from SFC_GCN import SFC_GCN


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

        # 2. Top-K 不同区域聚合子模块
        self.topk_module = CRCVA(embedding_size, topk)

        # 3. 融合分支输出映射 (各视图一个线性层)
        # self.W_align_out = nn.ModuleList([
        #     nn.Linear(embedding_size, embedding_size, bias=False)
        #     for _ in range(self.V)
        # ])
        # self.W_nbr_out = nn.ModuleList([
        #     nn.Linear(embedding_size, embedding_size, bias=False)
        #     for _ in range(self.V)
        # ])

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

        # ── 1. 调用同区域跨视图对齐模块 ───────────────────
        n_align, poi_align, s_align, d_align = self.align_module(
            n_emb, poi_emb, s_emb, d_emb
        )
        aligned_list = [n_align, poi_align, s_align, d_align]

        # ── 2. 调用 Top-K 不同区域聚合模块 ───────────
        nbr_msgs_list = self.topk_module(aligned_list, C_list)
        # 返回列表长度 4，每项 (N, d)

        # ── 3. 融合与插值 ───────────────────────────
        H_final_list = []
        # a_align = torch.clamp(self.alpha_align, 0.0, 1.0)
        a_align = torch.sigmoid(self.alpha_align)  # 始终 ∈ (0,1)
        a_nbr = 1.0 - a_align

        for p in range(self.V):
            # 分支输出映射
            # f_align = self.W_align_out[p](aligned_list[p])   # (N, d)
            # f_nbr   = self.W_nbr_out[p](nbr_msgs_list[p])    # (N, d)
            f_align = aligned_list[p]   # (N, d)
            f_nbr   = nbr_msgs_list[p]    # (N, d)

            # 融合
            f_fuse = F.relu(a_align * f_align + a_nbr * f_nbr)  # (N, d)

            # 与原始 GAT 嵌入插值
            Hp_orig = H_list[p]  # (N, d)
            # beta = torch.clamp(self.beta, 0.0, 1.0)
            beta = torch.sigmoid(self.beta)
            Hp_final = beta * Hp_orig + (1.0 - beta) * f_fuse  # (N, d)

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



class Model(nn.Module):
    def __init__(self, S_dict, C_list, embedding_size, gcn_layers, dropout, TopK, gamma):
        super(Model, self).__init__()
        # 空间功能一致性得分
        self.S_dict = S_dict
        # 跨视图区域相关性得分
        self.C_list = C_list
        # TopK
        self.TopK = TopK

        # 1. 实例化各视图对应的SFC_GCN
        self.sfc_gcns = SFC_GCN(embedding_size, dropout, gcn_layers, gamma)

        # 2. 跨视图交互模块，包含 AlignModule 和 TopKModule
        self.cross_view_interaction = CrossViewInteraction(embedding_size, self.TopK)

        # 3. 四视图融合模块（如 Gated Multimodal Unit 融合）
        self.four_view_fusion = FourViewGMUFusion(embedding_size)

        # 4. 四个解码器，将融合嵌入映射回各视图嵌入，用于重建
        self.decoder_s = nn.Linear(embedding_size, embedding_size)  # 解码到视图 S 的嵌入
        self.decoder_d = nn.Linear(embedding_size, embedding_size)  # 解码到视图 D 的嵌入
        self.decoder_p = nn.Linear(embedding_size, embedding_size)  # 解码到视图 P 的嵌入
        self.decoder_n = nn.Linear(embedding_size, embedding_size)  # 解码到视图 N 的嵌入


    def forward(self, feature, edge_index, S):
        # SFC_GCN
        n_emb, poi_emb, s_emb, d_emb = self.sfc_gcns(feature, edge_index, S, is_training=True)

        # 2.跨视图交互
        # CrossViewInteraction 内部应分别处理 AlignModule 和 TopKModule
        n_f, poi_f, s_f, d_f = self.cross_view_interaction([n_emb, poi_emb, s_emb, d_emb], self.C_list)

        # 3. 四视图融合
        fused_emb = self.four_view_fusion([n_f, poi_f, s_f, d_f])

        # 4. 解码重建：从融合嵌入重建回各视图的嵌入表示
        recon_s = self.decoder_s(fused_emb)  # 重建视图s嵌入 (N, d)
        recon_d = self.decoder_d(fused_emb)  # 重建视图d嵌入 (N, d)
        recon_p = self.decoder_p(fused_emb)  # 重建视图p嵌入 (N, d)
        recon_n = self.decoder_n(fused_emb)  # 重建视图n嵌入 (N, d)

        return fused_emb, recon_n, recon_p, recon_s, recon_d
























