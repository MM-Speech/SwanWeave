import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """
    将 logits 投影为双随机矩阵，支持形状：
      - [n, n]
      - [B, n, n]
      - [B, L, n, n]

    最后两维视为矩阵 (行、列)，前面的维度做 batch。
    """
    # 保证非负
    M = torch.exp(logits)

    if M.dim() < 2:
        raise ValueError(f"sinkhorn_knopp expects tensor with at least 2 dims, got shape {M.shape}")

    # 最后两维是 [n, n]，其余当 batch 维度处理
    for _ in range(n_iters):
        # 行归一化：沿着最后一维求和
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        # 列归一化：沿着倒数第二维求和
        M = M / (M.sum(dim=-2, keepdim=True) + eps)

    return M


class MHCResidual(nn.Module):
    """
    简化 + 动态版 mHC 残差:
    - 输入 / 输出: [B, L, C]
    - 将 C 维拆成 n_streams 个 group，当作多条 residual streams:
        C = n_streams * group_dim
    - 残差更新:
        X_mix = H_res @ X          # H_res 为双随机矩阵，可含动态部分
        delta = H_post^T * F(X)    # F(X) 即 sublayer_out
        X_out = X_mix + delta

    H_res, H_post 分为静态 + 动态两部分:
        H_res_logits_total(b,l,:,:) = H_res_logits_static + Dyn_res_logits(b,l,:,:)
        H_post_logits_total(b,l,:)  = H_post_logits_static + Dyn_post_logits(b,l,:)
    最终:
        H_res(b,l,:,:)  = Sinkhorn(H_res_logits_total(b,l,:,:))
        H_post(b,l,:)   = softplus + 归一化
    """

    def __init__(
        self,
        hidden_size: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 6,
        use_dynamic: bool = True,      # 新增参数，默认为 True，兼容旧调用
        rms_norm_cls: nn.Module = None,
    ):
        super().__init__()
        if hidden_size % n_streams != 0:
            raise ValueError(
                f"hidden_size={hidden_size} 不能被 n_streams={n_streams} 整除，"
                f"请调整 config.mhc_num_streams 或 hidden_size"
            )

        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.group_dim = hidden_size // n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.use_dynamic = use_dynamic

        # ---- v1: 静态参数（保留） ----
        # 残差流混合矩阵 H_res 的 logits，初始化为接近单位阵
        init = torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams)
        self.h_res_logits = nn.Parameter(init)          # [n, n]

        # 写入权重 H_post 的 logits，非负 + 归一化
        self.h_post_logits = nn.Parameter(torch.zeros(n_streams))  # [n]

        # ---- v2: 动态部分（新增） ----
        if self.use_dynamic:
            # 用 RMSNorm 归一化输入 residual（x_l）
            self.input_rmsnorm = rms_norm_cls(hidden_size, eps=1e-6)

            # 从归一化后的向量生成动态 logits:
            # 输出维度: n*n (H_res dyn) + n (H_post dyn)
            self.dynamic_proj = nn.Linear(hidden_size, n_streams * n_streams + n_streams, bias=True)

            # 一个可学习的缩放系数，初始较小，避免动态项一开始就太激进
            self.dynamic_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """
        residual:     [B, L, C]  残差输入 (x_l)
        sublayer_out: [B, L, C]  子层输出 (F(x_l))
        返回:         [B, L, C]  mHC 后的输出
        """
        B, L, C = residual.shape
        assert C == self.hidden_size

        # [B, L, C] -> [B, L, n, group_dim]
        X = residual.view(B, L, self.n_streams, self.group_dim)

        n = self.n_streams

        # =========================
        # 1. 计算 H_res
        # =========================
        if self.use_dynamic:
            # ---- v2: 动态部分 ----
            # 1) 对 residual 做 RMSNorm
            x_norm = self.input_rmsnorm(residual)              # [B, L, C]

            # 2) 线性映射到 (n*n + n) 维，并加一个缩放，避免太激进
            dyn = self.dynamic_proj(x_norm) * self.dynamic_scale  # [B, L, n*n + n]

            # 3) 切分成 dyn_res_logits 和 dyn_post_logits
            dyn_res_logits, dyn_post_logits = torch.split(
                dyn,
                [n * n, n],
                dim=-1,
            )
            dyn_res_logits = dyn_res_logits.view(B, L, n, n)  # [B, L, n, n]
            dyn_post_logits = dyn_post_logits.view(B, L, n)   # [B, L, n]

            # 4) 静态 + 动态，得到总 logits
            base_res = self.h_res_logits.view(1, 1, n, n)     # [1, 1, n, n]
            total_res_logits = base_res + dyn_res_logits      # [B, L, n, n]

            base_post = self.h_post_logits.view(1, 1, n)      # [1, 1, n]
            total_post_logits = base_post + dyn_post_logits   # [B, L, n]

            # 5) 对每个 [B, L] 位置分别做 Sinkhorn，得到 H_res[b,l,:,:] 双随机矩阵
            H_res = sinkhorn_knopp(total_res_logits, n_iters=self.sinkhorn_iters)  # [B, L, n, n]

            # 6) H_post 做 softplus + per-token 归一化
            H_post = F.softplus(total_post_logits)                            # [B, L, n] ≥ 0
            H_post = H_post / (H_post.sum(dim=-1, keepdim=True) + 1e-6)       # [B, L, n]
        else:
            # ---- v1: 只有静态 H_res / H_post 的情况（兼容模式）----
            H_res = sinkhorn_knopp(self.h_res_logits, n_iters=self.sinkhorn_iters)  # [n, n]
            H_post_vec = F.softplus(self.h_post_logits)                             # [n]
            H_post_vec = H_post_vec / (H_post_vec.sum() + 1e-6)                     # [n]

        # =========================
        # 2. 应用 H_res / H_post 做残差更新
        # =========================

        if self.use_dynamic:
            # H_res:  [B, L, n, n]
            # X:      [B, L, n, group_dim]
            # X_mix[b,l,i,c] = sum_j H_res[b,l,i,j] * X[b,l,j,c]
            X_mix = torch.einsum("blij,bljc->blic", H_res, X)      # [B, L, n, group_dim]

            # H_post: [B, L, n]
            # sublayer_out: [B, L, C= n*group_dim]
            # 先不分组，用每个 token 的 F(x) 做写入
            # delta[b,l,j,c] = H_post[b,l,j] * sublayer_out[b,l,c]
            delta = torch.einsum("blc,bln->blnc", sublayer_out, H_post)  # [B, L, n, group_dim]
        else:
            # H_res: [n, n]
            X_mix = torch.einsum("ij,bljc->blic", H_res, X)              # [B, L, n, group_dim]

            # H_post_vec: [n]
            delta = torch.einsum("blc,n->blnc", sublayer_out, H_post_vec)  # [B, L, n, group_dim]

        X_out = X_mix + delta  # [B, L, n, group_dim]

        # 还原回 [B, L, C]
        return X_out.view(B, L, C)

