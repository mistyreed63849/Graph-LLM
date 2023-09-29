import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.graphgym.register import *
import opt_einsum as oe

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out



class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self,
                 layer_id,
                 embed_dim,
                 initial_layer=False,
                 num_heads=8,
                 rrwp=8,
                 clamp=5.,
                 dropout=0.):
        super().__init__()

        self.dim = embed_dim
        self.n_heads = num_heads
        self.head_dim = self.dim // self.n_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.rrwp = rrwp
        self.layer_id = layer_id


        self.wq = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=True)
        self.wk = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=False)
        self.wv = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=False)


        self.w_eb = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=True)
        self.w_ew = nn.Linear(self.dim, self.head_dim * self.n_heads, bias=True)


        self.wo = nn.Linear(self.head_dim * self.n_heads, self.dim, bias=False)
        self.weo = nn.Linear(self.head_dim * self.n_heads, self.dim, bias=False)

        self.Aw = nn.Parameter(torch.zeros(self.head_dim, self.n_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        self.node_feed_forward = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.ReLU(),
            nn.Linear(self.dim * 2, self.dim),
        )

        self.edge_feed_forward = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.ReLU(),
            nn.Linear(self.dim * 2, self.dim),
        )

        self.node_feed_forward_norm = nn.LayerNorm(self.dim)
        self.node_attn_norm = nn.LayerNorm(self.dim)

        self.edge_feed_forward_norm = nn.LayerNorm(self.dim)
        self.edge_attn_norm = nn.LayerNorm(self.dim)

        #self.edge_feed_forward_norm = nn.Identity()
        #self.edge_attn_norm = nn.Identity()


    def propagate_attention(self, xq, xk, xv, edge_index, node_pair_emb):
        src = xk[edge_index[0]]      # E x H x out_dim
        dst = xq[edge_index[1]]     # E x H x out_dim
        score = src + dst           # element-wise add

        eb = self.w_eb(node_pair_emb).view(-1, self.n_heads, self.head_dim)
        ew = self.w_ew(node_pair_emb).view(-1, self.n_heads, self.head_dim)


        score = score * ew
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + eb

        score = F.relu(score)
        e_out = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, edge_index[1])  # E x num_heads x 1

        # Aggregate with Attn-Score
        msg = xv[edge_index[0]] * score  # E x num_heads x out_dim
        x_out = torch.zeros_like(xv) # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=x_out, reduce='add')

        return x_out, e_out


    def forward(self, x, edge_index, input_node_pair_embed):

        bsz, dim = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(-1, self.n_heads, self.head_dim)
        xk = xk.view(-1, self.n_heads, self.head_dim)
        xv = xv.view(-1, self.n_heads, self.head_dim)

        x_out, e_out = self.propagate_attention(xq, xk, xv, edge_index, input_node_pair_embed)

        h = x_out.view(x_out.shape[0], -1)

        h = self.wo(h)
        e_out = self.weo(e_out)
        #
        e_out = e_out + input_node_pair_embed
        h = h + x

        e_out = self.edge_attn_norm(e_out)
        h = self.node_attn_norm(h)

        h = h + self.node_feed_forward(h)
        x_out = self.node_feed_forward_norm(h)

        e_out = e_out + self.edge_feed_forward(e_out)
        e_out = self.edge_feed_forward_norm(e_out)

        return x_out, e_out