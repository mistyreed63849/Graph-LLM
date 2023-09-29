import torch
from typing import Optional, Tuple
from torch_geometric.utils import mask_to_index, index_to_mask
from torch_scatter import scatter
import torch
from torch_sparse import SparseTensor

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_single(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(-2)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def batch_subgraph_graph_level(edge_index,
                               node_ids,
                               num_nodes,
                               graph_size=20,
                               ):

    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    row, col = edge_index
    inc_num = 0
    batch_id = 0

    for node_idx in node_ids:

        subset = torch.arange(node_idx, node_idx+graph_size).to(node_idx.device)
        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # Relabel Node
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=edge_index.device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx[edge_index_sub]

        # Batching Graph
        edge_index_sub += inc_num

        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num)
        batch_list.extend([batch_id for _ in range(len(subset))])

        inc_num += len(subset)
        batch_id += 1


    subset, mapping, batch = torch.cat(subset_list), torch.as_tensor(mapping_list), torch.as_tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch




def batch_subgraph_pair_level(edge_index,
                               node_ids,
                               num_nodes,
                               graph_size=20,
                               ):

    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    row, col = edge_index
    inc_num = 0
    batch_id = 0

    for node_idx in node_ids:

        subset = torch.arange(node_idx, node_idx+graph_size).to(node_idx.device)
        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # Relabel Node
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=edge_index.device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx[edge_index_sub]

        # Batching Graph
        edge_index_sub += inc_num

        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num)
        batch_list.extend([batch_id for _ in range(len(subset))])

        inc_num += len(subset)
        batch_id += 1


    subset, mapping, batch = torch.cat(subset_list), torch.as_tensor(mapping_list), torch.as_tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch






def batch_subgraph(edge_index,
                   node_ids,
                   num_nodes,
                   num_hops = 3,
                   fans_out = (50, 50, 50)
                   ):

    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    row, col = edge_index
    inc_num = 0
    batch_id = 0

    for node_idx in node_ids:
        subsets = [node_idx.flatten()]
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)

        for _ in range(num_hops):
            node_mask.fill_(False)

            node_mask[subsets[-1]] = True
            edge_mask = torch.index_select(node_mask, 0, row)

            neighbors = col[edge_mask]
            if len(neighbors) > fans_out[_]:
                perm = torch.randperm(len(neighbors))[:fans_out[_]]
                neighbors = neighbors[perm]

            subsets.append(neighbors)

        subset, ind = torch.unique(torch.cat(subsets), return_inverse=True)

        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # Relabel Node
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=edge_index.device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx[edge_index_sub]

        # Batching Graph
        edge_index_sub += inc_num

        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num + ind[0].item())
        batch_list.extend([batch_id for _ in range(len(subset))])

        inc_num += len(subset)
        batch_id += 1


    subset, mapping, batch = torch.cat(subset_list), torch.as_tensor(mapping_list), torch.as_tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch


def padding_to_full_batch_edge_index(edge_index, batch):
    ## Step 1. batch complete edge index.
    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full



@torch.no_grad()
def add_full_rrwp(edge_index, num_nodes, walk_length=8, add_identity=True):
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes), )
    # Compute D^{-1} A:
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=edge_index.device))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    rrwp_index = rel_pe_idx
    rrwp_val = rel_pe_val

    return rrwp_index, rrwp_val