# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import copy
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field
import math

import torch
import torch_geometric.utils
from torch import nn
import torch.nn.functional as F
from .model_utils import *
from .grit import *
from typing import List, Optional, Tuple, Union
from torch_geometric.nn.pool import global_max_pool, global_mean_pool


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 4096

    adapter_len: int = 0
    adapter_layer: int = 0
    adapter_dim: int = 512
    adapter_n_heads: int = 4


    num_hops: int = 2
    w_adapter: bool = True
    w_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.05
    rrwp: int = 8


    n_decoder_layers: int = 2
    n_mp_layers: int = 2
    n_encoder_layers: int = 2

    # target_modules: Tuple[str] = ('q_proj', 'v_proj')     # Option
    fans_out: Tuple[int] = (50, 50, 50)

    # target_modules: Tuple[str] = ('q_proj', 'v_proj', 'k_proj')     # Option
    # target_modules: Tuple[str] = ('o_proj')     # Option
    target_modules: Tuple[str] = ('down_proj', 'up_proj', 'gate_proj')     # Option
    task_level: str = 'node'



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, lora_r, lora_alpha, lora_dropout=0.05,
    ):
        super().__init__()

        if lora_r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {lora_r} must be less or equal than {min(in_features, out_features)}"
            )
        self.lora_r = lora_r
        self.lora_down = nn.Linear(in_features, lora_r, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
        self.lora_up = nn.Linear(lora_r, out_features, bias=False)
        self.scale = 1. * lora_alpha / lora_r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        x = x.to(self.lora_up.weight.dtype)
        result = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        result = result.to(previous_dtype)
        return result



class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.params = args

        self.w1 = nn.Linear(dim, hidden_dim, bias=False,)    # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False,)    # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False,)    # up_proj

        if self.params.w_lora:
            if 'up_proj' in args.target_modules:
                self.lora_w3 = LoraInjectedLinear(self.w3.in_features, self.w3.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'down_proj' in args.target_modules:
                self.lora_w2 = LoraInjectedLinear(self.w2.in_features, self.w2.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'gate_proj' in args.target_modules:
                self.lora_w1 = LoraInjectedLinear(self.w1.in_features, self.w1.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)




    def forward(self, x):
        up_x = self.w3(x)
        gate_x = self.w1(x)

        if self.params.w_lora:
            if 'up_proj' in self.params.target_modules:
                up_x = up_x + self.lora_w3(x)

            if 'gate_proj' in self.params.target_modules:
                gate_x = gate_x + self.lora_w1(x)

        down_input = F.silu(gate_x) * up_x
        out = self.w2(down_input)

        if self.params.w_lora:
            if 'down_proj' in self.params.target_modules:
                out = out + self.lora_w2(down_input)

        return out



class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None,
            freqs_cis_prefix=None
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, freqs_cis_prefix)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False,)

        self.w_lora = args.w_lora
        self.target_modules = args.target_modules


        if self.w_lora:
            if 'q_proj' in args.target_modules:
                self.lora_wq = LoraInjectedLinear(self.wq.in_features, self.wq.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'k_proj' in args.target_modules:
                self.lora_wk = LoraInjectedLinear(self.wk.in_features, self.wk.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

            if 'v_proj' in args.target_modules:
                self.lora_wv = LoraInjectedLinear(self.wv.in_features, self.wv.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

            if 'o_proj' in args.target_modules:
                self.lora_wo = LoraInjectedLinear(self.wo.in_features, self.wo.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)


        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None



    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None


    def forward(
            self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor], adapter=None, freqs_cis_prefix=None
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.w_lora:
            if 'q_proj' in self.target_modules:
                xq = xq + self.lora_wq(x)
            if 'k_proj' in self.target_modules:
                xk = xk + self.lora_wk(x)
            if 'v_proj' in self.target_modules:
                xv = xv + self.lora_wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if adapter is not None:
            adapter_key, adapter_value = adapter
            adapter_len = adapter_key.shape[1]

            adapter_k = self.wk(adapter_key)
            adapter_k = adapter_k.view(bsz, adapter_len, self.n_heads, self.head_dim)
            adapter_v = self.wv(adapter_value)
            adapter_v = adapter_v.view(bsz, adapter_len, self.n_heads, self.head_dim)

            adapter_k = apply_rotary_emb_single(adapter_k, freqs_cis=freqs_cis_prefix)

            adapter_k = adapter_k.transpose(1, 2)
            adapter_v = adapter_v.transpose(1, 2)


        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.cache_enabled:
            if self.cache_k is None:
                assert start_pos == 0
                self.cache_k, self.cache_v = keys, values
            else:
                assert self.cache_k.size(2) >= start_pos
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys, values = self.cache_k, self.cache_v


        if adapter is not None:
            keys = torch.cat([adapter_k, keys], dim=2)
            values = torch.cat([adapter_v, values], dim=2)


        output = self._forward_scaled_dot_product_attention(xq, keys, values, attention_mask=mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        if self.w_lora and 'o_proj' in self.target_modules:
            return self.wo(output) + self.lora_wo(output)
        else:
            return self.wo(output)

    def _forward_scaled_dot_product_attention(self, q, k, v, attention_mask=None):
        if False and hasattr(F, "scaled_dot_product_attention"):
           return F.scaled_dot_product_attention(q, k, v, attention_mask if attention_mask is not None else None)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_weights = torch.matmul(attn_weights, v)

        return attn_weights


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs,
                 edge_index: torch.Tensor = None,
                 input_ids: torch.Tensor = None,
                 input_attention_mask: torch.Tensor = None,
                 ):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim,
        )

        self.w_adapter = params.w_adapter
        self.adapter_len, self.adapter_layer = params.adapter_len, params.adapter_layer
        self.rrwp = params.rrwp
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.register_buffer('input_ids', input_ids)
        self.register_buffer('input_attention_mask', input_attention_mask)


        self.register_buffer('edge_index', edge_index)


        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False,)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
        self.num_hops = params.num_hops


        if self.w_adapter:
            if params.dim == params.adapter_dim:
                self.down_projection = nn.Identity()
                self.up_projection = nn.Identity()
            else:
                self.down_projection = nn.Sequential(
                    nn.Linear(self.params.dim, self.params.adapter_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.params.adapter_dim, self.params.adapter_dim, bias=False))


                self.up_projection = nn.Sequential(
                        nn.Linear(self.params.adapter_dim, self.params.adapter_dim, bias=False),
                        nn.ReLU(),
                        nn.Linear(self.params.adapter_dim, self.params.dim, bias=False))



        if self.w_adapter:
            self.prefix_adapter = PrefixEncoder(params)

        if self.adapter_layer > 0:
            self.graph_adapter = GriTGraphAdapter(params)
            self.graph_adapter_encoder = NeighborContextEncoder(params)



    def forward(self, input_ids, labels, node_ids, attention_mask=None):
        _bsz, seqlen = input_ids.shape
        past_key_values_length = self.adapter_len

        inputs_embeds = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis.to(inputs_embeds.device)[past_key_values_length:]
        freqs_cis_prefix = self.freqs_cis.to(inputs_embeds.device)[:past_key_values_length]

        position_id = torch.arange(seqlen).repeat(_bsz, 1).to(inputs_embeds.device)
        position_id = position_id - ((attention_mask == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis = freqs_cis[position_id]



        if past_key_values_length > 0:
            prefix_attention_mask = torch.ones(
                (_bsz, past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(
                (_bsz, seqlen), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (_bsz, seqlen), inputs_embeds, past_key_values_length)


        start_pos, adapter = 0, None

        if self.w_adapter:
            adapter = self.adapter_forward(batch_size=_bsz, node_ids=node_ids)

        h = self.transformer_forward(input_embeds=inputs_embeds, freqs_cis=freqs_cis,
                                     attention_mask=attention_mask, adapter=adapter,
                                     freqs_cis_prefix=freqs_cis_prefix if self.w_adapter else None
                                     )

        h = self.norm(h)
        output = self.output(h)

        shift_outputs = output[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_outputs.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        c_loss = self.criterion(shift_logits, shift_labels)

        return c_loss


    def transformer_forward(self, input_embeds, freqs_cis, attention_mask, adapter, freqs_cis_prefix=None,
                            start_pos=0):
        h = input_embeds

        if adapter is None:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, attention_mask)
            return h

        adapter_index = 0
        adapter_key, adapter_value = adapter[0], adapter[1]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, attention_mask,
                      (adapter_key[:, adapter_index].bfloat16(), adapter_value[:, adapter_index].bfloat16()),
                      freqs_cis_prefix)

            adapter_index = adapter_index + 1

        return h



    @torch.inference_mode()
    def forward_inference(self, adapter, tokens: torch.Tensor, start_pos: int, attention_mask=None):
        _bsz, seqlen = tokens.shape

        past_key_values_length = self.adapter_len
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(h.device)[past_key_values_length:]
        freqs_cis_prefix = self.freqs_cis.to(h.device)[:past_key_values_length]

        position_id = torch.arange(self.params.max_seq_len).repeat(_bsz, 1).to(h.device)
        position_id = position_id - ((attention_mask == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis = freqs_cis[position_id]
        freqs_cis = freqs_cis[:, start_pos: start_pos + seqlen]

        if past_key_values_length > 0:
            prefix_attention_mask = torch.ones(
                (_bsz, past_key_values_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        if seqlen == 1:
            attention_mask = attention_mask
            attention_mask = prepare_decoder_attention_mask(
                attention_mask, (_bsz, seqlen), h, past_key_values_length)

        elif start_pos == 0:
            # Generate first time
            if attention_mask is None:
                attention_mask = torch.ones(
                    (_bsz, seqlen), dtype=torch.bool, device=h.device
                )
            attention_mask = prepare_decoder_attention_mask(
                attention_mask, (_bsz, seqlen), h, past_key_values_length)
        else:
            raise NotImplementedError()

        if adapter is None:
            for i, layer in enumerate(self.layers):
                h = layer(h, start_pos, freqs_cis, attention_mask)

        else:
            adapter_index = 0
            adapter_key, adapter_value = adapter[0], adapter[1]
            for i, layer in enumerate(self.layers):
                h = layer(h, start_pos, freqs_cis, attention_mask,
                          (adapter_key[:, adapter_index].bfloat16(), adapter_value[:, adapter_index].bfloat16()),
                          freqs_cis_prefix if self.w_adapter else None,
                          )
                adapter_index = adapter_index + 1


        h = self.norm(h)
        output = self.output(h[:, -1, :])
        return output.float()


    def adapter_forward(self, batch_size, node_ids):

        p_adapter_key, p_adapter_value = self.prefix_adapter()
        p_adapter_key = p_adapter_key.repeat(batch_size, 1, 1, 1)
        p_adapter_value = p_adapter_value.repeat(batch_size, 1, 1, 1)


        if self.adapter_len * self.adapter_layer > 0:
            if self.params.task_level == 'graph':
                subset, edge_index_sub, mapping, batch = batch_subgraph_graph_level(self.edge_index, node_ids,
                                                                        num_nodes=self.input_ids.shape[0],)

            elif self.params.task_level == 'pair':
                subset, edge_index_sub, mapping, batch = batch_subgraph_pair_level(self.edge_index, node_ids,
                                                                        num_nodes=self.input_ids.shape[0],)

            else:
                subset, edge_index_sub, mapping, batch = batch_subgraph(self.edge_index, node_ids,
                                                                        num_nodes=self.input_ids.shape[0],
                                                                        num_hops=self.num_hops,
                                                                        fans_out=self.params.fans_out
                                                                        )


            edge_index_full, input_node_pair_embed = add_full_rrwp(edge_index_sub, num_nodes=len(subset),
                                                                   walk_length=self.rrwp
                                                                   )


            adapter_input_ids, adapter_input_attn = self.input_ids[subset], self.input_attention_mask[subset]
            adapter_inputs_embeds = self.tok_embeddings(adapter_input_ids)

            adapter_inputs_embeds = adapter_inputs_embeds.float()
            adapter_inputs_embeds = self.down_projection(adapter_inputs_embeds)
            adapter_inputs_embeds = self.graph_adapter_encoder(adapter_inputs_embeds, adapter_input_attn)

            g_adapter = self.graph_adapter(adapter_inputs_embeds, adapter_input_attn, edge_index_full, mapping,
                                           input_node_pair_embed, batch)


            g_adapter = self.up_projection(g_adapter)
            g_adapter = g_adapter.repeat_interleave(self.n_layers // self.adapter_layer, dim=1)

            adapter_key = g_adapter + p_adapter_key
            adapter_value = g_adapter + p_adapter_value

            adapter = (adapter_key, adapter_value)

        else:
            adapter = (p_adapter_key, p_adapter_value)

        return adapter


    def generate(self, node_ids, input_ids, attention_mask,
            max_new_tokens: int,
            temperature: float = -1.,
            top_p: float = 1.,
            pad_token_id = 0
    ):
        bsz, prompt_size = input_ids.shape
        params = self.params
        self.enable_cache()

        total_len = prompt_size + max_new_tokens
        tokens = torch.full((bsz, total_len), pad_token_id).to(input_ids.device).long()
        tokens[:,:prompt_size] = input_ids

        start_pos = prompt_size
        prev_pos = 0

        _bsz, _ = tokens.shape
        adapter = None


        if self.w_adapter > 0:
            adapter = self.adapter_forward(batch_size=_bsz, node_ids=node_ids)

        for cur_pos in range(start_pos, total_len):
            logits = self.forward_inference(adapter, tokens[:, prev_pos:cur_pos], start_pos=prev_pos, attention_mask=attention_mask)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            attention_mask = torch.cat([attention_mask, torch.ones((bsz, 1)).to(attention_mask.device)], dim=-1)

        self.disable_cache()
        return tokens


    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()

    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()


    def set_trainable_params_new(self):
        param_adapter, param_lora  = [],  []
        adapter = ["graph_adapter", "prefix_adapter", "up_projection", "down_projection"]

        for name, param in self.named_parameters():
            if any(n in name for n in adapter):
                param.requires_grad = True
                param.data = param.data.float()
                param_adapter.append(param)
            elif "lora" in name:
                param.requires_grad = True
                param.data = param.data.float()
                param_lora.append(param)
            else:
                param.requires_grad = False

        return param_adapter, param_lora


    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params


        return trainable_params, all_param


class PrefixEncoder(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super().__init__()
        self.dim = params.dim
        self.adapter_len, self.adapter_layer = params.adapter_len, params.n_layers
        self.prefix_keys = nn.Parameter(torch.randn(1, self.adapter_layer, self.adapter_len, self.dim), requires_grad=True)
        self.prefix_values = nn.Parameter(torch.randn(1, self.adapter_layer, self.adapter_len, self.dim), requires_grad=True)

        nn.init.xavier_normal_(self.prefix_values)
        nn.init.xavier_normal_(self.prefix_keys)


    def forward(self):
        return self.prefix_keys, self.prefix_values



class GriTGraphAdapter(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(GriTGraphAdapter, self).__init__()

        self.graph_adapter_layers = torch.nn.ModuleList()
        self.adapter_layer = params.adapter_layer
        self.adapter_len = params.adapter_len

        self.dim = params.adapter_dim
        self.n_decoder_layers = params.n_decoder_layers
        self.decoder_layers = torch.nn.ModuleList()

        for d_layer_id in range(self.n_decoder_layers):
            self.decoder_layers.append(DecoderLayer(d_layer_id, params))

        self.mp = GritAdapterLayer(params)
        self.query_embed = nn.Parameter(torch.randn([self.adapter_layer, self.adapter_len, self.dim]))

        torch.nn.init.xavier_normal_(self.query_embed)

    def forward(self, input_embeds, input_attn, edge_index, mapping, input_node_pair_embed, batch):
        """
        :param input_embeds: embedding of batched subgraph (N, seq_len, dim)
        :param input_attn: Attn (N, seq_len)
        :param edge_index: edge_index of batched subgraph
        :param mapping:
        :return:
        """
        bsz = input_embeds.shape[0]

        prefix_list = []

        for l in range(self.adapter_layer):
            query = self.query_embed[l: l+1].repeat(bsz, 1, 1)

            for decoder_layer in self.decoder_layers:
                query = decoder_layer(query, input_embeds, input_attn)
            prefix = self.mp(query, edge_index, mapping, input_node_pair_embed, batch)
            prefix_list.append(prefix)

        prefix = torch.stack(prefix_list, dim=1)

        return prefix



class GritAdapterLayer(nn.Module):
    def __init__(self,
                 params: ModelArgs):
        super(GritAdapterLayer, self).__init__()
        self.dim = params.adapter_dim
        self.rrwp = params.rrwp
        self.n_mp_layers = params.n_mp_layers
        self.task_level = params.task_level
        self.params = params

        self.mp_layer = nn.ModuleList()
        self.mp_layer.append(MultiHeadAttentionLayerGritSparse(0, embed_dim=self.dim, num_heads=params.adapter_n_heads,
                                                               initial_layer=True, rrwp=self.rrwp))

        for n_mp_layer in range(1, self.n_mp_layers):
            self.mp_layer.append(MultiHeadAttentionLayerGritSparse(n_mp_layer, embed_dim=self.dim, num_heads=params.adapter_n_heads,
                                                                   initial_layer=False, rrwp=self.rrwp))

        self.wp = nn.Sequential(nn.Linear(self.rrwp, self.dim),
                                nn.ReLU(),
                                nn.Linear(self.dim, self.dim))

        self.wn = nn.Sequential(nn.Linear(self.rrwp, self.dim),
                                nn.ReLU(),
                                nn.Linear(self.dim, self.dim))


        self.ind_mlp = nn.Sequential(nn.Linear(1, self.dim, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(self.dim, self.dim, bias=False))



    def forward(self, query, edge_index, mapping, input_node_pair_embed, batch):
        """
        :param input_embeds: embedding of batched subgraph (N, seq_len, dim)
        :param input_attn: Attn (N, seq_len)
        :param edge_index: edge_index of batched subgraph
        :param mapping:
        :return:
        """

        node_pos = input_node_pair_embed[edge_index[0] == edge_index[1]]
        node_pos = node_pos.view([node_pos.shape[0], 1, node_pos.shape[1]]).repeat(1, query.shape[1], 1)
        node_pos = self.wn(node_pos)
        query = query + node_pos

        out_list = []

        if self.task_level == 'graph':
            for query_idx in range(query.shape[1]):
                x_out, e_out = query[:, query_idx, :], self.wp(input_node_pair_embed)
                for layer in self.mp_layer:
                    x_out, e_out = layer(x=x_out, edge_index=edge_index,
                                         input_node_pair_embed=e_out)

                x_out = global_mean_pool(x_out, batch.to(x_out.device))
                out_list.append(x_out)

            query = torch.stack(out_list, dim=1)

        elif self.task_level == 'pair':
            for query_idx in range(query.shape[1]):

                x_out, e_out = query[:, query_idx, :], self.wp(input_node_pair_embed)
                ind = torch.zeros([mapping[1] - mapping[0], 1], dtype=query.dtype).to(query.device)
                ind[mapping[0] + 1] = 1.
                ind_emb = self.ind_mlp(ind).repeat(mapping.shape[0], 1)
                x_out = x_out + ind_emb

                for layer in self.mp_layer:
                    x_out, e_out = layer(x=x_out, edge_index=edge_index,
                                         input_node_pair_embed=e_out)

                out_list.append(x_out)

            query = torch.stack(out_list, dim=1)
            # query = torch.cat([query[mapping][:, :self.params.adapter_len//2,:], query[mapping + 1][:, self.params.adapter_len//2:, :]], dim=1)
            query = query[mapping]


        else:

            for query_idx in range(query.shape[1]):
                x_out, e_out = query[:, query_idx, :], self.wp(input_node_pair_embed)
                for layer in self.mp_layer:
                    x_out, e_out = layer(x=x_out, edge_index=edge_index,
                                         input_node_pair_embed=e_out)

                out_list.append(x_out)

            query = torch.stack(out_list, dim=1)
            query = query[mapping]


        return query


class DecoderLayer(nn.Module):
    def __init__(self,
                 layer_id,
                 params: ModelArgs,
                 ):
        super(DecoderLayer, self).__init__()
        self.layer_id = layer_id
        self.adapter_len = params.adapter_len
        self.self_attention_layer = DecoderSelfAttentionLayer(params=params)
        self.cross_attention_layer = CrossAttentionLayer(params=params)
        self.dim = params.adapter_dim
        self.feed_forward = nn.Sequential(nn.Linear(self.dim, self.dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(2 * self.dim, self.dim))

        self.self_attention_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)
        self.cross_attention_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)
        self.ffn_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)


    def forward(self, query, input_embeds, input_attn):

        query = query + self.self_attention_layer(query)
        query = self.self_attention_norm(query)
        query = query + self.cross_attention_layer(query, input_embeds, input_attn)
        query = self.cross_attention_norm(query)


        query = query + self.feed_forward(query)
        query = self.ffn_norm(query)

        return query


class DecoderSelfAttentionLayer(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(DecoderSelfAttentionLayer, self).__init__()

        self.n_heads = params.adapter_n_heads
        self.dim = params.adapter_dim
        self.head_dim = self.dim //self.n_heads
        self.adapter_len = params.adapter_len


        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False,)


    def forward(self, query):
        _bsz, query_len, _ = query.shape

        xq, xk, xv = self.wq(query), self.wk(query), self.wv(query)

        xq = xq.view(_bsz, query_len, self.n_heads, self.head_dim)
        xk = xk.view(_bsz, query_len, self.n_heads, self.head_dim)
        xv = xv.view(_bsz, query_len, self.n_heads, self.head_dim)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv)

        output = output.transpose(1, 2).contiguous().view(_bsz, query_len, -1)
        return self.wo(output)


class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(CrossAttentionLayer, self).__init__()

        self.n_heads = params.adapter_n_heads
        self.head_dim = params.adapter_dim//self.n_heads
        self.adapter_len = params.adapter_len
        self.dim = params.adapter_dim

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False,)

        self.freqs_cis = precompute_freqs_cis(self.dim // self.n_heads, 4096 * 2)

    def forward(self, query, input_embeds, input_attn):
        """
        :param query:   [Adapter_len * Adapter_layer, Dim]
        :param input_embeds:   [N, seq_len, Dim]
        :return:
        """

        N, src_seqlen, _ = input_embeds.shape
        trg_seqlen = self.adapter_len

        freqs_cis = self.freqs_cis.to(input_embeds.device)

        position_id = torch.arange(src_seqlen).repeat(N, 1).to(input_embeds.device)
        position_id = position_id - ((input_attn == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis_src = freqs_cis[position_id]

        freqs_cis_trg = freqs_cis[:trg_seqlen]



        xk, xv = self.wk(input_embeds), self.wv(input_embeds)
        xq = self.wq(query)

        xq = xq.view(-1, self.adapter_len, self.n_heads, self.head_dim)
        xk = xk.view(N, src_seqlen, self.n_heads, self.head_dim)
        xv = xv.view(N, src_seqlen, self.n_heads, self.head_dim)

        xk = apply_rotary_emb_single(xk, freqs_cis=freqs_cis_src)
        xq = apply_rotary_emb_single(xq, freqs_cis=freqs_cis_trg)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Attn (N, seq_len) -> (N, ..., Adapter_len * Adapter_layer, seq_len)

        input_attn = input_attn.view(N, 1, 1, src_seqlen).repeat(1, 1, trg_seqlen, 1)
        input_attn = 1.0 - input_attn
        input_attn = input_attn.masked_fill(input_attn.to(torch.bool), torch.finfo(input_embeds.dtype).min).float()

        output = F.scaled_dot_product_attention(xq, xk, xv, input_attn if input_attn is not None else None)
        output = output.transpose(1, 2).contiguous().view(N, trg_seqlen, -1)

        return self.wo(output)



class NeighborContextEncoder(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(NeighborContextEncoder, self).__init__()
        self.n_encoder_layers = params.n_encoder_layers
        self.encoder = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_encoder_layers)])

    def forward(self, query, input_attn):
        for layer in self.encoder:
            query = layer(query, input_attn)
        return query



class EncoderLayer(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(EncoderLayer, self).__init__()
        self.dim = params.adapter_dim
        self.attention = EncoderSelfAttention(params)
        self.feed_forward = nn.Sequential(nn.Linear(self.dim, self.dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(2 * self.dim, self.dim))

        self.attention_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)
        self.ffn_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)

    def forward(self, query, input_attn):
        query = query + self.attention(query, input_attn)
        query = self.attention_norm(query)
        query = query + self.feed_forward(query)
        query = self.ffn_norm(query)
        return query



class EncoderSelfAttention(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(EncoderSelfAttention, self).__init__()

        self.n_heads = params.adapter_n_heads
        self.dim = params.adapter_dim
        self.head_dim = self.dim // self.n_heads
        self.adapter_len = params.adapter_len

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, )
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, )
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, )

        self.freqs_cis = precompute_freqs_cis(self.dim // self.n_heads, params.max_seq_len)


    def forward(self, query, input_attn):
        _bsz, query_len, _ = query.shape
        freqs_cis = self.freqs_cis.to(query.device)

        position_id = torch.arange(query_len).repeat(_bsz, 1).to(query.device)
        position_id = position_id - ((input_attn == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis = freqs_cis[position_id]


        xq, xk, xv = self.wq(query), self.wk(query), self.wv(query)

        xq = xq.view(_bsz, query_len, self.n_heads, self.head_dim)
        xk = xk.view(_bsz, query_len, self.n_heads, self.head_dim)
        xv = xv.view(_bsz, query_len, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        # Causal Attention

        # input_attn = prepare_decoder_attention_mask(
        #     input_attn, (_bsz, query_len), query, 0)


        # Full Attention

        input_attn = input_attn.view(_bsz, 1, 1, query_len).repeat(1, 1, query_len, 1)
        input_attn = 1.0 - input_attn
        input_attn = input_attn.masked_fill(input_attn.to(torch.bool), torch.finfo(query.dtype).min).float()

        output = F.scaled_dot_product_attention(xq, xk, xv, input_attn)

        output = output.transpose(1, 2).contiguous().view(_bsz, query_len, -1)
        return self.wo(output)
