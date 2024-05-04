# coding=utf-8
import math
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.functional import embedding
from transformers.activations import ACT2FN, gelu
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoModel
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

logger = logging.get_logger(__name__)

GraphDocLayerNorm = torch.nn.LayerNorm


# full connected graph
class FullGraph():
    def __call__(self, bbox, bbox_mask=None, addition_index=None, param=0):
        B, L, _ = bbox.shape
        if param == 1:
            limit_mask = (addition_index[:,:,None] == addition_index[:,None,:]).int()
            local_mask = limit_mask * bbox_mask.unsqueeze(1) * bbox_mask.unsqueeze(2)
        else:
            local_mask = bbox_mask.unsqueeze(1) * bbox_mask.unsqueeze(2)
        max_len = local_mask.sum(dim=1).max()
        full_index = local_mask.topk(max_len, dim=-1, largest=True)[1]
        return full_index, local_mask


class KnnGraph():
    # k-nearest graph
    def __call__(self, bbox, bbox_mask=None, addition_index=None, param=50):
        bbox = bbox.masked_fill((1-bbox_mask).unsqueeze(-1).to(torch.bool), int(1e8))
        B, L, _ = bbox.shape
        topk = min(L, param)
        x1 = bbox[:, :, 0]
        y1 = bbox[:, :, 1]
        x2 = bbox[:, :, 2]
        y2 = bbox[:, :, 3]
        xc = torch.div((x1 + x2), 2)
        yc = torch.div((y1 + y2), 2)
        diff_xc = xc[:, :, None] - xc[:, None, :]
        diff_yc = yc[:, :, None] - yc[:, None, :]
        distance = diff_xc.pow(2) + diff_yc.pow(2)
        topk_index = distance.topk(topk, dim=-1, largest=False)[1]
        # local attention mask
        B, L, Topk = topk_index.shape
        local_mask = torch.zeros((B, L, L),device=topk_index.device,dtype=topk_index.dtype)
        local_mask = local_mask.scatter(-1, topk_index, 1)
        return topk_index, local_mask

class GraphDocEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, graph_type, ratio=1):
        super(GraphDocEmbeddings, self).__init__()
        self.knn_graph = KnnGraph()
        self.full_graph = FullGraph()
        self.max_2d_position_embeddings = config.max_2d_position_embeddings
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size//ratio)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size//ratio)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size//ratio)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size//ratio)

        self.rel_position_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.pos_embed_size, padding_idx=config.pad_token_id)
        self.W_tl = nn.Linear(in_features=int(2*config.pos_embed_size//ratio), out_features=config.layout_hidden_size//ratio)
        self.W_tr = nn.Linear(in_features=int(2*config.pos_embed_size//ratio), out_features=config.layout_hidden_size//ratio)
        self.W_bl = nn.Linear(in_features=int(2*config.pos_embed_size//ratio), out_features=config.layout_hidden_size//ratio)
        self.W_br = nn.Linear(in_features=int(2*config.pos_embed_size//ratio), out_features=config.layout_hidden_size//ratio)

        self.LayerNorm = GraphDocLayerNorm(config.layout_hidden_size//ratio, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def _cal_rel_position_embeddings(self, bbox, topk_index):
        if topk_index is not None:
            x1 = bbox[:, :, 0]
            y1 = bbox[:, :, 1]
            x2 = bbox[:, :, 2]
            y2 = bbox[:, :, 3]
            
            diff_x1 = x1[:, :, None] - x1[:, None, :]
            diff_y1 = y1[:, :, None] - y1[:, None, :]
            diff_x2 = x2[:, :, None] - x2[:, None, :]
            diff_y2 = y2[:, :, None] - y2[:, None, :]

            diff_x1 = diff_x1.gather(2, topk_index)
            diff_y1 = diff_y1.gather(2, topk_index)
            diff_x2 = diff_x2.gather(2, topk_index)
            diff_y2 = diff_y2.gather(2, topk_index)

            diff_x1 = torch.clamp(diff_x1, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1)
            diff_y1 = torch.clamp(diff_y1, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1)
            diff_x2 = torch.clamp(diff_x2, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1)
            diff_y2 = torch.clamp(diff_y2, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1)

            diff_x1 = self.rel_position_embeddings(diff_x1)
            diff_y1 = self.rel_position_embeddings(diff_y1)
            diff_x2 = self.rel_position_embeddings(diff_x2)
            diff_y2 = self.rel_position_embeddings(diff_y2)

            p_tl = self.W_tl(torch.cat([diff_x1, diff_y1], dim=-1))
            p_tr = self.W_tr(torch.cat([diff_x2, diff_y1], dim=-1))
            p_bl = self.W_bl(torch.cat([diff_x1, diff_y2], dim=-1))
            p_br = self.W_br(torch.cat([diff_x2, diff_y2], dim=-1))
            p = p_tl + p_tr + p_bl + p_br
            return p
        else:
            return None

    def forward(
        self,
        bbox=None,
        attention_mask=None,
        params=None,
        graph_type=None,
        addition_index=None,
        position_ids=None
    ):
        layout_emb = self._cal_spatial_position_embeddings(bbox)
        if graph_type == 'knn':
            graph_index, local_mask = self.knn_graph(bbox,attention_mask,addition_index,params)
        elif graph_type == 'full':
            graph_index, local_mask = self.full_graph(bbox,attention_mask,addition_index,params)
        elif graph_type == 'none':
            graph_index, local_mask = None, None
        else:
            raise "graph type error!"
        rel_bbox_emb = self._cal_rel_position_embeddings(bbox, graph_index)
        layout_emb = self.LayerNorm(layout_emb)
        layout_emb = self.dropout(layout_emb)
        return layout_emb, rel_bbox_emb, graph_index, local_mask


class DocGraphSelfOutput(nn.Module):
    def __init__(self, config, ratio=1):
        super().__init__()
        self.dense = nn.Linear(config.layout_hidden_size//ratio, config.layout_hidden_size//ratio)
        self.LayerNorm = nn.LayerNorm(config.layout_hidden_size//ratio, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DocGraphIntermediate(nn.Module):
    def __init__(self, config, ratio=1):
        super().__init__()
        self.dense = nn.Linear(config.layout_hidden_size//ratio, config.layout_intermediate_size//ratio)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DocGraphOutput(nn.Module):
    def __init__(self, config, ratio=1):
        super().__init__()
        self.dense = nn.Linear(config.layout_intermediate_size//ratio, config.layout_hidden_size//ratio)
        self.LayerNorm = nn.LayerNorm(config.layout_hidden_size//ratio, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.max_positions = int(1e5)

    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
    
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        positions
    ):
        self.weights = self.weights.to(positions.device)

        return (
            self.weights[positions.reshape(-1)]
            .view(positions.size() + (-1,))
            .detach()
        )


class GraphDocSelfAttention(nn.Module):
    def __init__(self, config, ratio=1):
        super().__init__()
        if config.hidden_size*ratio % config.num_layout_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.layout_hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_layout_attention_heads})"
            )
        # layout
        self.layout_hidden_size = config.layout_hidden_size // ratio
        self.num_layout_attention_heads = config.num_layout_attention_heads // ratio
        self.attention_head_size = int(self.layout_hidden_size / self.num_layout_attention_heads)
        self.all_head_size = self.num_layout_attention_heads * self.attention_head_size
        self.qkv_linear = nn.Linear(self.layout_hidden_size, 3 * self.all_head_size, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        self.rel_bbox_query = nn.Linear(self.layout_hidden_size, self.all_head_size)
        
        self.layer_head_size = config.hidden_size // self.num_layout_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, head_size):
        new_x_shape = x.size()[:-1] + (self.num_layout_attention_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_bbox_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_layout_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def compute_qkv(self, hidden_states):
        qkv = self.qkv_linear(hidden_states)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        if q.ndimension() == self.q_bias.ndimension():
            q = q + self.q_bias
            v = v + self.v_bias
        else:
            _sz = (1,) * (q.ndimension() - 1) + (-1,)
            q = q + self.q_bias.view(*_sz)
            v = v + self.v_bias.view(*_sz)
        return q, k, v

    def forward(
        self,
        hidden_states,
        layer_inputs,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        local_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        layer_query = self.transpose_for_scores(layer_inputs[0], self.attention_head_size)
        layer_key = self.transpose_for_scores(layer_inputs[1], self.attention_head_size)
        layer_attention_scores = torch.matmul(layer_query, layer_key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        q, k, v = self.compute_qkv(hidden_states)
        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q, self.attention_head_size)
        key_layer = self.transpose_for_scores(k, self.attention_head_size)
        value_layer = self.transpose_for_scores(v, self.attention_head_size)
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        layout_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = layout_attention_scores + layer_attention_scores
        q_bbox = self.rel_bbox_query(q)
        query_bbox_layer = self.transpose_for_scores(q_bbox, self.attention_head_size)
        # (B, L, topk, H*D) -> (B, H, L, topk, D)
        rel_bbox_emb = self.transpose_for_bbox_scores(rel_bbox_emb)
        query_bbox_layer = query_bbox_layer / math.sqrt(self.attention_head_size)
        # cal rel bbox attention score
        attention_bbox_scores = torch.einsum('bhid,bhijd->bhij', query_bbox_layer, rel_bbox_emb)
        attention_scores = attention_scores.scatter_add(-1, rel_bbox_index.unsqueeze(1).expand_as(attention_bbox_scores), attention_bbox_scores)
        local_attention_mask = 1 - local_mask.unsqueeze(1).expand_as(attention_scores)
        attention_scores = attention_scores.float().masked_fill_(local_attention_mask.to(torch.bool), float(-1e8)) # remove too far token

        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float(-1e8)) # remove padding token
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class GraphDocAttention(nn.Module):
    def __init__(self, config, ratio=1):
        super().__init__()
        self.self = GraphDocSelfAttention(config, ratio)
        self.output = DocGraphSelfOutput(config, ratio)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_layout_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_layout_attention_heads = self.self.num_layout_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_layout_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        layer_inputs,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        local_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            layer_inputs,
            rel_bbox_emb,
            rel_bbox_index,
            attention_mask,
            local_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class GraphDocLayer(nn.Module):
    def __init__(self, config, ratio=1):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GraphDocAttention(config, ratio)
        self.is_decoder = config.is_decoder
        self.intermediate = DocGraphIntermediate(config, ratio)
        self.output = DocGraphOutput(config, ratio)

    def forward(
        self,
        hidden_states,
        layer_inputs,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        local_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            layer_inputs,
            rel_bbox_emb,
            rel_bbox_index,
            extended_attention_mask,
            local_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class GraphPlusEmbeddings(nn.Module):
    def __init__(self, config):
        super(GraphPlusEmbeddings, self).__init__()
        self.graph_type = config.graph_type.split('_')
        self.layout_type = config.layout_type.split('_')
        self.layout_num = len(self.layout_type)
        self.layout_params = [int(param) for param in config.layout_params.split(',')]
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.layout_hidden_size)
        self.basic_embedding = GraphDocEmbeddings(config, self.graph_type[0])
        
    def forward(
        self,
        bbox,
        position_ids=None,
        addition_index=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        node_emb, edge_emb, rel_index, local_mask  = [], [], [], []
        for i in range(self.layout_num):
            if i < len(addition_index):
                add_idx = addition_index[i]
            else:
                add_idx = None
            
            g_node_emb, g_edge_emb, g_rel, g_mask = self.basic_embedding(bbox[i], attention_mask[i], self.layout_params[i], self.graph_type[i], addition_index=add_idx)
            node_emb.append(g_node_emb)
            edge_emb.append(g_edge_emb)
            rel_index.append(g_rel)
            local_mask.append(g_mask)
        pos_emb = self.position_embeddings(position_ids)
        return node_emb, edge_emb, pos_emb, rel_index, local_mask


class GraphPlusDocLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layout_type = config.layout_type.split('_')
        layout_layers = []
        for layout_type in self.layout_type:
            layout_layers.append(GraphDocLayer(config))
        self.layout_layers = nn.ModuleList(layout_layers)
    
    def forward(
        self,
        layout_inputs,
        layer_inputs,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        local_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        layout_outs = []
        for i, layer_module in enumerate(self.layout_layers):
            layout = layer_module(layout_inputs[i], layer_inputs[i], rel_bbox_emb[i], rel_bbox_index[i], attention_mask[i], local_mask[i])
            layout_outs.append(layout)
        
        return layout_outs


class GraphStackDocLayer(nn.Module):
    def __init__(self, config, layer_no):
        super().__init__()
        self.layout_layer = GraphDocLayer(config)
        self.layout_no = 0
        layer_cnt = 0
        for num in config.layer_nums.split('_'):
            layer_cnt += int(num)
            if layer_no < layer_cnt:
                break
            self.layout_no += 1
    
    def forward(
        self,
        layout_inputs,
        layer_inputs,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        local_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None
    ):
        for i in range(min(self.layout_no+1,2)):
            layout = self.layout_layer(layout_inputs[i], layer_inputs[i], rel_bbox_emb[i], rel_bbox_index[i], attention_mask[i], local_mask[i])
            layout_inputs[i] = layout
        return layout_inputs

