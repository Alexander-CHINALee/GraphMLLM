# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from .configuration_GMLLM import GMLLMConfig
from .modeling_graphdoc import GraphPlusEmbeddings, GraphPlusDocLayer, GraphStackDocLayer

logger = logging.get_logger(__name__)

class GMLLMTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
       
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_ids

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class GMLLMSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.layout_hidden_size = config.layout_hidden_size
        self.layout_head_size = int(self.layout_hidden_size / config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x, head_size):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, head_size)
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 4:
            return x.permute(0, 2, 1, 3)
        elif len(new_x_shape) == 5:
            return x.permute(0, 3, 1, 2, 4)

    def forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # text kqv
        layer_query = self.transpose_for_scores(self.query(hidden_states),self.attention_head_size)
        layer_key = self.transpose_for_scores(self.key(hidden_states),self.attention_head_size)
        layer_value = self.transpose_for_scores(self.value(hidden_states),self.attention_head_size)
        attention_scores = torch.matmul(layer_query, layer_key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        # layout kq
        layout_query = self.transpose_for_scores(layout_inputs[0], self.attention_head_size)
        layout_key = self.transpose_for_scores(layout_inputs[1], self.attention_head_size)
        # text-layout attention
        layout_attention_scores = torch.matmul(layout_query, layout_key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + layout_attention_scores
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # Mask heads
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # text context aggregation
        context_layer = torch.matmul(attention_probs, layer_value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # output
        outputs = ((context_layer, ), attention_scores) if output_attentions else ((context_layer, ),)
        return outputs


class GMLLMSelfOutput(nn.Module):
    def __init__(self, config, out_size=None):
        super().__init__()
        if out_size is not None:
            hidden_size = config.hidden_size
            out_size = out_size
        else:
            hidden_size = config.hidden_size
            out_size = config.hidden_size
        self.dense = nn.Linear(hidden_size, out_size)
        self.LayerNorm = nn.LayerNorm(out_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GMLLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GMLLMSelfAttention(config)
        self.output = GMLLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            layout_inputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0][0], hidden_states)
        outputs = ((attention_output, ),) + self_outputs[1:]
        return outputs


class GMLLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GMLLMOutput(nn.Module):
    def __init__(self, config, reduce=False):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GMLLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GMLLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = GMLLMAttention(config)
        # text
        self.intermediate = GMLLMIntermediate(config)
        self.output = GMLLMOutput(config)

    def forward(
        self,
        hidden_states,
        layout_inputs,
        layout_index=None,
        attention_mask=None,
        layout_attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            layout_inputs,
            layout_index,
            attention_mask,
            layout_attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0][0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class GMLLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GMLLMLayer(config) for _ in range(config.num_hidden_layers)])
        self.graph_layer = nn.ModuleList([GraphStackDocLayer(config, layer_i) for layer_i in range(config.num_hidden_layers)])
        self.layer_query = nn.Linear(config.hidden_size, config.layout_hidden_size)
        self.layer_key = nn.Linear(config.hidden_size, config.layout_hidden_size)
        self.layout_query = nn.ModuleList()
        self.layout_key = nn.ModuleList()
        for layout_i in range(len(config.layout_type.split('_'))):
            self.layout_query.append(nn.Linear(config.layout_hidden_size*(layout_i+2), config.hidden_size))
            self.layout_key.append(nn.Linear(config.layout_hidden_size*(layout_i+2), config.hidden_size))

    def forward(
        self,
        hidden_states,
        layout_hidden,
        rel_bbox_emb,
        pos_emb,
        rel_bbox_index,
        layout_index=None,
        addition_index=None,
        attention_mask=None,
        extended_attention_mask=None,
        layout_attention_mask=None,
        local_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        layout_no = 0
        for i, layer_module in enumerate(self.layer):
            graph_module = self.graph_layer[i]
            if graph_module.layout_no > 0:
               self.layout_to_layout_agg(layout_hidden, addition_index, layout_attention_mask, graph_module.layout_no, 0.5)
            layout_no = graph_module.layout_no
            layout_inputs = self.layout_to_text_agg(layout_hidden, pos_emb, layout_index, graph_module.layout_no)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                layout_inputs,
                extended_attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions=output_attentions,
            )
            layer_inter = self.text_to_layout_agg(layout_hidden, hidden_states, layout_index, attention_mask, graph_module.layout_no)
            layout_hidden = graph_module(
                layout_hidden,
                layer_inter,
                rel_bbox_emb,
                rel_bbox_index,
                layout_attention_mask,
                local_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            ), layout_hidden
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), layout_hidden

    def layout_to_text_agg(self, layout_inputs, pos_emb, rel_index, inter_no=0):
        layout_aggs = [pos_emb]
        #layout_aggs = []
        for idx in range(inter_no+1):
            lay_in, rel_idx = layout_inputs[idx], rel_index[idx]
            B, L = rel_idx.size()
            D = lay_in.size(-1)
            rel_idx = rel_idx.unsqueeze(-1).expand((B, L, D))
            layout_aggs.append(torch.gather(lay_in, 1, rel_idx))
        layout_aggs = torch.cat(layout_aggs,dim=-1)
        return [self.layout_query[inter_no](layout_aggs), self.layout_key[inter_no](layout_aggs)]

    def layout_to_layout_agg(self, layout_inputs, rel_index, layout_mask, inter_no, ratio=0.5):
        lay_down, lay_up, rel_idx = layout_inputs[inter_no-1], layout_inputs[inter_no], rel_index[inter_no-1]
        rel_ones = torch.ones_like(rel_idx,dtype=lay_up.dtype)
        lay_nums = torch.zeros(lay_up.size()[:-1],device=lay_up.device,dtype=lay_up.dtype)
        lay_nums.scatter_add_(1, rel_idx, rel_ones)
        lay_nums[:,0] = 1
        lay_nums += (lay_nums==0).to(dtype=lay_nums.dtype)
        rel_idx = rel_idx.unsqueeze(-1).expand_as(lay_down)
        up_aggs = torch.zeros_like(lay_up)
        up_aggs.scatter_add_(1, rel_idx, lay_down * layout_mask[inter_no-1].unsqueeze(dim=-1))
        up_aggs /= lay_nums.unsqueeze(-1)
        down_aggs = torch.gather(lay_up, 1, rel_idx)
        layout_inputs[inter_no-1] = layout_inputs[inter_no-1] * ratio + down_aggs * (1-ratio)
        layout_inputs[inter_no] = layout_inputs[inter_no] * ratio + up_aggs * (1-ratio)
    
    def text_to_layout_agg(self, layout_inputs, text_inputs, rel_index, text_mask, inter_no=-1):
        text_inputs = text_inputs
        layout_outputs = []
        txt_in = text_inputs * text_mask.unsqueeze(dim=-1)
        for idx in range(inter_no+1):
            lay_in, rel_idx = layout_inputs[idx], rel_index[idx]
            rel_ones = torch.ones_like(rel_idx,dtype=lay_in.dtype)
            token_nums = torch.zeros(lay_in.size()[:-1],device=lay_in.device,dtype=lay_in.dtype)
            token_nums.scatter_add_(1, rel_idx, rel_ones)
            # special token_num is one
            token_nums[:,0] = 1
            token_nums += (token_nums==0).to(dtype=token_nums.dtype)
            rel_idx = rel_idx.unsqueeze(-1).expand_as(txt_in)
            agg_size = list(lay_in.size())
            agg_size[-1] = txt_in.size(-1)
            text_aggs = torch.zeros(agg_size,dtype=txt_in.dtype,device=txt_in.device)
            text_aggs.scatter_add_(1, rel_idx, txt_in)
            text_aggs /= token_nums.unsqueeze(-1)
            layout_outputs.append([self.layer_query(text_aggs), self.layer_key(text_aggs)])
        return layout_outputs


class GMLLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GMLLMPreTrainedModel(PreTrainedModel):
    config_class = GMLLMConfig
    base_model_prefix = "gmllm"
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GMLLMModel(GMLLMPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = GMLLMTextEmbeddings(config)
        self.graph_layout_embeddings = GraphPlusEmbeddings(config)
        self.encoder = GMLLMEncoder(config)
        self.pooler = GMLLMPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        layout_index=None,
        addition_index=None,
        attention_mask=None,
        bbox_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        text_emb, position_ids = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # layout_emb, rel_bbox_emb, rel_bbox_index
        layout_embs, rel_bbox_embs, pos_embs, rel_bbox_indexs, local_masks = self.graph_layout_embeddings(
            bbox=bbox,
            position_ids=position_ids,
            addition_index=addition_index,
            attention_mask=bbox_mask,
        )
        encoder_outputs, layout_outputs = self.encoder(
            text_emb,
            layout_embs,
            rel_bbox_embs,
            pos_embs,
            rel_bbox_indexs,
            layout_index=layout_index,
            addition_index=addition_index,
            attention_mask=attention_mask,
            extended_attention_mask=extended_attention_mask,
            local_mask=local_masks,
            layout_attention_mask=bbox_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ), layout_outputs, rel_bbox_indexs


class GMLLMForTokenClassification(GMLLMPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layout_types = config.layout_type.split('_')
        self.gmllm = GMLLMModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + len(self.layout_types)*config.layout_hidden_size,config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        wbbox=None,
        lbbox=None,
        rbbox=None,
        layout_windex=None,
        layout_lindex=None,
        layout_rindex=None,
        layout_wlindex=None,
        layout_lrindex=None,
        attention_mask=None,
        wbbox_mask=None,
        lbbox_mask=None,
        rbbox_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        bbox = []
        layout_index = []
        addition_index = []
        bbox_mask = []
        if 'word' in self.layout_types:
            bbox.append(wbbox)
            layout_index.append(layout_windex)
            bbox_mask.append(wbbox_mask)
        if 'line' in self.layout_types:
            bbox.append(lbbox)
            layout_index.append(layout_lindex)
            addition_index.append(layout_wlindex)
            bbox_mask.append(lbbox_mask)
        if 'region' in self.layout_types:
            bbox.append(rbbox)
            layout_index.append(layout_rindex)
            addition_index.append(layout_lrindex)
            bbox_mask.append(rbbox_mask)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs, layout_outputs, rel_bbox_indexs = self.gmllm(
            input_ids,
            bbox=bbox,
            layout_index=layout_index,
            addition_index=addition_index,
            attention_mask=attention_mask,
            bbox_mask=bbox_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        layout_fusion = [outputs[0]]
        for i in range(len(layout_index)):
            layout = layout_outputs[i]
            lay_idx = layout_index[i].unsqueeze(-1).expand((layout.size(0),layout_index[i].size(1),layout.size(-1)))
            layout_fusion.append(torch.gather(layout, 1, lay_idx))
        sequence_output = torch.cat(layout_fusion, -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=outputs.attentions,
        )


from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from transformers.file_utils import ModelOutput
from ...modules.decoders.re import REDecoder
from ...utils import ReOutput

class GMLLMForRelationExtraction(GMLLMPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        self.layout_types = config.layout_type.split('_')
        self.gmllm = GMLLMModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = REDecoder(config, config.hidden_size + len(self.layout_types)*config.layout_hidden_size)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        wbbox=None,
        lbbox=None,
        rbbox=None,
        layout_windex=None,
        layout_lindex=None,
        layout_rindex=None,
        layout_wlindex=None,
        layout_lrindex=None,
        attention_mask=None,
        wbbox_mask=None,
        lbbox_mask=None,
        rbbox_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entities=None,
        relations=None,
    ):
        bbox = []
        layout_index = []
        addition_index = []
        bbox_mask = []
        if 'word' in self.layout_types:
            bbox.append(wbbox)
            layout_index.append(layout_windex)
            bbox_mask.append(wbbox_mask)
        if 'line' in self.layout_types:
            bbox.append(lbbox)
            layout_index.append(layout_lindex)
            addition_index.append(layout_wlindex)
            bbox_mask.append(lbbox_mask)
        if 'region' in self.layout_types:
            bbox.append(rbbox)
            layout_index.append(layout_rindex)
            addition_index.append(layout_lrindex)
            bbox_mask.append(rbbox_mask)
        outputs, layout_outputs,rel_bbox_indexs = self.gmllm(
            input_ids,
            bbox=bbox,
            layout_index=layout_index,
            addition_index=addition_index,
            attention_mask=attention_mask,
            bbox_mask=bbox_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # fusion
        layout_fusion = [outputs[0]]
        for i in range(len(layout_index)):
            layout = layout_outputs[i]
            lay_idx = layout_index[i].unsqueeze(-1).expand((layout.size(0),layout_index[i].size(1),layout.size(-1)))
            layout_fusion.append(torch.gather(layout, 1, lay_idx))
        sequence_output = torch.cat(layout_fusion, -1)
        seq_length = input_ids.size(1)
        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)

        return ReOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
