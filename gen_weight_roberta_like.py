import torch
import os, json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gmllm', type=str, required=True, help='Path to GMLLM model.')
    parser.add_argument('--text', type=str, required=True, help='Path to text model.')
    parser.add_argument('--config', type=str, required=True, help='Path to text config.')
    parser.add_argument('--out', type=str, required=True, help='Path to output.')
    opt = parser.parse_args()

    with open(opt.config, 'r') as jf:
        config = json.load(jf)
    config['model_type'] = 'gmllm'
    config['layer_nums'] = '4_4_4'
    config['layout_hidden_size'] = 192
    config['layout_intermediate_size'] = 768
    config["coordinate_size"] = 32
    config["shape_size"] = 32
    config["pos_embed_size"] = 24
    config['graph_type'] = 'knn_knn_none'
    config['layout_params'] = "100,50,0"
    config['layout_tasks'] = 'word-kvp_line-kvp'
    config['layout_type'] = 'word_line_region'
    config['max_2d_position_embeddings'] = 1024
    config['num_layout_attention_heads'] = 12
    
    if not os.path.isdir(opt.out):
        os.makedirs(opt.out)
    with open(os.path.join(opt.out, 'config.json'), 'w') as jf:
        json.dump(config, jf, sort_keys=True, indent=2, separators=(',', ': '),)
   
    text_model = torch.load(opt.text)
    text_model = {k.replace('roberta.', 'gmllm.'): v for (k, v) in text_model.items()}
    gmllm_model = torch.load(opt.gmllm)
    gmllm_model = {k:v for (k, v) in gmllm_model.items() if k not in text_model}
    total_model = {**text_model, **gmllm_model}
    torch.save(total_model, os.path.join(opt.out, 'pytorch_model.bin'))
