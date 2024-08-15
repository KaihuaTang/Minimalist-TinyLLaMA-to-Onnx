import os 
import torch
import argparse
from safetensors.torch import load_file
import torch.onnx as torch_onnx
from torch.autograd import Variable
from transformers import AutoConfig, AutoModelForCausalLM
import numpy as np
import onnx
import onnxruntime
from utils import *

from llama.modeling_llama import LlamaForCausalLM, LlamaRotaryEmbedding


def main():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()
    
     # init output path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"==> Create output folder: {args.output_path}")    
        
    # init model
    print(f"==> Loading config from {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = LlamaForCausalLM(config)
    ckpt_path = os.path.join(args.model_path, "model.safetensors")
    print(f"==> Loading ckpt from {ckpt_path}")
    print(f"==> skip embedding layer to keep the entire onnx size < 2G")
    checkpoint = load_file(ckpt_path)
    print(model.load_state_dict(checkpoint, strict=False))
    # set to eval model
    model = model.half().cuda().eval()

    # get input embedding
    print(f"==> Get embedding weight")
    embed_weight = checkpoint['model.embed_tokens.weight'].data.detach().cpu().half()
    # save embedding
    embed_path = os.path.join(args.output_path, "embedding_weight.bin")
    print(f"==> Saving embed_weight (shape {embed_weight.shape}) to {embed_path}")
    embed_weight.numpy().astype(np.float16).tofile(embed_path)
    
    # prepare sample input
    print(f"==> Prepare sample input {str([[4790, 4792, 24954]])}")
    model_inputs = prepare_inputs_for_generation(torch.LongTensor([[4790, 4792, 24954]]), count=0, max_length=args.max_length)
    
    # saving position ID
    position_path = os.path.join(args.output_path, "pos_files")
    if not os.path.exists(position_path):
        os.makedirs(position_path)
        print(f"==> Create position embedding cos/sin output folder: {position_path}")
    
    input_ids = model_inputs['input_ids']
    input_embed = Variable(embed_weight[input_ids.reshape(-1).item()].reshape(1, 1, -1)).half().cuda()
    position_ids = Variable(model_inputs['position_ids'])
    
    head_dim = int(config.hidden_size / config.num_attention_heads)
    rotary_pos_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings=args.max_length)
    cos, sin = rotary_pos_emb(input_embed, seq_len=args.max_length)
    cos = cos.squeeze(1).squeeze(0).half().cuda()  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0).half().cuda()  # [seq_len, dim]
    for i in range(args.max_length):
        position_ids = torch.arange(args.max_length, dtype=torch.long).cuda()[i].view(1,-1).long()
        cos_i = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin_i = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        retary_cos_sin = torch.cat([cos_i, sin_i], dim=2).half().cuda()
        retary_cos_sin.cpu().numpy().astype(np.float16).tofile(os.path.join(position_path, '{}.bin'.format(i)))
    
    print('retary_cos_sin shape is: {}'.format(str(retary_cos_sin.shape)))
    attention_mask = Variable(model_inputs['attention_mask']).cuda()
    
    cache_list = [(torch.zeros(args.max_length, 1, config.num_key_value_heads, head_dim).half().to(device=model.device),
               torch.zeros(args.max_length, 1, config.num_key_value_heads, head_dim).half().to(device=model.device))
              for _ in range(config.num_hidden_layers)]

    token_id = Variable(torch.tensor(0).long())

    print(f"==> Start Export Onnx")
    onnx_path = os.path.join(args.output_path, "tinyllama_no_embed.onnx")
    torch_onnx.export(model, (input_embed, retary_cos_sin, attention_mask, cache_list, token_id), onnx_path, opset_version=13, verbose=False,
                        input_names=[f'input_{i}' for i in range(config.num_hidden_layers * 2 + 4)], 
                        output_names=[f'output_{i}' for i in range(config.num_hidden_layers * 2 + 1)])

    # count number of parameters
    total_num = sum(p.numel() for p in model.parameters())
    print('=========================')
    print('Model Token Length: {}'.format(args.max_length))
    print('Total number of parameters is {} '.format(total_num))
    print('TinyLLaMA onnx is saved to {}'.format(onnx_path))

    
if __name__=='__main__':
    main()