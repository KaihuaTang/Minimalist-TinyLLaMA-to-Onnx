import os 
from transformers import AutoConfig
import torch
import torch.onnx as torch_onnx
from torch.autograd import Variable
import numpy as np
from safetensors.torch import load_file
import onnx
import onnxruntime

from llama.modeling_llama import LlamaForCausalLM, LlamaRotaryEmbedding

model_path = "/data/pubuser/TinyLlama-1.1B-Chat-v1.0/"
ckpt_path = "/data/pubuser/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
onnx_path = "./outputs/onnx_file/tinyllama_no_embed.onnx"
embed_path = "./outputs/input_embedding_weight.bin"
position_path = "./outputs/pos_files/"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = LlamaForCausalLM(config)
checkpoint = load_file(ckpt_path)
print(model.load_state_dict(checkpoint, strict=False))

model = model.half().cuda()
model = model.eval()

MAX_LENGTH = 1024

# get input embedding 
embed_weight = checkpoint['model.embed_tokens.weight'].data.detach().cpu().half()
# save embedding
print('embed_weight shape{}'.format(embed_weight.shape))
embed_weight.numpy().astype(np.float16).tofile(embed_path)


def get_masks(input_ids, device):
    batch_size, seq_length = input_ids.shape
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device, dtype=torch.bool)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)
    attention_mask = ~attention_mask
    return attention_mask.int()


def get_position_ids(input_ids, device):
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids


def prepare_inputs_for_generation(input_ids: torch.LongTensor, count: int) -> dict:
    batch_size, seq_length = input_ids.shape
    assert batch_size == 1
    PADID = 0

    attention_mask = get_masks(input_ids, device=input_ids.device)

    position_ids = get_position_ids(input_ids, device=input_ids.device)

    # padding to fixed length
    if seq_length <= MAX_LENGTH:
        padding_size = MAX_LENGTH - seq_length
        # input ids
        input_ids_pad = input_ids.new(batch_size, padding_size).fill_(PADID)
        input_ids = torch.cat([input_ids, input_ids_pad], dim=-1)
        # position ids
        position_ids_pad = position_ids.new(batch_size, padding_size).fill_(MAX_LENGTH - 1)
        position_ids = torch.cat([position_ids, position_ids_pad], dim=-1)
        # attention mask
        attention_mask_padded = attention_mask.new(1, 1, MAX_LENGTH, MAX_LENGTH).fill_(1)
        attention_mask_padded[0, 0, :seq_length, :seq_length] = attention_mask
        attention_mask = attention_mask_padded
    else:
        input_ids = input_ids[:, -MAX_LENGTH:]
        position_ids = position_ids[:, -MAX_LENGTH:]
        attention_mask = attention_mask[:, :, -MAX_LENGTH:, -MAX_LENGTH:]

    return {"input_ids": input_ids[:, count].view(1, -1),
            "position_ids": position_ids[:, count].view(1, -1).long(),
            "attention_mask": attention_mask[:, :, count, :].unsqueeze(2)}


model_inputs = prepare_inputs_for_generation(torch.LongTensor([[4790, 4792, 24954]]), count=0)

input_ids = model_inputs['input_ids']
input_embed = Variable(embed_weight[input_ids.reshape(-1).item()].reshape(1, 1, -1)).half().cuda()
position_ids = Variable(model_inputs['position_ids'])

head_dim = int(config.hidden_size / config.num_attention_heads)
rotary_pos_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings=MAX_LENGTH)
cos, sin = rotary_pos_emb(input_embed, seq_len=MAX_LENGTH)
cos = cos.squeeze(1).squeeze(0).half().cuda()  # [seq_len, dim]
sin = sin.squeeze(1).squeeze(0).half().cuda()  # [seq_len, dim]
for i in range(MAX_LENGTH):
    position_ids = torch.arange(MAX_LENGTH, dtype=torch.long).cuda()[i].view(1,-1).long()
    cos_i = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin_i = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    retary_cos_sin = torch.cat([cos_i, sin_i], dim=2).half().cuda()
    retary_cos_sin.cpu().numpy().astype(np.float16).tofile(os.path.join(position_path, '{}.bin'.format(i)))
# rotary_pos_emb = rotary_pos_emb[model_inputs['position_ids']].transpose(0, 1).contiguous()
# rotary_pos_emb = Variable(rotary_pos_emb).half().cuda()
print('retary_cos_sin shape is: {}'.format(str(retary_cos_sin.shape)))
attention_mask = Variable(model_inputs['attention_mask']).cuda()

# cache_list = [(torch.zeros(MAX_LENGTH, 1, 2, 128).half().to(device=model.device),
#                torch.zeros(MAX_LENGTH, 1, 2, 128).half().to(device=model.device))
#               for _ in range(model.config.num_hidden_layers)]

cache_list = [(torch.zeros(MAX_LENGTH, 1, config.num_key_value_heads, head_dim).half().to(device=model.device),
               torch.zeros(MAX_LENGTH, 1, config.num_key_value_heads, head_dim).half().to(device=model.device))
              for _ in range(config.num_hidden_layers)]

token_id = Variable(torch.tensor(0).long())

torch_onnx.export(model, (input_embed, retary_cos_sin, attention_mask, cache_list, token_id), onnx_path, opset_version=13, verbose=False,
                      input_names=[f'input_{i}' for i in range(config.num_hidden_layers * 2 + 4)], 
                      output_names=[f'output_{i}' for i in range(config.num_hidden_layers * 2 + 1)])

# count number of parameters
total_num = sum(p.numel() for p in model.parameters())
print('=========================')
print('Model Token Length: {}'.format(MAX_LENGTH))
print('Total number of parameters is {} '.format(total_num))
print('Checkpoint Path: {}'.format(ckpt_path))


# inference onnxruntime
print('=========================')
print('onnx runtime inference')
session = onnxruntime.InferenceSession(onnx_path)
print('session complete')

input_data = {"input_0": input_embed.cpu().numpy().astype(np.float16),
              "input_1": retary_cos_sin.cpu().numpy().astype(np.float16),
              "input_2": attention_mask.cpu().numpy().astype(np.int32),}
for n in range(len(cache_list)):
    for m in range(2):
        input_data["input_{}".format(n*2+m+3)] = cache_list[n][m].detach().cpu().numpy().astype(np.float16)
        
input_data["input_{}".format(2+len(cache_list)*2+1)] = token_id.numpy().astype(np.int64)

# NPU推理
#logits, kv_caches = model(input_embed, rotary_pos_emb, attention_mask, kv_caches, count)
outputs = session.run(None, input_data)

print('onnx runtime finished')
logits = outputs[0]
print(logits)
