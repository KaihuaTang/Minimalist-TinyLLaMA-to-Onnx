import os
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.onnx as torch_onnx
from torch.autograd import Variable
import numpy as np
import onnx 
import onnxruntime

model_path = "/data/pubuser/TinyLlama-1.1B-Chat-v1.0/"
ckpt_path = "/data/pubuser/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
onnx_path = "./outputs/onnx_file/tinyllama_no_embed.onnx"
embed_path = "./outputs/input_embedding_weight.bin"
position_path = "./outputs/pos_files/"
kv_path = "./outputs/kv_cache/"

DUMP_PYTORCH = True
DUMP_INDEX = 5
MAX_LENGTH = 1024
pos_emb_shape = [1,1,2,64]
END_OF_SENTENCE_TOKEN = 13

print('==============================')
print('====== 该onnx仅在cpu推理，故比较慢 ======')
print('onnx runtime inference')
session = onnxruntime.InferenceSession(onnx_path)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
embed_weight = torch.from_numpy(np.fromfile(embed_path, dtype=np.float16)).reshape(32000, 2048).half().cuda()

def get_input_embed(input_ids):
    index = input_ids.reshape(-1).item()
    embed = embed_weight[index].reshape(1,1,-1)
    return embed

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
    
    
head_dim = int(config.hidden_size / config.num_attention_heads)

def clear_history():
    # 初始化空history
    history = []
    # 初始化空cache
    kv_caches = [[torch.zeros(MAX_LENGTH, 1, config.num_key_value_heads, head_dim).half().cpu().numpy(),
               torch.zeros(MAX_LENGTH, 1, config.num_key_value_heads, head_dim).half().cpu().numpy()]
              for _ in range(config.num_hidden_layers)]
    # 当前推理位置0（没有任何历史）
    count = 0
    return history, kv_caches, count

def process_response(response):
    response = response.strip()
    response = response.strip('</s>')
    return response
    

with torch.no_grad():
    # 初始化空history
    history, kv_caches, count = clear_history()
        
    # 这个while是多轮对话
    while True:
        query = input('Chat（输入clear清空历史）:')
        
        # 下面操作可以清楚历史记录
        if query == 'clear':
            history, kv_caches, count = clear_history()
            print('历史已经清空，请重新开始问答')
            continue
        
        if len(history) > 0:
            history.append({'role': 'user', 'content': query})
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            print('prompt stream:\n{}'.format(prompt))
            input_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
            input_ids = torch.LongTensor([input_ids])
        else:
            # add system
            history.append({'role': 'system', 'content': 'You are a friendly and helpful chatbot.'})
            history.append({'role': 'user', 'content': query})
            prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            print('prompt stream:\n{}'.format(prompt))
            input_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
            input_ids = torch.LongTensor([input_ids])
            
        print('input_ids: ', input_ids)

        # 这个index之前都是输入信息（上下文+问题），模型在这个index之前的输出都弃用
        input_size = input_ids.shape[-1]
        start_gen_idx = input_ids.shape[-1] - 1
        
        # 这个while是没一轮中，一个词一个词的推理
        while True:
            #print('=================== Start Count: {} =================='.format(count))
            # 模型预处理，pad到最大长度1024，已经提取出当前字对应位置的输入
            model_inputs = prepare_inputs_for_generation(input_ids, count)
            
            # current input id
            current_input_id = model_inputs["input_ids"]
            # data process
            attention_mask = model_inputs['attention_mask']
            # Rotary positional embeddings
            rotary_pos_emb = torch.from_numpy(np.fromfile(os.path.join(position_path, '{}.bin'.format(count)), dtype=np.float16)).reshape(pos_emb_shape)
            
            # input embed
            input_embed = get_input_embed(current_input_id)
            
            input_data = {"input_0": input_embed.cpu().numpy().astype(np.float16),
                          "input_1": rotary_pos_emb.cpu().numpy().astype(np.float16),
                          "input_2": attention_mask.cpu().numpy().astype(np.int32),}
            for n in range(len(kv_caches)):
                for m in range(2):
                    input_data["input_{}".format(n*2+m+3)] = kv_caches[n][m].astype(np.float16)
                    
            input_data["input_{}".format(2+len(kv_caches)*2+1)] = torch.tensor(count).long().numpy().astype(np.int64)

            # NPU推理
            outputs = session.run(None, input_data)
            
            # update kv cache
            for n in range(len(kv_caches)):
                for m in range(2):
                    kv_caches[n][m] = outputs[n*2+m+1]

            if count == DUMP_INDEX:
                for i, output_i in enumerate(outputs):
                    output_i.tofile(os.path.join(kv_path, 'output_{}.bin'.format(i)))
            
            next_token_logits = torch.from_numpy(outputs[0])[:, -1, :]
            # 现在去最大位置作为预测token
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            if count >= start_gen_idx: 
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                finished_sequences = (int(next_tokens) == END_OF_SENTENCE_TOKEN)
                # decode
                outputs = input_ids.tolist()[0][input_size:]
                print(outputs)
                response = tokenizer.decode(outputs)
                response = process_response(response)
                print(response)
            else:
                finished_sequences = False
            print('Count: {}, Predict: {}'.format(count, next_tokens))
            count += 1

            if finished_sequences or (count == MAX_LENGTH):
                break

        outputs = input_ids.tolist()[0][input_size:]
        print(outputs)
        response = tokenizer.decode(outputs)
        response = process_response(response)
        print(response)
        history.append({'role': 'assistent', 'content': response})
