import os
import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.onnx as torch_onnx
from torch.autograd import Variable
import numpy as np
import onnx 
import onnxruntime
from utils import *

pos_emb_shape = [1,1,2,64]
END_OF_SENTENCE_TOKEN = 13

def main():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--dump_index', type=int, default=None, help='dump kv cache for the i-th token.')
    args = parser.parse_args()
    
    onnx_path = os.path.join(args.output_path, "tinyllama_no_embed.onnx")
    embed_path = os.path.join(args.output_path, "embedding_weight.bin")
    position_path = os.path.join(args.output_path, "pos_files")
    if args.dump_index is not None:
        kvcache_path = os.path.join(args.output_path, "kv_cache")
        if not os.path.exists(kvcache_path):
            os.makedirs(kvcache_path)
            print(f"==> Create kv cache output folder for the {args.dump_index}-th token: {kvcache_path}")
    
    print('==============================')
    print('onnx runtime inference (CPU): could be very slow')
    
    session = onnxruntime.InferenceSession(onnx_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print('==> Load embed weight')
    embed_weight = torch.from_numpy(np.fromfile(embed_path, dtype=np.float16)).reshape(32000, 2048).half().cuda()
    
    
    with torch.no_grad():
        # clear history
        history, kv_caches, count = clear_history(config, args.max_length)
            
        # for multi-turn dialogue
        while True:
            query = input("Chat(input 'clear' to clear chat history):")
            
            # clean history
            if query == 'clear':
                history, kv_caches, count = clear_history(config, args.max_length)
                print('History is empty, start a new chat!')
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
                
            #print('input_ids: ', input_ids)

            input_size = input_ids.shape[-1]
            start_gen_idx = input_ids.shape[-1] - 1
            
            # incremental inference using onnx
            while True:
                model_inputs = prepare_inputs_for_generation(input_ids, count, args.max_length)
                
                # current input id
                current_input_id = model_inputs["input_ids"]
                # data process
                attention_mask = model_inputs['attention_mask']
                # Rotary positional embeddings
                rotary_pos_emb = torch.from_numpy(np.fromfile(os.path.join(position_path, '{}.bin'.format(count)), dtype=np.float16)).reshape(pos_emb_shape)
                
                # input embed
                input_embed = get_input_embed(embed_weight, current_input_id)
                
                input_data = {"input_0": input_embed.cpu().numpy().astype(np.float16),
                            "input_1": rotary_pos_emb.cpu().numpy().astype(np.float16),
                            "input_2": attention_mask.cpu().numpy().astype(np.int32),}
                for n in range(len(kv_caches)):
                    for m in range(2):
                        input_data["input_{}".format(n*2+m+3)] = kv_caches[n][m].astype(np.float16)
                        
                input_data["input_{}".format(2+len(kv_caches)*2+1)] = torch.tensor(count).long().numpy().astype(np.int64)

                # cpu inference
                outputs = session.run(None, input_data)
                
                # update kv cache
                for n in range(len(kv_caches)):
                    for m in range(2):
                        kv_caches[n][m] = outputs[n*2+m+1]

                if (args.dump_index is not None) and (count == args.dump_index):
                    for i, output_i in enumerate(outputs):
                        output_i.tofile(os.path.join(kvcache_path, 'output_{}.bin'.format(i)))
                
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

                if finished_sequences or (count == args.max_length):
                    break

            outputs = input_ids.tolist()[0][input_size:]
            print(outputs)
            response = tokenizer.decode(outputs)
            response = process_response(response)
            print(response)
            history.append({'role': 'assistent', 'content': response})

    
if __name__=='__main__':
    main()