import torch

def get_input_embed(embed_weight, input_ids):
    index = input_ids.reshape(-1).item()
    embed = embed_weight[index].reshape(1,1,-1)
    return embed

def clear_history(config, max_length):
    head_dim = int(config.hidden_size / config.num_attention_heads)
    # 初始化空history
    history = []
    # 初始化空cache
    kv_caches = [[torch.zeros(max_length, 1, config.num_key_value_heads, head_dim).half().cpu().numpy(),
               torch.zeros(max_length, 1, config.num_key_value_heads, head_dim).half().cpu().numpy()]
              for _ in range(config.num_hidden_layers)]
    # 当前推理位置0（没有任何历史）
    count = 0
    return history, kv_caches, count

def process_response(response):
    response = response.strip()
    response = response.strip('</s>')
    return response

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


def prepare_inputs_for_generation(input_ids: torch.LongTensor, count: int, max_length: int = 1024) -> dict:
    batch_size, seq_length = input_ids.shape
    assert batch_size == 1
    PADID = 0

    attention_mask = get_masks(input_ids, device=input_ids.device)

    position_ids = get_position_ids(input_ids, device=input_ids.device)

    # padding to fixed length
    if seq_length <= max_length:
        padding_size = max_length - seq_length
        # input ids
        input_ids_pad = input_ids.new(batch_size, padding_size).fill_(PADID)
        input_ids = torch.cat([input_ids, input_ids_pad], dim=-1)
        # position ids
        position_ids_pad = position_ids.new(batch_size, padding_size).fill_(max_length - 1)
        position_ids = torch.cat([position_ids, position_ids_pad], dim=-1)
        # attention mask
        attention_mask_padded = attention_mask.new(1, 1, max_length, max_length).fill_(1)
        attention_mask_padded[0, 0, :seq_length, :seq_length] = attention_mask
        attention_mask = attention_mask_padded
    else:
        input_ids = input_ids[:, -max_length:]
        position_ids = position_ids[:, -max_length:]
        attention_mask = attention_mask[:, :, -max_length:, -max_length:]

    return {"input_ids": input_ids[:, count].view(1, -1),
            "position_ids": position_ids[:, count].view(1, -1).long(),
            "attention_mask": attention_mask[:, :, count, :].unsqueeze(2)}