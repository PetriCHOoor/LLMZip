
import torch
from tqdm import tqdm
import zlib
import os

import zip_ui


def decompress(file_name: str,
               save_path: str,
               temperature: float = 0.6):

    assert len(file_name) > 0, "File can't be None"

    assert os.path.exists(file_name), f"There's no such file: {file_name}"

    models = ['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'GPT2-XL', 'LLaMA2-7B']
    ranks = []

    # read the compressed data
    with open(file_name, 'rb') as file:
        compressed_data = file.read()
    decompressed_data = zlib.decompress(compressed_data)

    # transfer the binary data to numbers
    for i in range(0, len(decompressed_data), 2):
        rank = int.from_bytes(decompressed_data[i:i + 2], 'little')
        ranks.append(rank)

    # recognize the version of the model and the value of the memory automatically
    version = ranks[0]
    memory = ranks[1]

    if version == 0 or version == 1 or version == 2 or version == 3:
        model = gpt_model

    if version == 4:
        model = llama_model

    # get the memory
    tokens = []
    for i in range(memory):
        tokens.append(ranks[i + 2])

    tokens = torch.tensor([tokens])

    cur_iterator = tqdm(range(0, len(ranks) - memory - 2), desc=f"Decompressing with {models[version]}")

    # LLaMA model decompress
    if version == 4:

        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = model.model.forward(tokens[:, cur_pos:cur_pos + memory], 0)

            probs = torch.softmax(logits[-1, :] / temperature, dim=-1)

            prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)

            token = prob_idx[ranks[memory + cur_pos + 2]].unsqueeze(0).unsqueeze(0)

            tokens = torch.cat((tokens, token), dim=1)

    # GPT2 models decompress
    if version == 0 or version == 1 or version == 2 or version == 3:
        model.model.eval()

        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = model.model(tokens[:, cur_pos:cur_pos + memory])[0][0, -1, :]

            probs = torch.softmax(logits[:] / temperature, dim=-1)
            prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)

            token = prob_idx[ranks[memory + cur_pos + 2]].unsqueeze(0).unsqueeze(0)

            tokens = torch.cat((tokens, token), dim=1)

    # decode tokens
    decompressed_text = model.tokenizer.decode(tokens[0].tolist())

    with open(save_path + 'decompressed.txt', 'w', encoding='utf-8') as file:
        file.write(decompressed_text)

    return decompressed_text