
import torch
from tqdm import tqdm
import zlib
import os

def get_unique_filename(base_path):

    filename, file_extension = os.path.splitext(base_path)
    print(filename,file_extension)
    counter = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = f"{filename}({counter}){file_extension}"
        counter += 1

    return new_path

def compress_zlib(ranks,
                  file_name,
                  save_path: str):
    byte_data = bytearray()
    for num in ranks:
        byte_data.extend(num.to_bytes(2, 'little'))  # 每个整数转换为2字节的字节串

    compressed_data = zlib.compress(byte_data)

    base_path = save_path + '/' + os.path.basename(file_name[0:-4]) + '.bin'
    unique_path = get_unique_filename(base_path)

    with open(unique_path, 'wb') as file:
        file.write(compressed_data)


def compress(model,
             file_name: str,
             memory: int,
             device: str,
             save_path: str,
             temperature: float = 0.6):
    assert len(file_name) > 0, "File can't be None"

    assert os.path.exists(file_name), f"There's no such file: {file_name}"

    version = model.version
    # print('model:',version)

    models = ['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'GPT2-XL', 'LLaMA2-7B']

    if model.__class__.__name__ == 'LLaMA':

        with open(file_name, 'r', encoding='utf-8') as file:
            text = [file.read()]

        # print('Currently using model:',models[version])

        tokens = torch.tensor(model.tokenizer.encode(text, out_type=int, add_bos=False, add_eos=False)).to(device)

        assert tokens.shape[
                   1] > memory, f'memory must be smaller than the length of tokens!(memory:{memory},length:{tokens.shape[1]})'

        cur_iterator = tqdm(range(0, len(tokens[0].tolist()) - memory), desc=f"Compressing with {models[version]}")

        ranks = [version, memory]

        for i in range(memory):
            ranks.append(tokens[0][i].item())

        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = model.model.forward(tokens[:, cur_pos:cur_pos + memory], 0)

            probs = torch.softmax(logits[-1, :] / temperature, dim=-1)

            prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)

            token_real_value = tokens[:, cur_pos + memory: cur_pos + memory + 1]

            rank = torch.where(prob_idx == token_real_value)[1].item()

            ranks.append(rank)

        compress_zlib(ranks, file_name, save_path)

    if model.__class__.__name__ == 'GPT2':

        # print('Currently using model:',models[version])

        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()

        tokens = torch.tensor([model.tokenizer.encode(text)]).to(device)

        assert tokens.shape[
                   1] > memory, f'memory must be smaller than the length of tokens!(memory:{memory},length:{tokens.shape[1]})'

        # print('tokens',tokens,tokens.shape)

        ranks = [version, memory]

        model.model.eval()

        for i in range(memory):
            ranks.append(tokens[0][i].item())

        cur_iterator = tqdm(range(0, len(tokens[0].tolist()) - memory), desc=f'Compressing with {models[version]}')

        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = model.model(tokens[:, cur_pos:cur_pos + memory])[0][0, -1, :]

            token_real_value = tokens[:, cur_pos + memory:cur_pos + memory + 1]

            probs = torch.softmax(logits[:] / temperature, dim=-1)
            prob_value, prob_idx = torch.sort(probs, dim=-1, descending=True)
            rank = torch.where(prob_idx == token_real_value)[1].item()

            ranks.append(rank)

        # print("ranks:",ranks)

        compress_zlib(ranks, file_name, save_path)

    return None