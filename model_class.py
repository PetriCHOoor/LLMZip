import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from llama_model import Transformer, ModelArgs

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs, version):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        self.version = version

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        version = 0

        return LLaMA(model, tokenizer, model_args, version)

class GPT2:

    def __init__(self, model, tokenizer, version):
        self.model = model
        self.tokenizer = tokenizer
        self.version = version

    @staticmethod
    def build(model_name, model_dir, device):
        # model_name = os.path.basename(model_dir)
        versions = {'gpt2_small': 1, 'gpt2_medium': 2, 'gpt2_large': 3, 'gpt2_xl': 4}

        version = versions[model_name]

        prev_time = time.time()

        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=model_dir)
        device = torch.device(device)

        model.to(device)

        print(f"Loaded {model_name} in {time.time() - prev_time:.2f}s")

        return GPT2(model, tokenizer, version)
