from model_class import LLaMA,GPT2


def build_llama(llama_model_path,device):
    llama_model = LLaMA.build(
        checkpoints_dir=llama_model_path,
        tokenizer_path=llama_model_path + '/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device
    )

    return llama_model



def build_GPT2(model_name, gpt_model_path,device):

    gpt_model = GPT2.build(model_name, gpt_model_path, device)

    return gpt_model


