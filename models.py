from huggingface_hub import InferenceClient

class ModelConfigs:
    Meta_Llama_3 = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "token": "LLAMA_3_8B_INSTRUCT"
    }

def get_client_from_env(model_config, file):
    kvs = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _k, _v = line.split('=')
            kvs[_k] = _v

    label = model_config["token"]
    if label not in kvs:
        raise KeyError(f'Label {label} not found in model configs')
    model_config["token"] = kvs[label]

    return InferenceClient(
        model=model_config["model"],
        token=model_config["token"]
    )


client = get_client_from_env(ModelConfigs.Meta_Llama_3, ".env.model")

def answer(question: str) -> str:
    return client.chat_completion([
        {"role": "user", "content": question}
    ], max_tokens=500).choices[0].message.content
