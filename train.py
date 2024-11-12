from db import *
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

tags_dataset = [{"Input": k, "Output": v} for k, v in jobs_df[['job_description', 'tags']].values]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createPrompt(example):
    bos_token = '<s>'
    system_prompt = '[INST] You are a custom AI and your role is to extract required job skill tags from job description \n'
    input_prompt = f" {example['Input']} [/INST]"
    output_prompt = f"{example['Output']} </s>"

    return bos_token + system_prompt + input_prompt + output_prompt

base_model_id = 'mistralai/Mixtral-8x7B-v0.1'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="cuda")

base_model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(base_model)

print(tags_dataset)

def printParameters(model):
    trainable_param = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()

    print(
        f"Total params : {total_params} , trainable_params : {trainable_param} , trainable % : {100 * trainable_param / total_params} ")
