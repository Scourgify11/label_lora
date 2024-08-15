#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig


# In[2]:


# 将JSON文件转换为CSV文件
df = pd.read_json('./datasets/ft_data.json')
ds = Dataset.from_pandas(df)


# In[3]:


print(ds[:3])


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# In[5]:


tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token_id


# In[6]:


def process_func(example):
    MAX_LENGTH = 888    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# In[7]:


tokenized_id = ds.map(process_func, remove_columns=ds.column_names)



# In[8]:


print(tokenizer.decode(tokenized_id[0]['input_ids']))


# In[9]:


tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))


# In[10]:


import torch

model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)


# In[11]:


model.enable_input_require_grads()



# In[13]:


from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


# In[14]:


model = get_peft_model(model, config)


# In[15]:


model.print_trainable_parameters()


# In[16]:


args = TrainingArguments(
    output_dir="./output/llama3_0808",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=8,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)


# In[17]:


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)


# In[18]:


trainer.train()


# In[ ]:


peft_model_id="./llama3_lora_0808"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)


# In[ ]:




