#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from peft import LoraConfig, TaskType, get_peft_model

# 定义路径
mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora_0808'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)

# 加载LoRA权重
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = PeftModel.from_pretrained(model, lora_path, config=config)


# In[2]:


import json
# 加载验证数据集
with open("./datasets/test_data.json", "r", encoding="utf-8") as f:
    valid_data = json.load(f)



# In[4]:


# 提取输入文本和标签
valid_texts = [item['text'] for item in valid_data]
valid_labels = [item['label'] for item in valid_data]
# 映射标签到ID
label_map = { "广告": 0, "高质量": 1, "其他": 2}
valid_label_ids = [label_map[label] for label in valid_labels]



# In[7]:


prompt_template = {
'classifier':"""
###分类标准: 
\n1. 广告: 意图为推销商品、鼓舞人购买的言论，包含有显式的广告格式的言论和隐式推销商品的言论。\
\n2. 高质量: 涉政、涉意识形态的言论，无论言论什么倾向，涉及即算此类。
\n3. 其他: 不涉及广告和高质量的数据都可视为其他。
\n\n优先级：高质量＞广告＞ 其他

###正式分析：结合以上分类标准，请一步一步的思考。现在开始正式分类，你要分类的数据如下{prompt}。 你的答案是：
"""
}


# In[8]:


def chat(query, mode, model, tokenizer, accelerator):
    import time
    accelerator.wait_for_everyone()
    device = 'cuda'
    start_time = time.time()
    prompt = prompt_template[mode].format(prompt = query)
    messages = [
    {"role": "system", "content": "将以下数据划分为：广告,高质量,其他"},
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    params = {
    "max_new_tokens": 512,
    "top_p": 0.9,
    "temperature": 0.25,
    "repetition_penalty": 1.0,
    "do_sample": False}
    with accelerator.split_between_processes(model_inputs.input_ids) as input_ids:
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=512
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time = time.time()
    print('用时：', end_time-start_time)
    used_time = end_time-start_time
    return response


# In[9]:


from tqdm import tqdm
predictions = []
batch_size = 16  # 定义批处理大小
mode ="classifier"
from accelerate import Accelerator
accelerator = Accelerator()
for i in tqdm(range(0, len(valid_texts)), desc="Generating predictions"):
    query = valid_texts[i]
    response = chat(query, mode, model, tokenizer, accelerator)
    
    predictions.append((i,response))



# In[13]:


from sklearn.metrics import confusion_matrix

# 提取预测类别和真实标签
labels = ["广告", "高质量","其他"]
conf_matrix = confusion_matrix(valid_labels, [label for _, label in predictions], labels=labels)

# 计算指标
TP = conf_matrix.diagonal()
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP
TN = conf_matrix.sum() - TP - FP - FN

accuracy = (TP + TN) / (TP + FP + FN + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy.mean())
print("Precision:", precision.mean())
print("Recall:", recall.mean())
print("F1 Score:", f1.mean())


# In[ ]:




