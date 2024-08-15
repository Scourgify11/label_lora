from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import json
import pandas as pd

mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora' # lora权重路径

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

prompt_template = {
'classifier':"""
###分类标准: 
\n1. 无语义: 无明显语义的闲聊对话，或无法识别其含义的短文本（胡言乱语），无强烈主题和意图表达。
\n2. 抖音特色: 包含特定于抖音平台，而微信平台不具备的内容。\n例句: '与xxx合拍', '求一个小红心', '我要上推荐'
\n3. 生活: 包含生活中一些主题的对话，主题包括旅游、美食、宠物、时尚、教育、科技、房产等。
\n4. 娱乐: 包含娱乐相关主题，包括游戏、影视、明星、艺术等。
\n5. 情感: 无主题，直接表达情感或者情绪的句子。
\n6. 健康: 包含身体健康和体育（运动健身）类的文本内容。
\n7. 广告: 意图为推销商品、鼓舞人购买的言论，包含有显式的广告格式的言论和隐式推销商品的言论。\
\n8. 高质量: 涉政、涉意识形态的言论，无论言论什么倾向，涉及即算此类。
\n\n优先级：高质量＞广告＞抖音特色|生活|娱乐|健康＞情感＞无语义

###正式分析：结合以上分类标准，请一步一步的思考。现在开始正式分类，你要分类的数据如下{prompt}。 你的答案是：
"""
}

def chat(query, model, tokenizer, accelerator):
    import time
    accelerator.wait_for_everyone()
    device = 'cuda'
    start_time = time.time()
    mode = "classifier"
    prompt = prompt_template[mode].format(prompt = query)
    messages = [
        {"role": "system", "content": "将以下数据划分为：抖音特色,无语义,生活,娱乐,情感,健康,广告,高质量"},
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

from tqdm import tqdm
# 读取CSV文件并提取指定行数据
column_names = ['query', 'response']
valid_texts_df = pd.read_csv("./datasets/zbw3.csv", header=None, names=column_names,encoding="utf-8")

# 定义批处理大小
batch_size = 16

# 使用Accelerator
from accelerate import Accelerator
accelerator = Accelerator()

# 确认DataFrame初始状态
print(valid_texts_df[500:515])
print(valid_texts_df.shape)

# 确保“response”列存在
if 'response' not in valid_texts_df.columns:
    valid_texts_df['response'] = None

for i in tqdm(range(500, 2500, batch_size), desc="Generating predictions"):
    # 提取当前批次的数据
    batch = valid_texts_df.iloc[i:i+batch_size]
    queries = batch['query'].tolist()

    # 进行预测
    responses = [chat(query, model, tokenizer, accelerator) for query in queries]

    # 将预测结果写回DataFrame
    for j, response in enumerate(responses):
        print(f"Index: {i+j}, Query: {queries[j]} - Response: {response}")
        valid_texts_df.loc[i+j, 'response'] = response
        print(valid_texts_df.loc[i+j, 'response'])

# 保存更新后的DataFrame到CSV文件
valid_texts_df.to_csv("./datasets/zbw3labeled.csv", header=None, index=False, encoding="utf-8")
print(valid_texts_df[500:515])  # 打印头部示例数据的预测结果