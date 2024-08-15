import pandas as pd
import json

# 定义Quality的映射字典
quality_mapping = {
    '抖音特色': '其他',
    '无语义': '其他',
    '生活': '其他',
    '娱乐': '其他',
    '情感': '其他',
    '健康': '其他',
    '广告': '广告',
    '高质量': '高质量'
}

# 加载训练集和验证集
file_train = pd.read_csv('all_0806_train.csv')
file_val = pd.read_csv('all_0806_val.csv')
file_test = pd.read_csv('all_0806_test.csv')

# 假设StrContent在第0列，Quality在第1列
file_train.columns = ['StrContent', 'Quality']
file_val.columns = ['StrContent', 'Quality']
file_test.columns = ['StrContent', 'Quality']

file_train = file_train.dropna(subset=['StrContent', 'Quality'])
file_val = file_val.dropna(subset=['StrContent', 'Quality'])
file_test = file_test.dropna(subset=['StrContent', 'Quality'])


# 映射Quality列
file_train['Quality'] = file_train['Quality'].map(quality_mapping)
file_val['Quality'] = file_val['Quality'].map(quality_mapping)
file_test['Quality'] = file_test['Quality'].map(quality_mapping)

# # 合并两个DataFrame
# combined_data = pd.concat([file_train, file_val])

# # 去除重复项
# combined_data = combined_data.drop_duplicates()

# instruction = "将以下多条数据划分为：广告,高质量,其他 分类标准: 1. 广告: 意图为推销商品、鼓舞人购买的言论，包含有显式的广告格式的言论和隐式推销商品的言论。 例句: '看到这么多人都喜欢这个法兰绒毛毯你就知道有多舒服了?不管冷气房盖著看电视还是冬天披著窝在沙发也是大享受仅此这档免费送限量的限量的限量的[url]抢完就没啰～#维纳斯五盒就送给你#价值2980正货?', '#??DeSlim好产品高复购#菌菌需求大不大菌菌效果好不好看看我们张医生的诊所?好产品高复购???新加坡??DeSlim益生菌·纤维' 2. 高质量: 涉政、涉意识形态的言论，无论言论什么倾向，涉及即算此类。3.其他: 不属于广告和高质量的均视为其他 优先级：高质量＞广告＞其他"

# # 转换为JSON格式
# data_json = combined_data.apply(lambda row: {'instruction': instruction, 'input': row['StrContent'], 'output': row['Quality']}, axis=1).tolist()
# print(data_json)

# # 写入JSON文件
# with open('ft_data.json', 'w') as f:
#     json.dump(data_json, f, ensure_ascii=False)


test_json = file_test.apply(lambda row: {'text': row['StrContent'], 'label': row['Quality']}, axis=1).tolist()
print(test_json)

with open('test_data.json', 'w') as f:
    json.dump(test_json, f, ensure_ascii=False)