import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 加载数据
def load_data(train_file, test_file):
    # 创建示例数据文件如果不存在
    if not os.path.exists(train_file):
        print(f"Creating sample {train_file}")
        sample_data = [
            {"context": "Hello, how are you?", "response": "I'm doing well, thank you for asking!"},
            {"context": "What is the capital of France?", "response": "The capital of France is Paris."},
            {"context": "Can you help me with math?", "response": "Of course! I'd be happy to help you with math problems."},
            {"context": "Tell me a joke", "response": "Why don't scientists trust atoms? Because they make up everything!"},
            {"context": "What's the weather like?", "response": "I don't have access to current weather data, but you can check a weather app or website."}
        ]
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    if not os.path.exists(test_file):
        print(f"Creating sample {test_file}")
        sample_test_data = [
            {"context": "Good morning!", "response": "Good morning! How can I help you today?"},
            {"context": "What time is it?", "response": "I don't have access to the current time, but you can check your device's clock."}
        ]
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in sample_test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    try:
        train_data = load_dataset("json", data_files={"train": train_file})["train"]
        test_data = load_dataset("json", data_files={"test": test_file})["test"]
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# 数据预处理
def preprocess_function(examples):
    try:
        # 处理 context 和 response 字段
        contexts = examples["context"]
        responses = examples["response"]
        
        # 确保是列表
        if not isinstance(contexts, list):
            contexts = [contexts]
        if not isinstance(responses, list):
            responses = [responses]
        
        # 创建完整的文本
        texts = []
        for context, response in zip(contexts, responses):
            # 简化格式，直接拼接
            text = f"{context} {response}"
            texts.append(text)
        
        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,  # 减少长度
            return_tensors=None
        )
        
        # 复制input_ids作为labels
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        
        # 只返回模型需要的字段，移除原始字段
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "labels": result["labels"]
        }
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print(f"Example keys: {list(examples.keys()) if hasattr(examples, 'keys') else 'No keys'}")
        raise

# 模型和分词器
model_name = "distilgpt2"  # DistilGPT2模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置pad_token，GPT模型通常没有pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载训练和测试数据
train_file = "train.jsonl"  # 替换为你的训练集文件路径
test_file = "test.jsonl"    # 替换为你的测试集文件路径

train_data, test_data = load_data(train_file, test_file)

# 检查数据结构
print("Train data structure:")
print(f"Keys: {train_data.column_names}")
print(f"First example: {train_data[0] if len(train_data) > 0 else 'No data'}")
print(f"Number of examples: {len(train_data)}")

print("\nTest data structure:")  
print(f"Keys: {test_data.column_names}")
print(f"First example: {test_data[0] if len(test_data) > 0 else 'No data'}")
print(f"Number of examples: {len(test_data)}")

# 数据集预处理
train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)

# 数据收集器 - 使用语言模型专用的收集器
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 对于GPT类模型，不使用掩码语言建模
)

# TrainingArguments 配置
training_args = TrainingArguments(
    output_dir="./results",          # 保存模型的路径
    eval_strategy="epoch",          # 修改为 eval_strategy 而不是 evaluation_strategy
    learning_rate=1e-3,             # 稍微提高学习率
    per_device_train_batch_size=2,  # 进一步减少批量大小
    per_device_eval_batch_size=2,   # 进一步减少批量大小
    num_train_epochs=1,             # 先用1个epoch测试
    weight_decay=0.01,              # 权重衰减
    save_strategy="epoch",          # 保存策略
    logging_dir="./logs",           # 日志保存路径
    logging_steps=10,               # 日志记录间隔
    push_to_hub=False,              # 是否推送到 Hugging Face Hub
    fp16=False,                     # 先关闭半精度训练避免兼容性问题
    dataloader_drop_last=True,      # 丢弃最后一个不完整的批次
    remove_unused_columns=True,     # 移除未使用的列
    do_eval=True                    # 明确启用评估
)

# Trainer 实例化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./fine_tuned_distilgpt2")
tokenizer.save_pretrained("./fine_tuned_distilgpt2")
