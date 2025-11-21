import json
import re
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 数据集路径
DATA_PATH = "/home/linyeli/cqj/exercise/gsm8k_attempt/Gsm8k"
# 模型路径
MODEL_PATH = "/home/linyeli/cqj/exercise/gsm8k_attempt/Qwen3-8B"

TEST_FILE_NAME = "test.jsonl"
# ===========================================

# 标准 GSM8K 4-shot Prompt (Standard CoT)
# 来源：GSM8K 原始论文标准设置
GSM8K_COT_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: {question}
Answer:"""

def load_data(file_path):
    """读取 GSM8K 数据集"""
    print(f"正在加载数据: {file_path}")
    questions = []
    answers = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                questions.append(data['question'])
                answers.append(data['answer']) 
            except json.JSONDecodeError:
                continue
    return questions, answers

def extract_answer_number(text):
    """提取答案中的数值"""
    text = text.replace(',', '').replace('$', '')
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1]
    return None

def is_correct(pred_str, gold_str):
    """比对答案"""
    if "####" in gold_str:
        gold_val_str = gold_str.split("####")[-1].strip()
    else:
        gold_val_str = extract_answer_number(gold_str)

    # vLLM 生成的纯文本可能不包含 #### (尽管有 Few-shot 引导它通常会生成)
    # 我们先尝试分割，如果不行就直接在全文找最后一个数字
    if "####" in pred_str:
        pred_val_str = pred_str.split("####")[-1].strip()
    else:
        pred_val_str = extract_answer_number(pred_str)

    try:
        gold_num = float(extract_answer_number(str(gold_val_str)))
        pred_num = float(extract_answer_number(str(pred_val_str)))
        return abs(gold_num - pred_num) < 1e-6
    except (ValueError, TypeError, IndexError):
        return False

def main():
    # 1. 准备数据
    full_data_path = os.path.join(DATA_PATH, TEST_FILE_NAME)
    questions, gold_answers = load_data(full_data_path)
    print(f"共加载 {len(questions)} 条测试数据。")

    # 2. 准备 Prompts (使用 4-shot 拼接，而不是 Chat Template)
    print("正在构建 4-shot Prompts...")
    prompts = []
    for q in questions:
        # 直接格式化字符串
        text = GSM8K_COT_PROMPT.format(question=q)
        prompts.append(text)

    # 3. 初始化 vLLM 引擎
    print("正在初始化 vLLM 引擎...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.90
    )

    # 4. 设置采样参数
    # 关键修改：设置 stop token，防止 Base 模型自己编造下一个问题
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=1024,
        stop=["Question:", "\n\nQuestion:", "<|endoftext|>"] # 遇到下一个"Question:"立即停止
    )

    # 5. 开始极速推理
    print("开始批量推理 (Base Model 4-shot)...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. 评估结果
    correct_count = 0
    total_count = len(questions)

    print("正在计算准确率...")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        
        # 安全措施：再次截断，确保只取 Question: 之前的部分 (双重保险)
        clean_text = generated_text.split("Question:")[0].strip()
        
        if is_correct(clean_text, gold_answers[i]):
            correct_count += 1
        
        # (可选) 调试：打印前 3 个看看格式是否符合预期
        if i < 3:
            print(f"\n--- Sample {i} ---")
            print(f"Pred: {clean_text}") 
            print(f"Gold: {gold_answers[i]}")

    accuracy = correct_count / total_count
    print("="*30)
    print(f"vLLM (Base 4-shot) 评估完成")
    print(f"总样本数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.2%}")
    print("="*30)

if __name__ == "__main__":
    main()