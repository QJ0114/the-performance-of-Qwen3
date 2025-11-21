import argparse
import json
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
# 数据集文件夹路径
DATA_PATH = "/home/linyeli/cqj/exercise/gsm8k_attempt/Gsm8k"
# 模型路径
MODEL_PATH = "/home/linyeli/cqj/exercise/gsm8k_attempt/Qwen3-8B"
# 测试文件
TEST_FILE_NAME = "test.jsonl" 
# ===========================================

# ⚠️ 关键修改：使用标准 4-shot CoT Prompt
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
    """从文本中提取最终的数字答案"""
    text = text.replace(',', '').replace('$', '')
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1] 
    return None

def is_correct(pred_str, gold_str):
    """比较预测值和标准答案"""
    if "####" in gold_str:
        gold_val_str = gold_str.split("####")[-1].strip()
    else:
        gold_val_str = extract_answer_number(gold_str)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--test-file", type=str, default=TEST_FILE_NAME)
    parser.add_argument("--max-samples", type=int, default=0, help="只运行前 N 个样本；0 表示全部")
    parser.add_argument("--dry-run-tokenizer", action="store_true", help="仅运行 tokenizer 流程用于调试")
    args = parser.parse_args()

    # 1. 加载模型和分词器
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = None
    if not args.dry_run_tokenizer:
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )

    # 2. 加载数据
    full_data_path = os.path.join(args.data_path, args.test_file)
    if not os.path.exists(full_data_path):
        print(f"未找到 {full_data_path}，正在搜索目录下的其他文件...")
        files = [f for f in os.listdir(args.data_path) if f.endswith('json') or f.endswith('jsonl')]
        if not files:
            raise FileNotFoundError("未找到数据文件")
        full_data_path = os.path.join(args.data_path, files[0])
    
    questions, gold_answers = load_data(full_data_path)
    print(f"共加载 {len(questions)} 条测试数据。")

    # 3. 开始推理和评估
    correct_count = 0
    total_count = 0
    
    test_questions = questions 
    test_answers = gold_answers

    if args.max_samples and args.max_samples > 0:
        test_questions = test_questions[: args.max_samples]
        test_answers = test_answers[: args.max_samples]

    print("开始评估 (Base Model 4-shot Mode)...")
    with tqdm(total=len(test_questions)) as pbar:
        for i, question in enumerate(test_questions):
            # =====================================================
            # 修改点：使用 4-shot 文本拼接，不使用 Chat Template
            # =====================================================
            prompt_text = GSM8K_COT_PROMPT.format(question=question)
            
            # 纯文本 tokenize，不需要 apply_chat_template
            model_inputs = tokenizer([prompt_text], return_tensors="pt")

            if args.dry_run_tokenizer or model is None:
                # Dry run 逻辑
                input_ids = model_inputs.get("input_ids")
                print(f"[DRY-RUN] sample {i}: input_ids shape={tuple(input_ids.shape)}")
                if input_ids is not None:
                    # 打印最后一部分看看是不是符合预期
                    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    print("[DRY-RUN] prompt tail:", decoded[-200:]) 
                response = ""
            else:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device("cpu")

                model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

                with torch.no_grad():
                    raw_outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=512, # 512 对 Base 模型输出答案通常够了
                        temperature=0.0,    # 贪婪解码
                        do_sample=False
                    )

                input_len = model_inputs['input_ids'].shape[1]
                gen_suffix = raw_outputs[:, input_len:]
                full_response = tokenizer.batch_decode(gen_suffix, skip_special_tokens=True)[0]
                
                # =====================================================
                # 修改点：手动截断
                # Base 模型可能会继续生成 "Question: ..."，必须切掉
                # =====================================================
                response = full_response.split("Question:")[0].strip()

            # 评估
            is_right = is_correct(response, test_answers[i])
            if is_right:
                correct_count += 1
            total_count += 1
            
            pbar.set_postfix({"Acc": f"{correct_count/total_count:.2%}"})
            pbar.update(1)
            
            # 调试：打印第一条看看格式
            if i == 0 and not args.dry_run_tokenizer:
                 print(f"\n--- Sample 0 Debug ---")
                 print(f"Pred Cleaned: {response}")
                 print(f"Gold: {test_answers[i]}")
                 print("----------------------")

    # 4. 最终结果
    accuracy = (correct_count / total_count) if total_count > 0 else 0.0
    print("="*30)
    print(f"最终结果 (Base 4-shot):")
    print(f"总样本数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.2%}")
    print("="*30)

if __name__ == "__main__":
    main()