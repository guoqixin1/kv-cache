# ----- 数据准备 -----
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 示例：加载 Self-Instruct 数据集
self_instruct_ds = load_dataset('yizhongw/self_instruct', 'default')
# 假设 LongAlpaca 数据集存在于 HuggingFace
long_alpaca_ds = load_dataset('Yukang/LongAlpaca-12k')
# P3 数据集非常庞大，这里仅示范加载一个子集（实际可选多个任务子集）
# 例如加载 P3 中的一个 QA 任务子集（此处仅为示例，具体任务名称需根据需要替换）
p3_subset_ds = load_dataset('bigscience/P3', 'super_glue/squad')  # 示例使用SQuAD风格任务子集

# 初始化分词器（选择与预训练模型对应的分词器）
model_name = "gpt2"  # 这里选用GPT-2作为示例模型，实际可换为LLaMA等LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT-2 没有明确的pad token，这里将eos作为pad以方便批处理
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 将数据集样本统一转换为 context、question、answer 三元组，并进行分词编码
train_data = []
max_length = 1024  # 根据模型最大长度进行截断（GPT2为1024，实际模型可能更长）
for example in self_instruct_ds['train']:
    # Self-Instruct 数据集中通常有 'instruction', 'input', 'output' 字段
    context_text = example.get('input', '')        # 可能的背景内容
    question_text = example.get('instruction', '') # 指令/问题部分
    answer_text = example.get('output', '')        # 模型应输出的答案
    # 将三部分连接（可以根据需要加入特殊分隔符，这里简单拼接空格）
    # 对于instruction类数据，context可能为空，question就是完整指令
    # 对于有context的数据(如QA)，则context为提供的材料
    # 这里保持三部分独立，训练时分别处理
    # 分词并获取 token ids
    context_ids = tokenizer.encode(context_text, add_special_tokens=False, max_length=max_length//2, truncation=True)
    question_ids = tokenizer.encode(question_text, add_special_tokens=False, max_length=max_length//4, truncation=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, max_length=max_length//4, truncation=True)
    train_data.append({
        "context_ids": context_ids,
        "question_ids": question_ids,
        "answer_ids": answer_ids
    })

# 将LongAlpaca数据也加入训练数据
for example in long_alpaca_ds['train']:
    # LongAlpaca-12k 数据集可能有 'instruction' 和 'output' 字段，以及可能的长上下文
    context_text = example.get('input', '')        # 长上下文
    question_text = example.get('instruction', '') # 问题/指令
    answer_text = example.get('output', '')
    context_ids = tokenizer.encode(context_text, add_special_tokens=False, max_length=max_length//2, truncation=True)
    question_ids = tokenizer.encode(question_text, add_special_tokens=False, max_length=max_length//4, truncation=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, max_length=max_length//4, truncation=True)
    train_data.append({
        "context_ids": context_ids,
        "question_ids": question_ids,
        "answer_ids": answer_ids
    })

# 从P3数据集中选取样本加入训练数据（这里只示范一个子集）
for example in p3_subset_ds['train']:
    # P3子集可能已有格式化的问答，例如 'inputs_pretokenized' 和 'targets_pretokenized'
    context_text = example.get('inputs_pretokenized', '')   # 包含上下文+问题
    # 对于类似SQuAD的任务，inputs_pretokenized可能包含问题和一段上下文
    # 这里简单假设整个inputs是context和question的组合，answer在targets
    # 实际需要根据具体任务格式解析，这里做简化处理
    question_text = ""   # 如果inputs本身就包含问题，这里不单独提取
    # 将inputs拆成context和question（具体实现需根据实际格式，这里简化为没有单独context）
    answer_text = example.get('targets_pretokenized', '')
    context_ids = tokenizer.encode(context_text, add_special_tokens=False, max_length=max_length//2, truncation=True)
    question_ids = tokenizer.encode(question_text, add_special_tokens=False, max_length=max_length//4, truncation=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False, max_length=max_length//4, truncation=True)
    train_data.append({
        "context_ids": context_ids,
        "question_ids": question_ids,
        "answer_ids": answer_ids
    })

print(f"训练样本数: {len(train_data)}")
# 打印一个示例的数据长度
print("示例样本 token 数量:", len(train_data[0]['context_ids']),
      len(train_data[0]['question_ids']), len(train_data[0]['answer_ids']))


# ----- LoRA 适配器设置 -----
from peft import LoraConfig, get_peft_model

# 加载预训练的因果语言模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.eval()  # 基础模型保持在评估模式（不训练原始权重）

# 配置 LoRA 参数
lora_rank = 8         # LoRA rank (秩)
lora_alpha = 32       # LoRA alpha (缩放因子)
lora_dropout = 0.1    # LoRA dropout 概率
target_modules = ["c_attn"]  # 选择LoRA作用的模块。对于GPT-2，"c_attn"是QKV投影层
# 如果是其他模型例如LLaMA，可使用 ["q_proj","v_proj"] 等名称

config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    bias="none",        # 不调节偏置
    task_type="CAUSAL_LM"
)
# 将 LoRA adapter 应用到模型
student_model = get_peft_model(base_model, config)
student_model.train()  # LoRA 模型切换到训练模式（仅LoRA权重参与训练）

# 冻结原始模型权重（一般get_peft_model已将其requires_grad设为False，这里再次确保）
for param in student_model.base_model.parameters():
    param.requires_grad = False

# 验证仅 LoRA 层的参数需要梯度
trainable_params = [n for n,p in student_model.named_parameters() if p.requires_grad]
print("需要训练的参数数:", sum(p.numel() for p in student_model.parameters() if p.requires_grad))
print("可训练参数列表:", trainable_params)

# 定义scorer网络，用于对上下文每个token的重要程度打分
hidden_size = student_model.config.hidden_size  # 模型隐层维度
scorer = torch.nn.Linear(hidden_size, 1, bias=False)  # 简单线性层，将每个token的表示映射为分数
# 初始化scorer，最好初始化为略偏向均匀分数
torch.nn.init.constant_(scorer.weight, 0.0)
scorer.train()


# ----- 模型训练 -----
from torch.optim import AdamW

# 准备优化器，只优化LoRA和scorer参数
learning_rate = 1e-4
weight_decay = 0.01
optimizer = AdamW(list(student_model.parameters()) + list(scorer.parameters()),
                  lr=learning_rate, weight_decay=weight_decay)

# 训练超参数
batch_size = 8
num_epochs = 3
compress_ratio = 0.2  # KV保留比例，如保留20%的token (可根据需要调整)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)
scorer.to(device)
# teacher模型也加载到设备（如果设备显存不足，可将teacher放CPU，每步前向后移动输出到GPU）
teacher_model = base_model.to(device)
teacher_model.eval()  # 教师模型不训练

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i: i + batch_size]
        if not batch:
            continue
        # 动态批处理: 找出该batch中 teacher 输入序列和 student 输入序列 的最大长度，用于padding
        teacher_inputs = []
        student_inputs = []
        teacher_attn_masks = []
        student_attn_masks = []
        # 记录每条样本 context/question/answer 长度以便切分
        context_lens = []
        question_lens = []
        answer_lens = []
        selected_context_lens = []  # 压缩后context长度
        for sample in batch:
            context_ids = sample["context_ids"]
            question_ids = sample["question_ids"]
            answer_ids = sample["answer_ids"]
            # 限制总长度不超过模型允许长度（必要时可截断或跳过过长样本）
            total_len = len(context_ids) + len(question_ids) + len(answer_ids)
            if total_len > max_length:
                # 简单截断策略：优先截断上下文
                overflow = total_len - max_length
                context_ids = context_ids[:-overflow] if overflow < len(context_ids) else []
                total_len = len(context_ids) + len(question_ids) + len(answer_ids)
            # 计算当前样本应保留token数
            k = max(1, int(len(context_ids) * compress_ratio))
            # 利用教师模型获取上下文每个token的隐状态用于打分
            with torch.no_grad():
                out = teacher_model(torch.tensor(context_ids + [tokenizer.eos_token_id]),
                                    output_hidden_states=True, return_dict=True)
                # out.hidden_states[-1] 为最后一层的隐状态，形状 (seq_len, hidden_size)
                # 注意我们只想要context部分的隐状态（不包含eos）
                context_hidden = out.hidden_states[-1][:-1]  # 移除最后的eos的hidden state
            context_hidden = context_hidden.to(device)
            # 计算重要性分数
            scores = scorer(context_hidden)  # (context_len, 1)
            scores = scores.squeeze(-1)  # (context_len,)
            # 选择分数最高的k个token索引
            if k < len(scores):
                topk = torch.topk(scores, k)  # 默认返回最大值
                top_indices = topk.indices.sort().values.tolist()  # 排序索引
            else:
                top_indices = list(range(len(scores)))  # 如果k不小于context长度，则全部保留
            # 根据选择的索引提取保留的token
            selected_context_ids = [context_ids[idx] for idx in top_indices]
            # 构造教师模型的输入ids和学生模型的输入ids（教师用完整context，学生用压缩后的context）
            teacher_input_ids = context_ids + question_ids + answer_ids
            student_input_ids = selected_context_ids + question_ids + answer_ids
            context_lens.append(len(context_ids))
            question_lens.append(len(question_ids))
            answer_lens.append(len(answer_ids))
            selected_context_lens.append(len(selected_context_ids))
            teacher_inputs.append(torch.tensor(teacher_input_ids, dtype=torch.long))
            student_inputs.append(torch.tensor(student_input_ids, dtype=torch.long))
        # Padding本batch序列
        teacher_inputs = torch.nn.utils.rnn.pad_sequence(teacher_inputs, batch_first=True,
                                                         padding_value=tokenizer.pad_token_id)
        student_inputs = torch.nn.utils.rnn.pad_sequence(student_inputs, batch_first=True,
                                                         padding_value=tokenizer.pad_token_id)
        # 生成attention mask，pad位置为0
        teacher_attn_masks = (teacher_inputs != tokenizer.pad_token_id).long()
        student_attn_masks = (student_inputs != tokenizer.pad_token_id).long()
        teacher_inputs = teacher_inputs.to(device)
        student_inputs = student_inputs.to(device)
        teacher_attn_masks = teacher_attn_masks.to(device)
        student_attn_masks = student_attn_masks.to(device)

        # 获取教师模型输出 (不需要计算梯度)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=teacher_inputs, attention_mask=teacher_attn_masks,
                                            return_dict=True)
            teacher_logits = teacher_outputs.logits  # (batch, seq_len_T, vocab_size)
        # 获取学生模型输出 (LoRA适配器模型，有梯度)
        student_outputs = student_model(input_ids=student_inputs, attention_mask=student_attn_masks, return_dict=True)
        student_logits = student_outputs.logits  # (batch, seq_len_S, vocab_size)

        # 计算 KL 散度损失 (逐token，与教师输出分布匹配)
        # 我们比较每个样本答案部分每个token的预测分布
        losses = []
        for j in range(len(batch)):  # 遍历batch内每个样本
            Lc = context_lens[j]
            Lq = question_lens[j]
            La = answer_lens[j]
            Lc_compressed = selected_context_lens[j]
            # 答案在教师序列中的起始索引和终止索引（不含终止）
            teacher_start_idx = Lc + Lq
            teacher_end_idx = Lc + Lq + La
            # 答案在学生序列中的起始和终止索引
            student_start_idx = Lc_compressed + Lq
            student_end_idx = Lc_compressed + Lq + La
            # 提取对应的logits序列
            # teacher_logits[j, teacher_start_idx:teacher_end_idx] 对应每个答案token的预测（即下一个token概率）
            # student_logits 类似
            t_logits = teacher_logits[j, teacher_start_idx:teacher_end_idx, :]  # (La, vocab)
            s_logits = student_logits[j, student_start_idx:student_end_idx, :]  # (La, vocab)
            # 计算概率分布
            t_probs = torch.nn.functional.softmax(t_logits, dim=-1)
            s_log_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
            # 计算 KL(p_teacher || p_student) = sum(p_teacher * log(p_teacher / p_student))
            # 这里对每个token求KL，然后取平均
            kl_per_token = torch.nn.functional.kl_div(s_log_probs, t_probs, reduction='none')  # (La, vocab)
            # 按词汇维度求和得到每个token位置的KL散度
            kl_per_token = kl_per_token.sum(dim=-1)  # (La,)
            loss_sample = kl_per_token.mean()  # 平均每个token的KL
            losses.append(loss_sample)
        # 平均整个batch的loss
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 定期打印损失
        if (i // batch_size) % 100 == 0:
            print(f"Epoch {epoch + 1}, step {i // batch_size + 1}: loss = {loss.item():.4f}")
    avg_loss = total_loss / (len(train_data) / batch_size)
    print(f"Epoch {epoch + 1} finished, average loss = {avg_loss:.4f}")


# ----- 推理与评估 -----
# 推理示例
student_model.eval()
scorer.eval()
# 准备一个示例上下文和问题（可以使用训练/验证集中的例子）
example_context = "Alice went to Paris in the summer. She loved the city and enjoyed her time there."
example_question = "Where did Alice go in the summer?"
print("Context:", example_context)
print("Question:", example_question)

# 分词
context_ids = tokenizer.encode(example_context, add_special_tokens=False)
question_ids = tokenizer.encode(example_question, add_special_tokens=False)
# 通过scorer选择重要tokens
with torch.no_grad():
    out = student_model.base_model(**tokenizer(example_context, return_tensors='pt').to(device), output_hidden_states=True)
    context_hidden = out.hidden_states[-1][0, :-1, :]  # 获取批中第一个样本（只有一个）且除去最后的EOS的隐状态
    scores = scorer(context_hidden)  # (context_len, 1)
    scores = scores.squeeze(-1)      # (context_len,)
    k = max(1, int(len(context_ids) * compress_ratio))
    if k < len(scores):
        top_indices = torch.topk(scores, k).indices.sort().values.tolist()
    else:
        top_indices = list(range(len(scores)))
selected_context_ids = [context_ids[idx] for idx in top_indices]
# 构造模型输入：压缩后的context + question
input_ids = torch.tensor([selected_context_ids + question_ids]).to(device)
# 生成答案
with torch.no_grad():
    output_ids = student_model.generate(input_ids, max_new_tokens=50, do_sample=False)
generated_answer = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
print("Generated Answer:", generated_answer)

# 在SQuAD上评估模型
squad = load_dataset('squad')['validation']
exact_matches = 0
f1_sum = 0.0

def normalize_text(s: str) -> str:
    """将文本转为小写，去掉标点和多余空格"""
    import string, re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)  # 去掉冠词
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)  # 去标点
    s = " ".join(s.split())  # 去除多余空格
    return s

def compute_f1(pred: str, truth: str) -> float:
    pred_tokens = normalize_text(pred).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    common = (set(pred_tokens) & set(truth_tokens))
    if not common:
        return 0.0
    # 计算精确率和召回率
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

for example in squad:
    context = example['context']
    question = example['question']
    gold_answers = example['answers']['text']  # 可能有多个正确答案（同义表述）
    # 压缩context
    context_ids = tokenizer.encode(context, add_special_tokens=False, truncation=True, max_length=max_length//2)
    question_ids = tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=max_length//4)
    with torch.no_grad():
        out = student_model.base_model(**tokenizer(context, return_tensors='pt').to(device), output_hidden_states=True)
        context_hidden = out.hidden_states[-1][0, :-1, :]
        scores = scorer(context_hidden).squeeze(-1)
        k = max(1, int(len(context_ids) * compress_ratio))
        if k < len(scores):
            top_indices = torch.topk(scores, k).indices.sort().values.tolist()
        else:
            top_indices = list(range(len(scores)))
    selected_context_ids = [context_ids[idx] for idx in top_indices]
    # 模型生成答案
    input_ids = torch.tensor([selected_context_ids + question_ids]).to(device)
    with torch.no_grad():
        output_ids = student_model.generate(input_ids, max_new_tokens=32, do_sample=False)
    pred_answer = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
    # 计算EM和F1（取匹配最佳的参考答案）
    normalized_pred = normalize_text(pred_answer)
    # Exact Match
    em = 0
    for ga in gold_answers:
        if normalize_text(ga) == normalized_pred:
            em = 1
            break
    # F1
    f1 = 0.0
    for ga in gold_answers:
        f1 = max(f1, compute_f1(pred_answer, ga))
    exact_matches += em
    f1_sum += f1

total = len(squad)
print(f"SQuAD Exact Match: {exact_matches/total*100:.2f}%")
print(f"SQuAD F1: {f1_sum/total*100:.2f}%")

# 在QuALITY上评估模型
quality = load_dataset('emozilla/quality', split='validation')  # 假设有validation拆分
correct = 0
total = 0
for example in quality:
    article = example['article']
    question = example['question']
    options = example['options']            # 选项列表，如 ["A", "B", "C", "D"]
    gold_label = example['gold_label']      # 正确选项，例如 "B"
    # 构建提示: 上下文+问题+选项
    prompt = article + "\nQuestion: " + question + "\nOptions:\n"
    for idx, opt in enumerate(options):
        prompt += f"{chr(65+idx)}. {opt}\n"
    prompt += "Answer: "
    # 对提示进行压缩（主要针对article部分）
    article_ids = tokenizer.encode(article, add_special_tokens=False, truncation=True, max_length=max_length)
    question_opt_text = f"Question: {question}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i,opt in enumerate(options)]) + "\nAnswer:"
    question_opt_ids = tokenizer.encode(question_opt_text, add_special_tokens=False, truncation=True, max_length=256)
    # 选择重要的article tokens
    with torch.no_grad():
        out = student_model.base_model(**tokenizer(article, return_tensors='pt').to(device), output_hidden_states=True)
        article_hidden = out.hidden_states[-1][0, :-1, :]
        scores = scorer(article_hidden).squeeze(-1)
        k = max(1, int(len(article_ids) * compress_ratio))
        if k < len(scores):
            top_indices = torch.topk(scores, k).indices.sort().values.tolist()
        else:
            top_indices = list(range(len(scores)))
    selected_article_ids = [article_ids[idx] for idx in top_indices]
    # 生成回答（这里希望模型输出选项字母）
    input_ids = torch.tensor([selected_article_ids + tokenizer.encode("\nQuestion: "+question, add_special_tokens=False) +
                               tokenizer.encode("\nOptions:\n", add_special_tokens=False) +
                               tokenizer.encode("\n".join([f"{chr(65+i)}. {opt}" for i,opt in enumerate(options)]) + "\nAnswer:", add_special_tokens=False)
                              ]).to(device)
    with torch.no_grad():
        output_ids = student_model.generate(input_ids, max_new_tokens=1, do_sample=False)  # 生成1个token（预测选项）
    pred = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
    if pred:
        pred_choice = pred[0].upper()  # 取第一个字符作为模型选择的选项
    else:
        pred_choice = ""
    if pred_choice == gold_label:
        correct += 1
    total += 1

print(f"QuALITY Accuracy: {correct/total*100:.2f}%")


