## 声明
此仓库是论文：[KV-Distill: Nearly Lossless Learnable Context Compression for LLMs](https://arxiv.org/pdf/2503.10337) 的初步实现

其官方仓库为：https://github.com/vnchari/kv-distill

目前（2025/03/17）还未上传代码

此仓库当前处于草稿状态，代码由AI工具完成，并不保证能运行，也可能与原文思路存在误差，仅供学习参考使用，后续可能会进行人工复现，也欢迎PR

This repository is currently in a draft state, and the code is completed by AI tools. It cannot be guaranteed to run and may have errors in the original idea. 

It is only for learning and may be manually reproduced in the future. Please feel free to submit a PR

## 数据准备
首先，加载并预处理训练数据，包括 Self-Instruct、P3、LongAlpaca 等数据集。将每条数据划分为 context（上下文）、question（提问）和 answer（回答）三部分，并使用 Hugging Face 的 datasets 和 transformers 工具将文本转换为模型可用的 token IDs。为了简洁起见，这里演示性地加载部分数据，并进行必要的预处理。注意实际处理中可能需要根据内存情况对 P3 等大型数据集进行采样或拆分。

## KV-DISTILL 方法实现
下面实现 KV-DISTILL 方法的关键组件，包括 KV 缓存压缩（选择重要 token）和 LoRA 适配器，以及 KL 散度损失的计算。
### LoRA 适配器设置
使用 Hugging Face 的 PEFT 库将 LoRA 插入预训练模型，以实现参数高效微调。我们为模型的部分权重（如自注意力层）添加 LoRA 权重，并冻结原始模型权重。从论文描述推测，可以对多头自注意力的投影矩阵（如 Wq, Wk, Wv 或合并后的 c_attn）应用 LoRA，以便在压缩上下文时调整重要 token 的表示。这里以 GPT-2 模型为例，将 LoRA 应用于所有 Transformer Block 中的自注意力输入投影层。

## 模型训练
使用 AdamW 优化器对 LoRA 参数和 scorer 参数进行训练。训练过程中，对于每个批次的数据，我们按以下步骤计算损失：
1. 前向计算（Teacher模型）：将完整上下文+问题+答案输入 教师模型（未压缩上下文）获取每个位置的输出分布（仅用于提供目标分布，不更新梯度）。
2. 前向计算（Student模型）：用 学生模型（应用LoRA后的模型）处理压缩后的上下文+问题+答案，获取输出分布。
3. KV缓存压缩：通过 scorer 对上下文每个位置打分，选择得分最高的若干 token 保留（例如保留一定比例的重要token），丢弃其余token，从而构建压缩后的上下文。
4. KL散度损失：计算学生模型输出分布相对于教师模型输出分布的 KL 散度，鼓励学生（压缩上下文）与教师（完整上下文）的下一个token预测分布尽可能接近。这实现了知识蒸馏，使压缩后的KV缓存保留原上下文的大部分信息。
5. 反向传播和优化：仅更新 LoRA 参数和 scorer 参数。

超参数包括 LoRA的秩 (lora_rank)，学习率 (learning_rate)，batch size等。我们在优化器中仅传入可训练参数（LoRA权重和scorer权重）。下面的代码实现了上述训练流程。注意为了简洁，我们将每个batch大小设小，实际可根据硬件调整，并使用梯度累积或分布式训练加速。训练过程中会周期性打印损失以观察收敛情况。

以上训练过程会调整 LoRA 层的参数和 scorer 的权重，使得学生模型在仅利用压缩后的 KV 缓存（少量重要 token）时，其对下一个token的预测分布尽可能接近在完整上下文下的预测分布。这样经过若干轮训练后，模型学会高效利用保留下来的关键信息，从而在大幅压缩上下文的情况下仍保持接近原模型的性能。

## 推理与评估

训练完成后，我们使用压缩后的 KV 缓存进行推理，并在基准数据集上评估模型性能。推理时，对于给定输入文本（上下文+问题），先通过 scorer 选择重要的 token 构建压缩上下文，然后使用训练后的学生模型（包含 LoRA 适配器）进行回答生成。模型可以按正常的自回归方式生成答案，而因为上下文已被压缩，所以推理开销降低。同时，由于在训练中对齐了分布，生成的答案应与使用完整上下文时接近。 下面展示推理的流程，以及在 SQuAD 和 QuALITY 数据集上的评估方法：
1. 推理示例：给定一个上下文和问题，利用压缩后的上下文生成答案。
2. SQuAD评估：对 SQuAD 数据集的问答进行生成，计算预测答案与参考答案之间的 Exact Match (EM) 和 F1-score。
3. QuALITY评估：对于 QuALITY 数据集（长文多项选择阅读理解），提供文章、问题和选项，生成模型的选择，并计算 准确率。

在以上评估中，我们首先对 SQuAD 验证集进行了评测，计算了模型预测答案与标准答案的 Exact Match 和 F1 分数。随后，我们对 QuALITY 验证集进行评测，对于每道长文多项选择题，我们让模型读取压缩后的文章和问题选项，输出一个选项字母，并计算选择正确的比例（准确率）。 通过这些评估，可以验证 KV-DISTILL 方法在下游任务上的效果。例如，论文中报告在 SQuAD 上压缩到保留25%上下文时，模型准确率仅下降几个百分点；在 QuALITY 上即使将7k长度的文章压缩到仅7个token，模型准确率相比未压缩时只下降约20个百分点，体现了该方法在长上下文压缩下仍能保持接近原始模型的性能。
