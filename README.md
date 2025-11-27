# HomeBench 复现与微调指南

Reproduction of HomeBench,https://github.com/BITHLP/HomeBench/tree/main

本项目包含 HomeBench 论文代码的修复版本，支持 Qwen2.5 等模型的监督微调 (SFT)、分布式推理测试以及指标评估。

请确保您位于项目的**根目录**下执行所有指令 (例如 `/root/autodl-tmp/HomeBenchProduction`)。

## 目录

1. [环境准备](https://www.google.com/search?q=%231-环境准备)
2. [第一步：监督微调 (SFT)](https://www.google.com/search?q=%232-第一步监督微调-sft)
3. [第二步：模型推理测试](https://www.google.com/search?q=%233-第二步模型推理测试)
4. [第三步：结果评估](https://www.google.com/search?q=%234-第三步结果评估)

## 1. 环境准备

确保已安装必要的依赖库，并正确设置了 HF 镜像（代码中已内置 `HF_ENDPOINT` 设置），并将模型下载到models文件夹下。
hf auth login
export HF_ENDPOINT=https://hf-mirror.com
echo $HF_ENDPOINT

export HF_TOKEN=Your_HF_TOKEN
echo $HF_TOKEN

核心文件结构说明：

```
HomeBenchReproduction/
├── code/
│   ├── sft_fixed.py        # 修复版微调脚本 (解决了 Loss=0 和 Processor 报错问题)
│   ├── model_test_fixed.py # 修复版推理脚本 (支持 DDP 多卡并行)
│   └── eval_fixed.py       # 修复版评估脚本 (支持多种错误类型分析)
├── dataset/                # 数据集存放目录
├── models/                 # 原始模型权重目录
├── model_output/           # (自动生成) 微调后的 LoRA 权重保存目录
└── output/                 # (自动生成) 推理结果和评估报告保存目录
```

## 2. 第一步：监督微调 (SFT)

使用 `code/sft_fixed.py` 对基座模型进行 LoRA 微调。此脚本修复了原版代码中 Tokenizer 模板匹配导致的 Loss 为 0 问题。

### 运行指令

#### 单卡训练

```
torchrun --nproc_per_node=1 code/sft_fixed.py \
    --model_name qwen \
    --batch_size 4 \
    --grad_accum 16 \
    --cuda_devices "0"
```

#### 多卡训练 (DDP)

```
torchrun --nproc_per_node=2 code/sft_fixed.py \
    --model_name qwen \
    --batch_size 4 \
    --grad_accum 16 \
    --cuda_devices "0,1"
```

### 参数说明

| 参数             | 说明            | 建议值                                                   |
| ---------------- | --------------- | -------------------------------------------------------- |
| `--model_name`   | 模型名称        | `qwen`, `llama`, `mistral`, `gemma`                      |
| `--batch_size`   | 单卡 Batch Size | 显存允许的情况下尽可能大 (如 4, 8)                       |
| `--grad_accum`   | 梯度累积步数    | 用于模拟大 Batch。若显存小导致 Batch Size 小，应调大此值 |
| `--cuda_devices` | 指定 GPU 编号   | 例如 `"0"` 或 `"0,1"`                                    |

> **注意**：训练完成后，LoRA 权重将保存在 `model_output/` 目录下。

## 3. 第二步：模型推理测试

使用 `code/model_test_fixed.py` 加载微调后的权重进行推理。脚本会自动检测 `model_output` 目录下的 LoRA 适配器。支持多卡 DDP 并行加速。

### 运行指令

#### 单卡推理 (推荐先测试)

```
torchrun --nproc_per_node=1 code/model_test_fixed.py \
    --model_name qwen \
    --batch_size 16 \
    --cuda_devices "0"
```

#### 多卡并行推理 (加速)

```
torchrun --nproc_per_node=2 code/model_test_fixed.py \
    --model_name qwen \
    --batch_size 32 \
    --cuda_devices "0,1"
```

### 参数说明

| 参数               | 说明                      | 建议值                                |
| ------------------ | ------------------------- | ------------------------------------- |
| `--model_name`     | 模型名称 (需与训练时一致) | `qwen` 等                             |
| `--batch_size`     | 推理 Batch Size           | A800/A100 可设 32/64；24G 显存设 8/16 |
| `--test_type`      | 测试任务类型              | 默认为 `normal`，会影响输出文件名     |
| `--nproc_per_node` | 进程数                    | **必须等于** 使用的显卡数量           |

> **注意**：运行结束后，结果会自动合并并保存在 `output/` 目录下，文件名为 `{model_name}_{test_type}_test_result.json`。

## 4. 第三步：结果评估

使用 `code/eval_fixed.py` 对生成的 JSON 结果文件进行指标计算（EM, Precision, Recall, F1）。

### 运行指令

```
python code/eval_fixed.py \
    --result_file output/qwen_normal_test_result.json
```

### 参数说明

| 参数            | 说明                             |
| --------------- | -------------------------------- |
| `--result_file` | 推理步骤生成的 JSON 结果文件路径 |

### 输出解读

控制台将输出以下分类的指标：

1. **ALL DATA**: 整体性能。
2. **normal_single**: 单指令正常场景。
3. **unexist_single**: 幻觉测试（不存在的设备/属性），考察模型是否能正确输出 `error_input`。
4. **normal_multi**: 多指令正常场景。
5. **mix_multi**: 混合指令场景。

同时，会在 `output/` 目录下生成详细的错误分析文件（如 `normal_multi_errors.json`），记录了具体的预测错误案例供分析。