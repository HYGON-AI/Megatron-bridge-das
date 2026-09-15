<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
-->

# Qwen 3 VL - 视觉语言模型

> 本文是 [README.md](README.md) 的中文版。英文原文为准,若两者出现分歧请以英文版为准。

本目录包含 Qwen 3 视觉语言模型的示例脚本。

模型介绍与架构细节见 [Qwen 3 - VL 文档](../../../../docs/models/qwen/qwen3-vl.md)。

## 工作目录配置

所有脚本都用 `WORKSPACE` 环境变量指定检查点和结果的根目录,默认为 `/workspace`。可以覆盖:

```bash
export WORKSPACE=/your/custom/path
```

目录结构:
- `${WORKSPACE}/models/` — 转换后的检查点
- `${WORKSPACE}/results/` — 训练输出与实验结果

## 检查点转换

### 导入 HF → Megatron
把 HF 的 VL 模型导入到指定的 Megatron 路径:
```bash
./scripts/conversion/convert.sh import \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct
```

### 导出 Megatron → HF
```bash
./scripts/conversion/convert.sh export \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --megatron-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct-hf-export
```

## 推理

### 在转换后的检查点上运行推理

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
  --hf_model_path Qwen/Qwen3-VL-8B-Instruct \
  --megatron_model_path ${WORKSPACE}/models/Qwen3-VL-8B-Instruct/iter_0000000 \
  --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
  --prompt "Describe this image." \
  --max_new_tokens 100 
```

注意:
- `--megatron_model_path` 是可选的。不指定时,脚本会先转换模型再执行前向。
- `--image_path` 也接受图片 URL:`--image_path="https://example.com/image.jpg"`

[inference.sh](inference.sh) 脚本里有以下几种用法的命令:
- 用 Hugging Face 检查点推理
- 用导入的 Megatron 检查点推理
- 用导出的 Hugging Face 检查点推理

**预期输出:**
```
...
Generation step 46
Generation step 47
Generation step 48
Generation step 49
======== GENERATED TEXT OUTPUT ========
Image: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png
Prompt: Describe this image.
Generated: <|im_start|>user
<|vision_start|><|image_pad|><|image_pad|>
...
<|image_pad|><|vision_end|>Describe this image.<|im_end|>
<|im_start|>assistant
This image displays a **technical specifications table** comparing two variants of NVIDIA's H100 GPU: the **H100 SXM** and the **H100 NVL**.

The table is organized into rows, each detailing a specific performance or hardware characteristic, with columns showing the corresponding value for each GPU variant.

Here is a breakdown of the key specifications:

**Performance (FLOPS & TOPS):**
*   **FP64 (Double Precision):** The
=======================================
```

## 微调配方

| 模型 | 全参数 SFT | PEFT |
| --- | --- | --- |
| 8B | `qwen3_vl_8b_sft_config`(2 卡) | `qwen3_vl_8b_peft_config`(1 卡) |
| 30B-A3B | `qwen3_vl_30b_a3b_sft_config`(8 卡) | `qwen3_vl_30b_a3b_peft_config`(4 卡) |
| 235B-A22B | `qwen3_vl_235b_a22b_sft_config`(32 卡) | `qwen3_vl_235b_a22b_peft_config`(16 卡) |

训练前确保设置了以下环境变量:
1. `HF_TOKEN`:从 HF Hub 下载模型时需要
2. `HF_HOME`:(可选)避免重复下载模型和数据集
3. `WANDB_API_KEY`:(可选)启用 WandB 日志

### 预训练

仓库自带的预训练配方使用合成 VLM 数据做环境验证:`qwen3_vl_8b_pretrain_mock_config`、`qwen3_vl_30b_a3b_pretrain_mock_config`、`qwen3_vl_235b_a22b_pretrain_mock_config`。

### 有监督微调(SFT)

[sft_unpacked.sh](sft_unpacked.sh) 是全参数微调、序列不打包、模型并行可配置的脚本。
[sft.sh](sft.sh) 是全参数微调 + 序列打包的脚本。

### 参数高效微调(PEFT / LoRA)

[peft_unpacked.sh](peft_unpacked.sh) 是 LoRA 微调、序列不打包、张量并行与流水并行可配置的脚本。
[peft.sh](peft.sh) 是 LoRA 微调 + 序列打包的脚本。

**注意:** LoRA/DoRA 大幅降低显存需求,因此可以用更大的 batch size 和更少的卡。

托管或本地的 Hugging Face 数据、以及完整的单卡 Qwen3-VL 跑通流程,从 [Hugging Face 多模态教程](../../../../tutorials/data/hf-multimodal/README.md) 开始。分片的 WebDataset 数据用 [多模态 Energon 教程](../../../../tutorials/data/energon/README.md)。
需要一个版本固定、自包含的真实数据例子,参考 [Nemotron Image Training v3 Energon 教程](../../../../tutorials/data/energon/nemotron-image-v3.md)。
图文 caption 训练的完整示例见 [DataComp Energon 教程](../../../../tutorials/data/datacomp/README.md)。

## 控制 Energon 的视觉 token 计算预算

以下字段属于 `QwenVLEnergonTaskEncoderConfig`,不用于配置 Direct-HF 数据源。

三组彼此独立、都可以从命令行覆盖的开关,共同约束单条样本的 GPU 开销,可叠加使用:
- **`dataset.task_encoder.min_pixels` / `dataset.task_encoder.max_pixels`** — 图像/视频帧的分辨率上下界(默认 `200704` / `1003520`)。
- **`dataset.task_encoder.max_num_images` / `dataset.task_encoder.max_num_frames`** — 图像/帧的数量上限(默认 `10` / `60`)。图像超限会**丢弃整条样本**,帧超限只**截断多余帧**。
- **`dataset.task_encoder.max_visual_tokens`** — 所有图像和帧加起来的视觉 token 总量上限,在缩放之后按 `prod(T,H,W) // merge_size²` 计算(默认 `16384`,设为 `None` 可关闭)。它能兜住前两项漏掉的情况,超限样本会被丢弃。

## 用 Energon 数据集微调

[多模态 Energon 教程](../../../../tutorials/data/energon/README.md) 详述了 tar 成员契约、版本兼容的索引方式、标准的 `ChatMLWebdataset` YAML、单卡不打包的冒烟流程,以及基于 Energon candidate-buffer API 的原生在线序列打包。[peft_energon.sh](peft_energon.sh) 启动的是配合 `vlm_step` 的原生 MBS1 打包路径;先把基线跑通,再去调 `PACKING_BUFFER_SIZE`。

### 预期的训练表现
我们提供了一份 [Weights & Biases 报告](https://api.wandb.ai/links/nvidia-nemo-fw-public/lczz4ixx),包含预期的 loss 曲线和梯度范数。

## 多图数据集

下面是在单条样本含多张图片的数据集上微调的例子,用的是 [TIGER-Lab/Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) 的一个子集。

1. 从 Hugging Face 下载 TIGER-Lab/Mantis-Instruct 的 `llava_665k_multi` 子集并解压图片目录(注意:需要 44GB 磁盘空间):

    ```
    pip install -U "huggingface_hub[cli]"
    huggingface-cli download TIGER-Lab/Mantis-Instruct \
        --include "llava_665k_multi/*" \
        --repo-type dataset \
        --local-dir /path/to/Mantis-Instruct-LLaVA    
    ```

2. 转换并索引下载好的子集。这个脚本会写出确定性的 train/validation 分片、调用 Energon 的准备 API,并写出 `.nv-meta/dataset.yaml`:

    ```bash
    uv run python 3rdparty/Megatron-Bridge/examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
        --source-dir /path/to/Mantis-Instruct-LLaVA \
        --output-dir /path/to/Mantis-Instruct-LLaVA/wds \
        --max-samples-per-tar 10000 \
        --validation-fraction 0.01
    ```

3. 在标准的 Energon 启动命令里设置 `dataset.path=/path/to/Mantis-Instruct-LLaVA/wds`。不需要交互式选择类,也不需要手工替换 YAML。

## 从 COCO Caption CSV 生成数据集

如果手上是「一个 CSV + 一堆已下载的图片」而不是 Mantis 那种 parquet 布局,用 [prepare_coco_caption_source.py](prepare_coco_caption_source.py) 先转成 Mantis 布局,再接上面的 `prepare_mantis_energon.py`。两者是上下游关系:

```
train2014.csv + coco_images/
        │
        │  prepare_coco_caption_source.py
        ▼
<source-dir>/<subset>/train-*.parquet + images/     ← Mantis 布局
        │
        │  prepare_mantis_energon.py
        ▼
<output-dir>/{train,val}-shard-*.tar + .nv-meta/    ← Energon 数据集
        │
        ▼
  训练时 dataset.path=<output-dir>
```

### 第一步:`prepare_coco_caption_source.py`

把 ModelScope [`coco_2014_caption`](https://modelscope.cn/datasets/modelscope/coco_2014_caption) 的 CSV 导出加上本地已下载的图片,转成 Mantis 布局。

**输入**

| 项 | 要求 |
|---|---|
| CSV | 四列 `uniq_id,image_id,caption,image`,最后一列是图片 URL |
| 图片目录 | 文件名形如 `{uniq_id}_{image_id}_{URL basename}`,例如 `490489_199134_COCO_train2014_000000199134.jpg` |

**输出**:`<output-dir>/<subset-name>/` 下的 `train-{i:05d}-of-{n:05d}.parquet`,以及 `images/`(`reference` 模式除外)。

parquet 的两列 schema 是 `prepare_mantis_energon.py` 唯一认的形式:

```python
{"images":       [{"path": "images/490489_199134_COCO_train2014_000000199134.jpg"}],  # 相对 subset 目录
 "conversation": [{"role": "user",      "content": "介绍一下图片内容<image>"},          # content 必须是纯 str
                  {"role": "assistant", "content": "a couple of men standing on a stage."}]}
```

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--csv PATH` | **必填** | 可重复。**一个 CSV = 一个 subset 目录**,目录名取文件名 stem |
| `--images-dir PATH` | **必填** | 存放 `{uniq}_{imgid}_{basename}` 图片的目录 |
| `--output-dir PATH` | **必填** | 要生成的 `--source-dir`;subset 是它的子目录 |
| `--subset-name NAME` | CSV 文件 stem | 只在传入**单个** `--csv` 时可用,否则报错 |
| `--prompt TEXT` | `介绍一下图片内容<image>` | user 轮内容,**必须恰好含一个 `<image>`**,否则 `ValueError` |
| `--rows-per-parquet N` | `5000` | 每个 parquet 的行数 |
| `--image-mode` | `symlink` | `symlink` 建软链 / `copy` 真拷贝 / `reference` 不落文件,直接存相对路径 |
| `--multi-caption` | `first` | `&&` 拼接的多 caption 单元格:`first` 只取第一条,`all` 展开成多条样本(共用同一张图) |
| `--columns U I C URL` | `uniq_id image_id caption image` | 四列的列名 |
| `--no-header` | 关 | CSV 无表头时加上 |
| `--encoding` | `utf-8` | CSV 文本编码 |
| `--limit N` | 无 | 只读前 N 行,冒烟测试用 |

**行为细节**

- **用 pandas 解析,不是按逗号切。** 公开的 CSV 会给含逗号的 caption 加引号,且 caption 内可能嵌换行,手工切会静默错行。
- **图片定位有两级**:先试精确名 `{uniq}_{imgid}_{basename}`;miss 再回退到按 URL basename 建的索引,因此 `&&` 拼接的 uniq_id 也能命中。
- **图片找不到就跳过并计数**,不中断整轮。公开 CSV 的行数常多于实际下载量。
- 返回 `{"samples": …, "rows": …, "missing_image": …, "empty_caption": …}` 并打日志;一条样本都没产出时抛 `RuntimeError`。

### 第二步:`prepare_mantis_energon.py`

**输入**:`--source-dir`,下面**必须是一层 subset 子目录**,每个 subset 内含 `train-*.parquet`,可选 `train_images.zip`。

**输出**:`{train,val}-shard-%06d.tar`,每条样本两个成员——`.jpgs`(pickle 的 image bytes 列表)和 `.json`(ChatML 对话),外加 `.nv-meta/`(`dataset.yaml`、`split.yaml`、`index.sqlite`)。

**参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source-dir PATH` | **必填** | 含 subset 子目录的根目录 |
| `--output-dir PATH` | **必填** | tar 与 `.nv-meta/` 的输出位置,**这个值就是训练时的 `dataset.path`** |
| `--max-samples-per-tar N` | `1000` | 每个 tar 的样本数;真实 Mantis 用 `10000` |
| `--validation-fraction F` | `0.01` | 按 key 的 SHA256 稳定切出 val,取值须在 `[0, 1)` |
| `--num-workers N` | `8` | Energon 建索引的并行度,与 `dataset.num_workers` 无关 |
| `--skip-energon-prepare` | 关 | 只写 tar,不建索引也不写 `dataset.yaml` |

**行为细节**

- 每个 subset 先解压 `train_images.zip`,用 `.extracted_*` 标记保证只解一次,并做了 zip-slip 防护。
- subset 里没有 `train-*.parquet` 时静默跳过。
- 样本 key 为 `{subset}__{parquet_stem}__{index:06d}`,其中 `index` 是**该 parquet 文件内**的序号。切分基于 key 的 SHA256,因此完全确定、重跑一致。
- **三种静默跳过**:`images` 为空;图片打不开(`OSError` / `KeyError`);`<image>` 占位符数量 ≠ 图片数量。
- 空 split 的 tar 会被删除;全部为空则抛 `RuntimeError`。

### 完整示例

```bash
# 第一步:16000 行,约 15 秒,产出 ~884KB parquet + 16000 个软链
python examples/qwen/prepare_coco_caption_source.py \
  --csv         /path/to/coco/train2014.csv \
  --images-dir  /path/to/coco/coco_images \
  --output-dir  /path/to/coco-source \
  --subset-name coco_train2014

# 第二步:约 2 分钟,产出 ~2.5G tar
python 3rdparty/Megatron-Bridge/examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
  --source-dir /path/to/coco-source \
  --output-dir /path/to/coco-energon \
  --max-samples-per-tar 10000 \
  --validation-fraction 0.01

# 第三步:训练时覆盖 dataset.path(配方里默认是 None,必须显式传入)
... dataset.path=/path/to/coco-energon
```

实测 16000 行的 COCO train2014:第一步 16000/16000 全部命中、0 跳过;第二步切出 **15854 train + 146 val**、0 跳过。

### 常见坑

**`--source-dir` 必须多一层。** 直接指到装着 parquet 的目录会报 `No subset directories found`——转换器的 `iterdir()` 只收子目录。

**`content` 必须是纯字符串。** 校验逻辑是 `turn["content"].count("<image>")`。如果写成教程里那种 `[{"type": "image"}, {"type": "text", ...}]` 结构化列表,会直接 `AttributeError: 'list' object has no attribute 'count'`,而且**不在** try 保护范围内。缺 `conversation` 列同理,是未捕获的 `KeyError`。

**`symlink` 模式下别提前删原图。** source-dir 里是指向原图目录的软链接;等 tar 生成完之后,source-dir 才可以删。

**验证集来自 train 切分。** ModelScope 的 `val2014.csv` 每行都是 `&&` 拼接的多 caption,且其 `COCO_val2014_*` 图片通常没有随 train 一起下载,所以 val 只能靠 `--validation-fraction` 切。

**每个 split 的样本数必须 ≥ `dp_size × num_workers`**,其中 `dp_size = world_size / (TP × PP × CP)`,`num_workers=0` 按 1 算。否则部分 DataLoader worker 分不到分片,触发:

```
AssertionError: Cannot repeat empty dataset indefinitely
```

按 `--validation-fraction 0.01` 从 16000 条切出的 146 条 val,对 8 卡 × `num_workers=2`(=16)是够的。

**索引那步需要 `megatron` 模块。** `prepare_mantis_energon.py` 顶部有 `from megatron.bridge.data.energon import prepare_webdataset`。若环境没装 megatron-bridge,可先用 `--skip-energon-prepare` 产出 tar,再到有该模块的环境补索引。

## 评测

敬请期待。
