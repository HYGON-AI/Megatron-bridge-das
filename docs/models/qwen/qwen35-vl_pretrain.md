<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->

# Qwen3.5-VL 系列

Qwen3.5-VL 多模态预训练（pretrain）性能测试，覆盖 dense 和 MoE 两类模型，
支持 BW1000 / BW1100 两个平台。每个尺寸都有两个变体：

- **Energon recipe**：读真实 Energon 格式数据集
- **Mock recipe**：生成确定性的合成图文样本，用于分布式冒烟测试

所有 recipe 都不加载预训练权重，只测性能。

## Supported

对应的 recipe 名（以 BW1100 为例）：

| 模型 | Energon recipe | Mock recipe |
|---|---|---|
| 2B | `qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 4B | `qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 9B | `qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_9b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 27B | `qwen35_vl_27b_pretrain_8gpu_bw1100_bf16_config` | `qwen35_vl_27b_pretrain_8gpu_bw1100_bf16_mock_config` |
| 35B-A3B | `qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_config` | `qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_mock_config` |

把名字里的 `bw1100` 换成 `bw1000` 就是 BW1000 平台的 recipe

## 1、环境准备

### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist)拉取 HCU 基础 docker 镜像

### 1.2 创建容器

参考命令如下（请按需修改，正确挂载工作目录，修改正确镜像名称）：

```shell
docker run -it \
       --name megatron-bridge_test \
       --shm-size=64G \
       --device=/dev/kfd \
       --device=/dev/mkfd \
       --device=/dev/dri \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --ulimit memlock=-1:-1 \
       --ipc=host \
       --network=host \
       --workdir=/workspace \
       --privileged \
       -v /opt/hyhal:/opt/hyhal:ro \
       -v /home:/home \
       REPOSITORY:TAG \
       /bin/bash
```

### 1.3 拉取代码仓

```shell
cd /workspace

git clone --recurse-submodules https://github.com/HYGON-AI/Megatron-bridge-das.git
```

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r requirements.txt
```

## 2、数据处理

### 2.1 Mock 数据

mock recipe 自己生成合成图文样本，无需任何数据集准备，设置
`MOCK_NUM_BASE_EXAMPLES` 控制样本数即可。

### 2.2 Energon 数据

Energon recipe 需要一个已经准备好的 Energon 数据集，路径通过 `DATA_PATH`
（或按尺寸的 `QWEN35_<SIZE>_DATA_PATH`，如 `QWEN35_9B_DATA_PATH`）指定。
数据集的 task encoder、`pad_to_max_length` 等由 recipe 内部固定：

- 数据类型：`EnergonDatasetConfig`，`dataloader_type="external"`
- task encoder：`QwenVLEnergonTaskEncoderConfig`（processor 从
  `LOCAL_HF_PATH` 加载，与权重保持同一目录）
- `do_validation` / `do_test` 关闭，`enable_in_batch_packing` 关闭

下面是把单条样本含多张图片的数据集转成 Energon 的例子,用的是 [TIGER-Lab/Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) 的一个子集。

1. 从 modelscope 下载 TIGER-Lab/Mantis-Instruct 的 `llava_665k_multi` 子集并解压图片目录(注意:需要 44GB 磁盘空间):

    ```
    modelscope download --dataset TIGER-Lab/Mantis-Instruct \
      --include "llava_665k_multi/*" \
      --repo-type dataset  \
      --local_dir ./Mantis-Instruct  
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

## 3、模型准备

```shell
pip install modelscope
cd /workspace

# 以 2B 为例
modelscope download --model Qwen/Qwen3.5-2B --local_dir ./Qwen3.5-2B
```

或者[点击链接](https://modelscope.cn/models)搜索 `Qwen/Qwen3.5-<尺寸>`，
手动下载模型文件后上传至 `/workspace` 路径下。

VL 的 processor 和 tokenizer 也从 `LOCAL_HF_PATH` 读取，所以必须是完整的
HuggingFace 目录（含 `config.json`、`tokenizer.json`、`preprocessor_config.json`），
不能只放权重。

## 4、启动任务

### 4.1 参数填写

（1）配置 hostfile 文件

进入 `/workspace/Megatron-bridge-das/examples/qwen`，填写 `hostfile`，示例如下
（多个节点填写多行）：

```shell
127.0.0.1
127.0.0.2
```

可以填写节点 ip 或者节点主机名。单机运行时不需要 hostfile。

（2）设置环境变量

```shell
PORT=""   # mpirun 跨节点 ssh 用的端口，需与容器内 sshd 端口一致（默认 25951）
export LOCAL_HF_PATH=/path/to/Qwen3.5-VL-2B
```

### 4.2 pretrain

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

export LOCAL_HF_PATH=/workspace/Qwen3.5-2B
export PORT=25951

./train_qwen35_vl_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES] [KEY=VALUE ...]
```

- `RECIPE`：脚本里 `SUPPORTED_RECIPES` 列出的名字，见上面的表格
- `MODE`：`dryrun` 或 `train`
- `HOSTFILE`：hostfile 路径，单机不需要
- `NODES`：使用的节点数，从 hostfile 里取前 N 行
- `KEY=VALUE`：余下参数会最后转发给 runner，可覆盖脚本里设置的项

launcher 会从 recipe 名字里读回尺寸和数据后端，所以拓扑、序列长度和数据集
接线始终与 recipe 一致。

单机 mock 冒烟测试：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-2B
MOCK_NUM_BASE_EXAMPLES=100 ./train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config train
```

单机 Energon 训练：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-4B
export DATA_PATH=/data/energon/qwen35-4b
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_4b_pretrain_8gpu_bw1100_bf16_config train
```

多机训练（35B-A3B，2 节点）：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-35B-A3B
export PORT=25951
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_35b_a3b_pretrain_16gpu_bw1100_bf16_mock_config train ./hostfile 2
```

### 4.3 dryrun

只解析并校验配置，不占卡、不需要数据集：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-2B
./train_qwen35_vl_pretrain.sh \
  qwen35_vl_2b_pretrain_8gpu_bw1100_bf16_mock_config dryrun
```

解析出的完整配置会写到
`examples/qwen/<RECIPE>_tp<>_pp<>_cp<>/<RECIPE>.yaml`。
