# DeepSeek V4 Flash

## Supported
| Hardware | Recipe | Precision and optimizer |
|----------|--------|-------------------------|
| BW1101, 16 GPUs | `deepseek_v4_flash_pretrain_16gpu_bw1101_bf16_test_config` | BF16 Adam |
| BW1101, 16 GPUs | `deepseek_v4_flash_pretrain_16gpu_bw1101_bf16_muon_test_config` | BF16 Muon |
| BW1101, 16 GPUs | `deepseek_v4_flash_pretrain_16gpu_bw1101_fp8sc_test_config` | FP8 Adam |
| BW1101, 16 GPUs | `deepseek_v4_flash_sft_16gpu_bw1101_bf16_test_config` | BF16 Adam |
| BW1101, 32 GPUs | `deepseek_v4_flash_peft_32gpu_bw1101_bf16_config` | BF16 Adam |
| BW1300, 16 GPUs | `deepseek_v4_flash_pretrain_16gpu_bw1300_bf16_test_config` | BF16 Adam |
| BW1300, 16 GPUs | `deepseek_v4_flash_pretrain_16gpu_bw1300_bf16_muon_test_config` | BF16 Muon |
| BW1300, 16 GPUs | `deepseek_v4_flash_pretrain_16gpu_bw1300_fp8sc_test_config` | FP8 Adam |
| BW1300, 16 GPUs | `deepseek_v4_flash_sft_16gpu_bw1300_bf16_test_config` | BF16 Adam |


## 1、环境准备

### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist)拉取HCU基础docker镜像

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

git clone --recurse-submodules -b dev https://github.com/HYGON-AI/Megatron-bridge-das.git
```

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r examples_hcu/requirements.txt
```

## 2、数据处理

### 2.1 pretrain

#### 数据集下载
```python
cd /workspace
hf download NeelNanda/pile-10k --repo-type=dataset --local-dir pile-10k
hf download openai-community/gpt2 config.json tokenizer.json tokenizer_config.json --local-dir ./gpt2
```
或
[点击下载](https://huggingface.co/datasets/NeelNanda/pile-10k/tree/main)数据集。

假设下载后文件目录如下
```
├── data
│   └── train-00000-of-00001-4746b8785c874cc7.parquet
├── dataset_infos.json
└── README.md
```

#### 数据集处理
```python
cd /workspace/Megatron-bridge-das
python3 examples_hcu/data_preparation/prepare_pile10k.py --stage all --parquet /workspace/pile-10k/data/train-00000-of-00001-4746b8785c874cc7.parquet --output-dir /workspace/the-pile --tokenizer-type HuggingFaceTokenizer --tokenizer-model /workspace/gpt2 --workers 32
```

### 2.2 sft、lora

#### 数据集下载
```python
modelscope download --dataset AI-ModelScope/OpenMathInstruct-2 --local_dir ./OpenMathInstruct-2
```
或
[点击下载](https://www.modelscope.cn/datasets/AI-ModelScope/OpenMathInstruct-2/files)数据集。

假设下载后文件目录如下
```
OpenMathInstruct-2
├── data
├── dataset_infos.json
├── README.md
├── scaling_plot.jpg
└── SFT Data Diagram 1.jpg
```

## 3、模型准备

```shell
pip install modelscope

cd /workspace

modelscope download --model deepseek-ai/DeepSeek-V4-Flash --local_dir ./DeepSeek-V4-Flash
```

或者[点击链接](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/files)，手动下载模型文件，上传至`/workspace`路径下。

## 4、启动任务

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/Megatron-bridge-das/examples_hcu/deepseekv4_flash`，填写`hostfile`文件，示例如下（多个节点填写多行）

```shell
127.0.0.1
127.0.0.2
```

可以填写节点ip或者节点主机名。

（2）根据任务填写脚本变量并执行

#### pretrain
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/deepseekv4_flash

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/the-pile}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/DeepSeek-V4-Flash}

./train_deepseekv4_flash_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_deepseekv4_flash_pretrain.sh
MODE: dryrun or pretrain # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```

#### sft
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/deepseekv4_flash

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/OpenMathInstruct-2}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/DeepSeek-V4-Flash}
export LOCAL_CORE_PATH=${LOCAL_CORE_PATH:-/path/to/DeepSeek-V4-Flash-core}

./train_deepseekv4_flash_sft.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_deepseekv4_flash_sft.sh
MODE: dryrun or sft # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```

#### lora
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/deepseekv4_flash

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/OpenMathInstruct-2}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/DeepSeek-V4-Flash}
export LOCAL_CORE_PATH=${LOCAL_CORE_PATH:-/path/to/DeepSeek-V4-Flash-core}

./train_deepseekv4_flash_lora.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_deepseekv4_flash_lora.sh
MODE: dryrun or lora # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```