# GLM-5.3-Flash

## Supported
| Hardware | Recipe | Precision and optimizer |
|----------|--------|-------------------------|
| BW1101, 16 GPUs | `glm53_flash_pretrain_16gpu_bw1101_bf16_test_config` | BF16 Adam |
| BW1300, 16 GPUs | `glm53_flash_pretrain_16gpu_bw1300_bf16_test_config` | BF16 Adam |
| BW1101, 16 GPUs | `glm53_flash_sft_16gpu_bw1101_bf16_test_config` | BF16 Adam |
| BW1300, 16 GPUs | `glm53_flash_sft_16gpu_bw1300_bf16_test_config` | BF16 Adam |
| BW1101, 32 GPUs | `glm53_flash_peft_32gpu_bw1101_bf16_config` | BF16 Adam |
| BW1300, 32 GPUs | `glm53_flash_peft_32gpu_bw1300_bf16_config` | BF16 Adam |


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
modelscope download --dataset Salesforce/wikitext --local_dir ./wikitext
```
或
[点击下载](https://www.modelscope.cn/datasets/Salesforce/wikitext/files)数据集。

假设下载后文件目录如下

```
wikitext
├── dataset_infos.json
├── README.md
├── wikitext-103-raw-v1
│   ├── test-00000-of-00001.parquet
│   ├── train-00000-of-00002.parquet
│   ├── train-00001-of-00002.parquet
│   └── validation-00000-of-00001.parquet
├── wikitext-103-v1
│   ├── test-00000-of-00001.parquet
│   ├── train-00000-of-00002.parquet
│   ├── train-00001-of-00002.parquet
│   └── validation-00000-of-00001.parquet
├── wikitext-2-raw-v1
│   ├── test-00000-of-00001.parquet
│   ├── train-00000-of-00001.parquet
│   └── validation-00000-of-00001.parquet
└── wikitext-2-v1
    ├── test-00000-of-00001.parquet
    ├── train-00000-of-00001.parquet
    └── validation-00000-of-00001.parquet
```

#### 数据集处理
```python
cd /workspace/Megatron-bridge-das
python3 examples_hcu/data_preparation/prepare_wikitext103.py --parquet-dir /workspace/wikitext/wikitext-103-raw-v1/ --output-dir  /workspace/wikitext103 --tokenizer-model /workspace/GLM-5.3-Flash --workers 32
```

### 2.2 sft、lora

#### 数据集下载
```python
hf download allenai/tulu-3-sft-mixture --repo-type=dataset --local-dir=/workspace/tulu-3-sft-mixture
```
或
[点击下载](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/tree/main)数据集。

假设下载后文件目录如下

```
workspace
├── data
│   ├── train-00000-of-00006.parquet
│   ├── train-00001-of-00006.parquet
│   ├── train-00002-of-00006.parquet
│   ├── train-00003-of-00006.parquet
│   ├── train-00004-of-00006.parquet
│   ├── train-00005-of-00006.parquet
└── README.md

```

## 3、模型准备

```shell
pip install modelscope

cd /workspace

modelscope download --model ZhipuAI/GLM-5.3-Flash --local_dir ./GLM-5.3-Flash
```

或者[点击链接](https://modelscope.cn/models/ZhipuAI/GLM-5.3-Flash/files)，手动下载模型文件，上传至`/workspace`路径下。

## 4、启动任务

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/Megatron-bridge-das/examples_hcu/glm53_flash`，填写`hostfile`文件，示例如下（多个节点填写多行）

```shell
127.0.0.1
127.0.0.2
```

可以填写节点ip或者节点主机名。

（2）根据任务填写脚本变量并执行

#### pretrain
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/glm53_flash

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/wikitext103}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/GLM-5.3-Flash}

./train_glm53_flash_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_glm53_flash_pretrain.sh
MODE: dryrun or pretrain # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```

#### sft
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/glm53_flash

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/tulu-3-sft-mixture}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/GLM-5.3-Flash}

./train_glm53_flash_sft.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_glm53_flash_sft.sh
MODE: dryrun or sft # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```

#### lora
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples_hcu/glm53_flash

PORT=${PORT:-12345} # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=${DATA_PATH:-/path/to/tulu-3-sft-mixture}
export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/GLM-5.3-Flash}

./train_glm53_flash_lora.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_glm53_flash_lora.sh
MODE: dryrun or lora # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```