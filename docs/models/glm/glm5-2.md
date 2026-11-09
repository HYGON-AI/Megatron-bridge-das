# GLM-5.2

## Supported
| Hardware | Recipe | Precision and optimizer |
|----------|--------|-------------------------|
| BW1300, 32 GPUs | `glm52_pretrain_32gpu_bw1000_bf16_test_config` | BF16 Adam |
| BW1300, 32 GPUs | `glm52_sft_32gpu_bw1000_bf16_test_config` | BF16 Adam |


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

git clone --recurse-submodules https://github.com/HYGON-AI/Megatron-bridge-das.git
```

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r requirements.txt
```

## 2、数据处理(pretrain不需要)

### 2.1 数据集下载

以tulu-3-sft-mixture数据集为例
```python
hf download allenai/tulu-3-sft-mixture --repo-type=dataset --local-dir=/workspace/tulu-3-sft-mixture
```
或
[点击下载](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/tree/main)数据集。

假设下载后文件目录如下

```
workspace
├── data
│   ├── train-00000-of-00006.parquet
│   ├── train-00001-of-00006.parquet
│   ├── train-00002-of-00006.parquet
│   ├── train-00003-of-00006.parquet
│   ├── train-00004-of-00006.parquet
│   ├── train-00005-of-00006.parquet
└── README.md

```

### 2.2 数据集处理

执行如下命令

```shell
cd /workspace/Megatron-bridge-das/data_preparation
MEGATRON_PATH=/workspace/Megatron-bridge-das
export PYTHONPATH=${MEGATRON_PATH}/hcu_megatron_bridge/src:${MEGATRON_PATH}/3rdparty/Megatron-Bridge/src:${MEGATRON_PATH}/3rdparty/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}
python convert_tulu3_sft_mixture_to_jsonl.py --src /workspace/tulu-3-sft-mixture/data --dst /workspace/tulu-3-sft-mixture/data
```

## 3、模型准备

```shell
pip install modelscope

cd /workspace

modelscope download --model ZhipuAI/GLM-5.2 --local_dir ./GLM-5.2
```

或者[点击链接](https://modelscope.cn/models/ZhipuAI/GLM-5.2/files)，手动下载模型文件，上传至`/workspace`路径下。

## 4、启动任务

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/Megatron-bridge-das/examples/glm`，填写`hostfile`文件，示例如下（多个节点填写多行）

```shell
127.0.0.1
127.0.0.2
```

可以填写节点ip或者节点主机名。

（2）根据任务填写脚本变量并执行

#### pretrain
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/glm

# train_glm52_pretrain.sh
PORT="" # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
export LOCAL_HF_PATH=/path/to/GLM-5.2

./train_glm52_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_glm52_pretrain.sh
MODE: dryrun or train # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```

#### sft
```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/glm

# train_glm52_sft.sh
PORT="" # Set the SSH port by running /usr/sbin/sshd -p <port> inside the Docker container on each node. For example: /usr/sbin/sshd -p 12345
DATA_PATH=/path/to/tulu-3-sft-mixture/data
export LOCAL_HF_PATH=/path/to/GLM-5.2

./train_glm52_sft.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_glm52_sft.sh
MODE: dryrun or sft # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```