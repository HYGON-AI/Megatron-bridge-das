# 基于Megatron-Bridge-das启动Qwen3.5-text pretrain示例

## 1、环境准备

### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist)拉取HCU基础docker镜像

### 1.2 创建容器

参考命令如下（请按需修改，正确挂载工作目录，修改正确镜像名称）：

```shell
docker run -it \
       --name megatron-bridge-das_test \
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

如代码仓无法正常clone，可离线下载对应分支的`Megatron-bridge-das`，`Megatron-bridge`仓库，解压后按`Megatron-bridge-das`仓库目录位置放置。

### 1.4 安装依赖

```shell
cd /workspace/Megatron-bridge-das
pip install -r requirements.txt
```

## 2、数据处理

pretrain使用mock的数据集，无需处理

## 3、模型准备

```shell
pip install modelscope

cd /workspace

#根据需求下载模型：
#Qwen3.5-35B-A3B模型下载命令：
modelscope download --model Qwen/Qwen3.5-35B-A3B --local_dir ./Qwen3.5-35B-A3B

#Qwen3.5-397B-A17B模型下载命令：
modelscope download --model Qwen/Qwen3.5-397B-A17B --local_dir ./Qwen3.5-397B-A17B
```

或者点击链接[Qwen3.5-397B-A17B](https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B/files)，[Qwen3.5-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B/files)，手动下载模型文件，上传至`/workspace`路径下。

## 4、启动任务

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/Megatron-bridge-das/examples/qwen`，填写`hostfile`文件，示例如下（多个节点填写多行）

```shell
127.0.0.1
127.0.0.2
```

可以填写节点ip或者节点主机名。

（2）根据任务选择并填写`train_qwen35_text_pretrain.sh` 或`train_qwen35_text_397b_a17b_pretrain.sh`脚本

进入`/workspace/Megatron-bridge-das/examples/qwen`，根据上述文件路径，填写选择脚本文件中如下变量：

```shell
#Qwen3.5-397B-A17B
export LOCAL_HF_PATH=/path/to/Qwen3.5-397B-A17B

#Qwen3.5-35B-A3B
export LOCAL_HF_PATH=/path/to/Qwen3.5-35B-A3B
```

### 4.2 执行脚本

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

#以Qwen3.5-397B-A17B为例
bash train_qwen35_text_397b_a17b_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]

RECIPE: SUPPORTED_RECIPES in train_qwen35_text_397b_a17b_pretrain.sh
MODE: dryrun or train # 配置dryrun时不需要再跟HOSTFILE和NODES参数
HOSTFILE: path to hostfile
NODES: The number of nodes you use

```
