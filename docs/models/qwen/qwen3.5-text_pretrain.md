# Qwen3.5 Text 系列

Qwen3.5 纯文本预训练（pretrain）性能测试，覆盖 dense 和 MoE 两类模型，
支持 BW1000 / BW1100 两个平台。所有 recipe 走 mock（随机初始化 + 合成数据）
只测性能，不加载预训练权重。

## Supported

BW1100 为 144G 显存卡；`节点数` 是 recipe 名字里标称的部署规模。

| 模型 | 平台 | Recipe | 节点 | 卡数 | TP/PP/EP |
|---|---|---|---:|---:|---|
| 0.8B | BW1000 | `qwen35_text_0_8b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 1/1/1 |
| 0.8B | BW1100 | `qwen35_text_0_8b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 1/1/1 |
| 2B | BW1000 | `qwen35_text_2b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 1/1/1 |
| 2B | BW1100 | `qwen35_text_2b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 1/1/1 |
| 4B | BW1000 | `qwen35_text_4b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 2/1/1 |
| 4B | BW1100 | `qwen35_text_4b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 2/1/1 |
| 9B | BW1000 | `qwen35_text_9b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 2/1/1 |
| 9B | BW1100 | `qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 2/1/1 |
| 27B | BW1000 | `qwen35_text_27b_pretrain_8gpu_bw1000_bf16_config` | 1 | 8 | 4/2/1 |
| 27B | BW1100 | `qwen35_text_27b_pretrain_8gpu_bw1100_bf16_config` | 1 | 8 | 2/2/1 |
| 35B-A3B (MoE) | BW1000 | `qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config` | 4 | 32 | 1/4/8 |
| 35B-A3B (MoE) | BW1100 | `qwen35_text_35b_a3b_pretrain_16gpu_bw1100_bf16_config` | 2 | 16 | 1/2/8 |


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

pretrain 使用 mock 的数据集，无需处理。

## 3、模型准备

每个尺寸单独下载到 `/workspace` 下，`LOCAL_HF_PATH` 指向对应目录即可：

```shell
pip install modelscope
cd /workspace

# 按需选择尺寸，下面以 9B 为例
modelscope download --model Qwen/Qwen3.5-9B --local_dir ./Qwen3.5-9B
```

或者[点击链接](https://modelscope.cn/models)搜索 `Qwen/Qwen3.5-<尺寸>`，
手动下载模型文件后上传至 `/workspace` 路径下。

注意目录名大小写：35B-A3B 的官方目录名是 `Qwen3.5-35B-A3B`，其余尺寸为
小写 `qwen3.5-<尺寸>`（如 `qwen3.5-9b`）。

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
PORT=""   # mpirun 跨节点 ssh 用的端口，需与容器内 sshd 端口一致
          # 在各节点的容器内执行 /usr/sbin/sshd -p <port> 启动，例如 -p 25951
export LOCAL_HF_PATH=/path/to/Qwen/Qwen3.5-9B
```

### 4.2 pretrain

```shell
docker exec -it "your container name" bash
cd /workspace/Megatron-bridge-das/examples/qwen

export LOCAL_HF_PATH=/workspace/Qwen/Qwen3.5-9B
export PORT=25951

./train_qwen35_text_pretrain.sh <RECIPE> <MODE> [HOSTFILE] [NODES]
```

- `RECIPE`：脚本里 `SUPPORTED_RECIPES` 列出的名字，见上面的表格
- `MODE`：`dryrun` 或 `train`
- `HOSTFILE`：hostfile 路径，单机不需要
- `NODES`：使用的节点数，从 hostfile 里取前 N 行

单机 8 卡示例：

```shell
export LOCAL_HF_PATH=/workspace/Qwen/Qwen3.5-9B
./train_qwen35_text_pretrain.sh \
  qwen35_text_9b_pretrain_8gpu_bw1100_bf16_config train
```

多机示例（2 节点跑 35B-A3B）：

```shell
export LOCAL_HF_PATH=/workspace/Qwen3.5-35B-A3B
export PORT=25951
./train_qwen35_text_pretrain.sh \
  qwen35_text_35b_a3b_pretrain_16gpu_bw1100_bf16_config train ./hostfile 2
```
