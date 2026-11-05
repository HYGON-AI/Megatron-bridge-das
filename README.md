<h1 align="center">
  <strong>
  Megatron-Bridge-das is a patched training library based on 
  <a href="https://github.com/NVIDIA-NeMo/Megatron-Bridge" target="_blank"><strong>Megatron-Bridge</strong></a> .
  <strong>
  <br>
</h1>

## 简介
Megatron-Bridge-das：基于HCU生态的强化学习训练套件，旨在为HCU生态合作伙伴提供强化学习训练解决方案。

## 上游归属与许可证

本仓库是 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 的二次开发仓库，以上游固定版本为基线，通过 Git submodule 引入上游源码，并在适配层维护 HYGON 侧修改。

| 项目 | 内容 |
| --- | --- |
| 上游项目 | Megatron-Bridge |
| 上游仓库 | https://github.com/NVIDIA-NeMo/Megatron-Bridge |
| 上游分支 | v0.6.0 |
| 上游 Tag | v0.6.0 |
| 上游 Commit | 51885cf132b2814188b6855c25a8588254274c2a |
| 上游许可证 | Apache-2.0 |
| 上游子模块路径 | 3rdparty/Megatron-Bridge |

上游代码版权归 NVIDIA CORPORATION & AFFILIATES 所有，按 Apache License 2.0 授权；许可证全文见 [3rdparty/Megatron-Bridge/LICENSE](3rdparty/Megatron-Bridge/LICENSE)。

本仓库对上游代码的修改由 HYGON 完成：

Modified by Hygon Information Technology Co., Ltd.

## 第三方源码

本仓库除上游 Megatron-Bridge 外，适配层还包含源自 NVIDIA Megatron-LM 等第三方项目的源码文件，以及随上游一并引入的第三方组件。各组件的项目、仓库 URL、固定 Commit/版本、Copyright、许可证、本地路径和 HYGON 修改情况，统一登记在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Third-Party components and their source attribution are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 支持模型
|Model                 | Model type            | `pretrain` | `sft` |Quick Start|
|:-------------------: | :-------------------: | :--------: | :---: | :-------: |
|DeepSeek              | DeepSeek-V4-Flash-0731| ✓          | ✓     |[pretrain](docs/models/deepseek/deepseek-v4-pretrain.md)/[sft](docs/models/deepseek/deepseek-v4-sft.md)|
|Gemma3                | Gemma3                | ✓          |       |[pretrain](docs/models/gemma/gemma3-pretrain.md)|
|Gemma4                | Gemma4                | ✓          |       |[pretrain](docs/models/gemma/gemma4-pretrain.md)|
|GLM                   | GLM-5.2               | ✓          | ✓     |[pretrain](docs/models/glm/glm5-2-pretrain.md)/[sft](docs/models/glm/glm5-2-sft.md)|
|GLM                   | GLM-5.3               | ✓          |       |[pretrain](docs/models/glm/glm5-3.md)|
|MiniMax               | MiniMax-M3            | ✓          |       |[pretrain](docs/models/minimax/minimax-m3.md)|
|Qwen                  | Qwen3.5-2B            | ✓          |       |[pretrain](docs/models/qwen/qwen35-vl-bw1000.md)|
|Qwen                  | Qwen3.5-4B            | ✓          |       |[pretrain](docs/models/qwen/qwen35-vl-bw1000.md)|
|Qwen                  | Qwen3.5-9B            | ✓          |       |[pretrain](docs/models/qwen/qwen35-vl-bw1000.md)|
|Qwen                  | Qwen3.5-35B-A3B       | ✓          |       |[pretrain](docs/models/qwen/qwen3.5-35b-a3b-pretrain.md)|
|Qwen                  | Qwen3.8-2.4T-A95B     | ✓          |       |[pretrain](docs/models/qwen/qwen3.8-pretrain.md)|
