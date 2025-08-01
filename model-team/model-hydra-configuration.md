# Hydra 配置系统:competition/first\_run/config.yaml 生成过程详解

## 概述

Hydra 配置系统是:

1.  先根据 `src/state/configs/config.yaml` 中的 defaults 部分生成初版配置
2.  然后根据命令行参数修改相应的设置
3.  最终生成 `competition/first_run/config.yaml`

## 训练命令

```bash
uv run state tx train \
data.kwargs.toml_config_path="competition_support_set/starter.toml" \
data.kwargs.num_workers=4\
data.kwargs.batch_col="batch_var" \
data.kwargs.pert_col="target_gene" \
data.kwargs.cell_type_key="cell_type" \
data.kwargs.control_pert="non-targeting" \
data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt" \
training.max_steps $=40000$ \
training.ckpt_every_n_steps $=2000$ \
model=state_sm \
wandb.tags=" [first_run]" \
output_dir="competition"
name="first_run"
```

## \- 配置合并过程(按Hydra 处理顺序)

### 步骤1:主配置文件定义默认组合

**文件**: `src/state/configs/config.yaml`

```yaml
defaults:
  - data: perturbation
  - model: pertsets
  - training: default
  - wandb: default
  - _self_

# 主配置参数
name: debug               # 会被命令行 name="first_run" 覆盖
output_dir: ./debugging   # 会被命令行 output_dir="competition" 覆盖
use_wandb: true
overwrite: false
return_adatas: false
pred_adata_path: null
true_adata_path: null
```

### 步骤2:命令行修改配置组

命令行参数 `model=state_sm` 将默认的 `model: pertsets` 改为 `model: state_sm`

**最终的 defaults 顺序**:

1.  `data: perturbation`
2.  `model: state_sm` 被命令行修改
3.  `training: default`
4.  `wandb: default'`
5.  `_self_`
6.  命令行覆盖参数

## 各配置文件详细内容及贡献

### 数据配置:`src/state/configs/data/perturbation.yaml`

```yaml
name: PerturbationDataModule
kwargs:
  toml_config_path: null              # 被命令行覆盖为 "competition_support_set/starter.toml"
  embed_key: null
  output_space: all
  pert_rep: onehot
  basal_rep: sample
  num_workers: 4
  pin_memory: true
  n_basal_samples: 1
  l_mapping_strategy: random
  should_yield_control_cells: true
  batch_col: gem_group                # 被命令行覆盖为"batch_var"
  pert_col: gene                      # 被命令行覆盖为 "target_gene"
  cell_type_key: cell_type            # 命令行确认了这个值
  control_pert: DMSO_TF               # 被命令行覆盖为"non-targeting"
  map_controls: true
  perturbation_features_file: null    # 被命令行覆盖为"competition_support_set/ESM2_pert_features.pt"
  store_raw_basal: false
  int_counts: false
  barcode: true
  output_dir: null
  debug: true
```

### 模型配置对比:默认 vs 命令行选择

#### 默认模型配置:`src/state/configs/model/pertsets.yaml`

```yaml
name: PertSets
checkpoint: null
device: cuda
kwargs:
  cell_set_len: 512                     # 会被 state_sm.yaml 改为 128
  extra_tokens: 1
  decoder_hidden_dims: [1024, 1024, 512]
  blur: 0.05
  hidden_dim: 328                       # 会被 state_sm.yaml 改为 672
  loss: energy
  confidence_token: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  freeze_pert_backbone: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  decoder_loss_weight: 1.0
  use_basal_projection: False           # STATE 关键调整1(两个配置都是 false)
  mask_attn: False
  distributional_loss: energy
  regularization: 0.0
  init_from: null
  transformer_backbone_key: GPT2        # 会被 state_sm.yaml 改为 llama
  transformer_backbone_kwargs:
    max_position_embeddings: ${model.kwargs.cell_set_len}       # llama 用
    n_positions: ${model.kwargs.cell_set_len}                   # gpt2 用
    hidden_size: ${model.kwargs.hidden_dim}                     # llama 用
    n_embd: ${model.kwargs.hidden_dim}                          # gpt2 用
    n_layer: 8
    n_head: 8
    resid_pdrop: 0.0
    embd_pdrop: 0.0
    attn_pdrop: 0.0
    use_cache: false
```

#### 命令行选择:`src/state/configs/model/state_sm.yaml`

```yaml
name: PertSets
checkpoint: null
device: cuda
kwargs:
  cell_set_len: 128                   # 比pertsets.yaml 的512 更小
  blur: 0.05
  hidden_dim: 672                     # 比pertsets.yaml 的328 更大
  loss: energy
  confidence_head: False              # 名称略有不同(confidence_token vs confidence_head)
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True              # STATE 关键调整2
  softplus: True                      # pertsets.yaml 中没有此参数
  freeze_pert: False                  # 名称略有不同(freeze_pert_backbone vs freeze_pert)
  transformer_decoder: False          # pertsets.yaml 中没有此参数
  finetune_vci decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False      # pertsets.yaml 中没有此参数
  use_basal_projection: False         # STATE 关键调整1
  distributional_loss: energy
  gene_decoder_bool: False            # pertsets.yaml 中没有此参数
  init_from: null
  transformer_backbone_key: llama     # 使用 LLAMA 而非 GPT2
  transformer_backbone_kwargs:        # 完全不同的LLaMA 配置
    max_position_embeddings: ${model.kwargs.cell_set_len}
    hidden_size: ${model.kwargs.hidden_dim}
    intermediate_size: 2688
    num_hidden_layers: 4
    num_attention_heads: 8
    num_key_value_heads: 8
    head_dim: 84
    use_cache: false
    attention_dropout: 0.0
    hidden_dropout: 0.0
    layer_norm_eps: 1e-6
    pad_token_id: 0
    bos_token_id: 1
    eos_token_id: 2
    tie_word_embeddings: false
    rotary_dim: 0
    use_rotary_embeddings: false
```



#### 关键差异对比:

| 参数                          | pertsets.yaml | state\_sm.yaml | 影响                                     |
| :---------------------------- | :------------ | :------------- | :--------------------------------------- |
| `cell_set_len`                | 512           | 128            | 更小的细胞集合大小                       |
| `hidden_dim`                  | 328           | 672            | 更大的隐藏维度                           |
| `transformer_backbone_key`    | GPT2          | llama          | 不同的transformer 架构                   |
| `use_basal_projection`        | ❌             | ❌              | STATE 调整1:两者都禁用 basal projection  |
| `predict_residual`            | ✅             | ✅              | STATE 调整2:两者都启用残差预测           |
| `transformer_backbone_kwargs` | GPT2 参数     | LLAMA 参数     | 完全不同的 transformer 配置              |



### 训练配置:`src/state/configs/training/default.yaml`

```yaml
wandb_track: false
weight_decay: 0.0005
batch_size: 16
lr: 1e-4                 # 命令行确认了这个值
max_steps: 40000         # 命令行确认了这个值
train_seed: 42
val_freq: 2000
ckpt_every_n_steps: 2000 # 命令行确认了这个值
gradient_clip_val: 10
loss fn: mse
devices: 1
strategy: auto
```



### Wandb配置:`src/state/configs/wandb/default.yaml`

```yaml
entity: your_entity_name
project: state
local_wandb_dir: ./wandb_logs
tags: []                  # 被命令行覆盖为 ["first_run"]
```



## 配置合并规则

### 深度合并(Deep Merge)

Hydra 进行**深度合井**,这意味着:

  * 嵌套字典会递归合并
  * 后面的配置覆盖前面的配置
  * 列表会被完全替换(不是追加)

### 优先级顺序(从低到高)

1.  `data: perturbation`
2.  `model: state_sm`
3.  `training: default`
4.  `wandb: default`
5.  `_self_`(主配置文件)
6.  **命令行参数**(最高优先级)

## 关键参数变化追踪

| 参数路径                                | 来源文件               | 原始值        | 最终值                                       | 命令行覆盖 |
| :-------------------------------------- | :--------------------- | :------------ | :------------------------------------------- | :--------- |
| `data.kwargs.toml_config_path`          | `perturbation.yaml`    | `null`        | "competition\_support\_set/starter.toml"     | ✅         |
| `data.kwargs.num_workers`               | `perturbation.yaml`    | `12`          | `4`                                          | ✅         |
| `data.kwargs.batch_col`                 | `perturbation.yaml`    | "gem\_group"  | "batch\_var"                                 | ✅         |
| `data.kwargs.pert_col`                  | `perturbation.yaml`    | "gene"        | "target\_gene"                               | ✅         |
| `data.kwargs.control_pert`              | `perturbation.yaml`    | "DMSO\_TF"    | "non-targeting"                              | ✅         |
| `data.kwargs.perturbation_features_file`| `perturbation.yaml`    | `null`        | "competition\_support\_set/ESM2\_pert\_features.pt" | ✅         |
| `model`                                 | `config.yaml defaults` | `pertsets`    | `state_sm`                                   | ✅         |
| `model.kwargs.use_basal_projection`     | `state_sm.yaml`        |               | `false`                                      |             |
| `model.kwargs.predict_residual`         | `state_sm.yaml`        |               | `true`                                       |             |
| `model.kwargs.hidden_dim`               | `state_sm.yaml`        |               | `672`                                        |            |
| `model.kwargs.transformer_backbone_key` | `state_sm.yaml`        |               | "llama"                                      |            |
| `training.max_steps`                    | `default.yaml`         | `40000`       | `40000`                                      | ✅(确认)   |
| `training.ckpt_every_n_steps`           | `default.yaml`         | `2000`        | `2000`                                       | ✅(确认)   |
| `wandb.tags`                            | `default.yaml`         | `[]`          | "[\"first\_run\"]"                           | ✅         |
| `name`                                  | `config.yaml`          | "debug"       | "first\_run"                                 | ✅         |
| `output_dir`                            | `config.yaml`          | ".\/debugging"| "competition"                                | ✅         |



## STATE 模型的关键调整

从最终配置中可以确认,STATE 模型的两个关键调整都正确实现:

1.  **Basal Encoder 简化为线性层**:

    ```yaml
    model.kwargs.use_basal_projection: false
    ```

2.  **残差连接转移到最终表达预测空间**:

    ```yaml
    model.kwargs.predict_residual: true
    ```

## 自动保存机制

训练脚本在`src/state/_cli/_tx/_train.py` 第49行执行:


```python
with open(join(run_output_dir, "config.yaml"), "w") as f:
    f.write(cfg_yaml)
```



这确保了最终的合并配置被保存到 `competition/first_run/config.yaml`,提供完整的实验可重现性记录。

## 总结

Hydra 配置系统通过以下步骤生成最终配置:

1.  **加载 defaults**: 按`config.yaml` 中定义的顺序加载各配置组
2.  **应用命令行覆盖**: `model=state_sm` 改变配置组选择
3.  **深度合井**: 递归合并所有嵌套配置
4.  **参数覆盖**: 命令行参数具有最高优先级
5.  **自动保存**:将最终配置保存到输出目录

这个过程确保了 STATE 模型获得正确的配置,同时保持了实验的灵活性和可重现性。

