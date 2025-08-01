让我们从模型架构的概览开始，该架构在 `src/state/tx/models/state_transition.py` 中定义为 `StateTransitionPerturbationModel`。

### 模型架构概述

该模型旨在根据细胞的初始（基础）状态和施加的扰动，预测扰动后细胞的基因表达。它将一组细胞视为一个“句子”，并使用 Transformer 来学习从基础状态到一组细胞的扰动状态的映射。

以下是数据流的可视化分解：

1.  **输入**：模型为一个细胞“句子”接收两个主要输入：
    *   **基础细胞嵌入 (Basal Cell Embeddings)**：一个表示一组细胞初始状态的张量。
        *   形状: `[batch_size, cell_set_len, input_dim]`
    *   **扰动嵌入 (Perturbation Embeddings)**：一个表示施加于这些细胞的扰动的张量。
        *   形状: `[batch_size, cell_set_len, pert_dim]`

2.  **编码 (Encoding)**：这些输入被投影到一个共享的 `hidden_dim` 空间中。
    *   `basal_encoder` (MLP): `[..., input_dim]` -> `[..., hidden_dim]`
    *   `pert_encoder` (MLP): `[..., pert_dim]` -> `[..., hidden_dim]`

3.  **融合 (Fusion)**：编码后的表示进行逐元素相加。
    *   `combined_input` = `basal_encoding` + `pert_encoding`
    *   形状: `[batch_size, cell_set_len, hidden_dim]`

4.  **Transformer 主干网络 (Transformer Backbone)**：`combined_input` 由一个 Transformer 模型（如 LLaMA 或 GPT-2）处理。这使得模型能够学习集合中细胞之间的复杂关系。
    *   输入: `[batch_size, cell_set_len, hidden_dim]`
    *   输出: `[batch_size, cell_set_len, hidden_dim]`

5.  **输出投影 (Output Projection)**：Transformer 的输出被投影到期望的输出维度。
    *   `project_out` (MLP): `[..., hidden_dim]` -> `[..., output_dim]`

6.  **残差连接 (Residual Connection)**：如果 `predict_residual: true`，投影的输出会与初始的基础细胞嵌入相加。这意味着模型学习预测表达的*变化*，而不是最终的表达值。

7.  **损失计算 (Loss Calculation)**：模型的预测结果与真实的扰动细胞表达使用分布损失函数（例如 `energy` 或 `sinkhorn` 距离）进行比较，这种损失函数很适合比较点集。

现在，让我们逐一查看 `training_configs/config.yaml` 文件。

---

### `data` 部分

此部分配置数据加载和预处理流程。这里的参数定义了原始数据如何转换为模型将使用的张量。

| 参数 | 描述 | 维度 / 效果 |
| :--- | :--- | :--- |
| `name` | 要使用的数据模块类的名称。这里指定为 `PerturbationDataModule`。 | --- |
| `toml_config_path` | 指向包含数据集特定配置（如数据文件路径）的 TOML 文件的路径。 | --- |
| `embed_key` | AnnData 对象 (`.obsm`) 中用于细胞嵌入的键。如果为 `null`，则表示使用 `.X` 中的基因表达值。 | 模型的 `input_dim` 由此决定。如果是嵌入，`input_dim` 是嵌入大小。如果为 `null`，`input_dim` 是基因数量。 |
| `output_space` | 指定模型的输出空间。可以是 `'gene'` (高变基因), `'all'` (所有基因), 或 `'latent'` (潜空间)。 | 决定 `output_dim`。如果为 `'gene'` 或 `'all'`，`output_dim` 是基因数量。如果为 `'latent'`，`output_dim` 是潜空间维度。 |
| `pert_rep` | 扰动的表示方式。`'onehot'` 表示独热向量。 | `pert_dim` 由此决定。对于 `'onehot'`，`pert_dim` 是可能的扰动数量。 |
| `basal_rep` | 如何表示基础（对照）细胞。`'sample'` 表示从对照细胞中采样。 | --- |
| `num_workers` | 用于数据加载的 CPU 线程数。 | 更高的值可以加快数据加载速度，但会使用更多 CPU。 |
| `pin_memory` | 如果为 `true`，则锁定内存以加快到 GPU 的数据传输。 | 一种性能优化。 |
| `n_basal_samples` | 与每个扰动细胞配对的基础样本数量。 | --- |
| `basal_mapping_strategy` | 如何将基础细胞映射到扰动细胞。`'random'` 表示随机配对。 | --- |
| `should_yield_control_cells` | 如果为 `true`，数据加载器也将提供对照细胞数据。 | 对此模型架构至关重要。 |
| `batch_col` | AnnData 对象中指定每个细胞批次信息的列。 | 如果 `batch_encoder: true`，用于批次校正。 |
| `pert_col` | 标识每个细胞扰动的列。 | --- |
| `cell_type_key` | 标识每个细胞类型的列。 | --- |
| `control_pert` | 对照扰动的名称 (例如, 'non-targeting')。 | --- |
| `map_controls` | 如果为 `true`，将对照细胞映射到扰动。 | --- |
| `perturbation_features_file` | 指向包含预计算扰动特征（例如，来自 ESM-2）的文件的路径。 | 如果使用，`pert_dim` 将是这些特征的维度。 |
| `store_raw_basal` | 如果为 `true`，存储原始的基础表达值。 | --- |
| `int_counts` | 如果为 `true`，使用整数计数作为基因表达。 | --- |
| `barcode` | 如果为 `true`，在数据中包含细胞条形码。 | --- |
| `output_dir` | 保存处理后数据的目录。 | --- |
| `debug` | 如果为 `true`，以调试模式运行（例如，使用较小的数据集）。 | --- |

---

### `model` 部分

这是定义 `StateTransitionPerturbationModel` 架构的核心部分。

| 参数 | 描述 | 维度 / 效果 |
| :--- | :--- | :--- |
| `name` | 要使用的模型类的名称。`PertSets` 是 `StateTransitionPerturbationModel` 的别名。 | --- |
| `checkpoint` | 用于加载权重的模型检查点路径。 | 如果提供，模型将从预训练的权重开始。 |
| `device` | 运行模型的设备 (`'cuda'` 或 `'cpu'`)。 | --- |
| `cell_set_len` | 一个“句子”中的细胞数量。这是 Transformer 的序列长度。 | 设置 `[B, S, E]` 张量中的 `S` 维度。这里 `S=128`。 |
| `blur` | `geomloss` 中 Sinkhorn 损失函数的一个参数，控制最优传输计划的“平滑度”。 | 较小的值使匹配更严格。 |
| `hidden_dim` | 模型的主要隐藏维度。这是 Transformer 内部的嵌入大小。 | 设置 `[B, S, E]` 张量中的 `E` 维度。这里 `E=672`。这是一个决定模型容量的关键参数。 |
| `loss` | 要使用的主要损失函数。`'energy'` 指的是能量距离。也可以是 `'mse'`, `'sinkhorn'`, 或 `'se'` (sinkhorn + energy)。 | --- |
| `n_encoder_layers` | `basal_encoder` 和 `pert_encoder` MLP 中的层数。 | 这里是 4 层。更深的编码器可以学习更复杂的输入表示。 |
| `n_decoder_layers` | `project_out` MLP 中的层数。 | 这里是 4 层。更深的解码器可以从 Transformer 的隐藏状态中学习更复杂的输出映射。 |
| `predict_residual` | 如果为 `true`，模型预测从基础状态到扰动状态的*变化*，并且这个变化被加到基础表达上。 | 如果为 `true`，`prediction = project_out(transformer_output) + basal_embedding`。如果为 `false`，`prediction = project_out(transformer_output)`。 |
| `batch_encoder` | 如果为 `true`，则为每个批次学习一个批次嵌入，并将其添加到 Transformer 的输入中，以校正批次效应。 | 如果为 `true`，将创建一个 `nn.Embedding` 层，其 `num_embeddings` 等于批次数，`embedding_dim` 等于 `hidden_dim`。 |
| `use_basal_projection` | 如果为 `true`，`basal_encoder` 是一个 MLP。如果为 `false`，它是一个单一的线性层。 | `false` 通过为基础细胞使用更简单的投影来简化模型。 |
| `distributional_loss` | 要使用的 `geomloss` 中的特定分布损失。 | 在此配置中与 `loss` 重复，但 `distributional_loss` 是代码中用于初始化 `SamplesLoss` 的那个。 |
| `gene_decoder_bool` | 如果为 `true`，则构建一个基因解码器，将模型的潜输出映射回基因表达空间。 | `false` 表示不会使用基因解码器。 |
| `transformer_backbone_key` | 要使用的 Transformer 类型。这里是 `'llama'`。也可以是 `'GPT2'` 等。 | 这决定了 Transformer 的核心架构。 |
| `transformer_backbone_kwargs` | 传递给 Transformer 构造函数的参数字典。这里定义了 Transformer 的内部架构。 | 见下面的子表。 |

#### `transformer_backbone_kwargs` 子部分

这些参数配置 LLaMA Transformer 模型。

| 参数 | 描述 | 维度 / 效果 |
| :--- | :--- | :--- |
| `max_position_embeddings` | 模型可以处理的最大序列长度。必须 `>= cell_set_len`。 | 这里是 `128`。 |
| `hidden_size` | Transformer 的隐藏大小。必须与模型的 `hidden_dim` 匹配。 | 这里是 `672`。 |
| `intermediate_size` | Transformer 块内部前馈层的大小。 | `2688` 是 `4 * hidden_size`，这是一个标准做法。 |
| `num_hidden_layers` | Transformer 块（层）的数量。 | `4` 层。更多的层会增加模型的深度和容量。 |
| `num_attention_heads` | 多头注意力机制中的注意力头数。 | `8` 个头。`hidden_size` 必须能被这个数整除。 |
| `head_dim` | 每个注意力头的维度。 | `head_dim` = `hidden_size` / `num_attention_heads` = `672 / 8 = 84`。 |

---

### `training` 部分

此部分控制训练循环和优化过程。

| 参数 | 描述 | 维度 / 效果 |
| :--- | :--- | :--- |
| `weight_decay` | L2 正则化强度。 | 有助于防止过拟合。 |
| `batch_size` | 每个训练批次中细胞“句子”的数量。 | 这里是 `16`。每个批次的总细胞数是 `batch_size * cell_set_len` = `16 * 128 = 2048`。 |
| `lr` | Adam 优化器的学习率。 | `0.0001` 是一个常见的起始点。 |
| `max_steps` | 要执行的总训练步数。 | `40000`。 |
| `gradient_clip_val` | 梯度的最大范数。 | 有助于防止梯度爆炸。 |
| `loss_fn` | LightningModule 的 `training_step` 的损失函数。请注意，模型本身使用 `distributional_loss`。这可能有点令人困惑，但在 `model` 部分中的那个是用于核心损失计算的。 | 这里指定了 `mse`，但模型的内部损失函数 (`energy`) 将被使用。这可能是一个未使用的参数或由训练脚本的其他部分使用。 |

`wandb` 部分用于 Weights & Biases 的日志记录，内容很直观。
