# Configs 实验说明

这个目录下的 YAML 文件不是随意的参数模板，而是当前 infra 中已经定义好的几类实验入口。  
每个 config 都对应一个比较明确的研究问题、一个固定的实验实现、以及一组固定的结果输出形式。


## 总览

| Config 文件 | `experiment.type` | 分析对象 | 核心问题 | 主要方法 | 主要指标 | 主要输出 |
| --- | --- | --- | --- | --- | --- | --- |
| `probing_concepts.yaml` | `probing` | encoder + decoder hidden states | 某个 concept 是否能在各层表征中被线性读出 | layer-wise probing | probe accuracy | `summary.json` + probing accuracy plots |
| `patching_decoder_activation.yaml` | `patching` | decoder image-token activations | decoder 某层 image-token activation 是否因果影响属性判断 | activation patching | `probability_change / P(Yes)` | `results/*.json/csv` + `plots/probability_change/` |
| `patching_encoder_component.yaml` | `patching` | encoder block 内部组件 | encoder 中 residual / attention / MLP / qkv 子模块分别贡献多少 | component patching | `probability_change` + ratio views | 5 类 component plots + JSON/CSV |
| `patching_decoder_component.yaml` | `patching` | decoder block 内部组件 | decoder 中 residual / attention / MLP / q/k/v/proj 分别贡献多少 | component patching | `probability_change` + ratio views | 5 类 component plots + JSON/CSV |
| `attention_analysis_decoder.yaml` | `attention_analysis` | decoder attention heads | 哪些 decoder heads 与 affective concepts 最相关 | head-level attention analysis | head importance + allocation stats | heatmaps + top-head detail + JSON/CSV |

## 1. `probing_concepts.yaml`

### 1.1 实验目的

这个实验的目标是回答一个“可读性”问题：  
对于某个概念，比如 `scary`、`soothing` 或一些 non-affective control concepts，这个概念信息是否已经存在于模型的中间表征里，并且可以被一个简单的线性分类器读出来？

它关心的不是因果作用，而是“表征中有没有这类信息”。  
因此，这个实验特别适合用来回答：

- affective information 在 encoder 还是 decoder 中更容易被读出？
- 某个 concept 是早期层就已经可分，还是到了高层才逐渐变得可分？
- affective concepts 与 non-affective concepts 的层级曲线有什么不同？

### 1.2 实验方法

这个实验在实现上是一个标准的 layer-wise probing pipeline，但这里有几个具体细节很重要：

1. 对每个 concept，先读取对应的 SUN CSV 文件。
   CSV 路径由 `dataset.root / "SUN" / "{csv_stem}.csv"` 决定。

2. 对每张图都构造同一个 prompt。
   当前 config 中的 prompt 是：

   ```text
   Describe this image.
   ```

   这个 prompt 会被包装成 Qwen3-VL 的 chat template，然后和图像一起送进模型。

3. 在 forward 过程中，对 encoder 和 decoder 分别挂 hook。

   对 encoder：

   - 会在第一个 encoder block 前挂一个 pre-hook，抓取 encoder input
   - 然后对每个 encoder block 的输出挂 forward hook
   - 因此 encoder 一共有 `num_enc_layers + 1` 个 probing 点
   - 其中第一个点对应 `input`，后面才是 `L0, L1, ...`

   对 decoder：

   - 会在 decoder layer 0 前挂 pre-hook，抓 decoder 输入
   - 然后对每个 decoder layer 输出挂 forward hook
   - 因此 decoder 一共有 `num_dec_layers + 1` 个 probing 点

4. 对 hook 到的 hidden states 做特征抽取。

   - encoder 特征：对空间 token 做 mean-pooling，得到一个固定长度向量
   - decoder 特征：取最后一个 token 位置的 hidden vector

   这样每张图在每一层都会对应一个向量，可以送去训练 probe。

5. 对每一层单独训练一个 probe。

   当前实现固定使用：

   - `StandardScaler`
   - `LogisticRegression`

   每一层都是独立训练、独立测试，因此最后得到的是一条 layer-wise accuracy curve。

6. 训练集/测试集划分采用 stratified split。

   这样可以保证正负样本在 train/test 中的标签比例尽量一致，避免某一层 accuracy 偶然因为划分偏差而失真。

### 1.3 测量指标

这个实验的核心指标只有一个：

- **layer-wise probe accuracy**

也就是说，对每一层的 hidden representation，训练一个线性 probe，然后报告它在测试集上的分类准确率。

这里要特别说明：

- accuracy 高，表示这个 concept 在该层表征中**可线性解码**
- 但这**不代表该层对最终决策有因果作用**
- probing 说明的是“信息存在”，不是“信息被使用”

### 1.4 实验结果形式

这个实验会输出：

- `results/summary.json`
  - 汇总所有 concepts 的 probing 结果
- `results/{concept}_probing_stats.csv`
  - 每个 concept 每一层的 accuracy、train/test 样本数
- `plots/encoder_probing_accuracy.png`
  - encoder 总图
- `plots/decoder_probing_accuracy.png`
  - decoder 总图

图的风格不是简单的单概念折线，而是：

- affective concepts 和 non-affective concepts 分组画
- 每个 concept 一条曲线
- 再叠加组均值曲线和标准差阴影

### 1.5 当前配置中的关键参数

- `dataset.max_samples_per_attr: 0`
  - 表示不截断，使用当前 CSV 中所有可用样本
- `probing.components: [encoder, decoder]`
  - 表示两个部分都跑
- `probing.prompt`
  - 控制 decoder probing 时看到的文本上下文
- `probing.test_size: 0.2`
  - 20% 做测试集

### 1.6 适合回答什么研究问题

- affective concepts 是否在模型表征里可读？
- affective 与 non-affective concepts 的层级曲线是否不同？
- encoder 和 decoder 哪一边更容易承载这些概念信息？

## 2. `patching_decoder_activation.yaml`

### 2.1 实验目的

这个实验不再问“信息能不能被读出”，而是问一个因果问题：

> decoder 某一层的 image-token activation，如果从 positive image 换成 negative image，或者反过来替换，会不会改变模型对属性的判断？

它关注的是 decoder 中与图像位置相关的 token activations 是否真正参与了最终判断。

### 2.2 实验方法

这个实验的实现是 decoder activation patching，当前 config 对应的是 decoder-only、image-token-only 的版本。

具体流程如下：

1. 对每个 concept 构造 positive / negative contrastive pairs。

   - positive：`attribute_label = 1`
   - negative：`attribute_label = 0`

   当前 activation patching 使用的是 global pairing，而不是 class-aware pairing。
   当 `num_pairs: 0` 时，会使用所有可匹配的 pairs。

2. 对每对样本构造同一个 yes/no prompt。

   当前 config 中的 prompt 是：

   ```text
   Does this image contain the attribute {attribute}? Answer with yes, or no.
   ```

3. 对每一对样本，先跑两次前向：

   - clean run：positive image
   - corrupted run：negative image

   这两次前向的作用是建立 clean / corrupted baseline。

4. 推断 input sequence 中哪些位置是 image tokens。

   当前实现不是只靠 `image_token_id`，而是先做一个更强的 heuristic：

   - 在 decoder layer 0 输入位置，比较 clean 和 corrupted hidden states 的差异
   - 对每个 token 位置计算 hidden norm difference
   - 差异特别大的位置被视为 image-token candidates

   如果这种 heuristic 失败，再退回到：

   - 用 `image_token_id` 在 `input_ids` 中找 image token

5. 在每个 decoder layer 上做 activation patching。

   当前这个 config 是：

   - `stage: decoder`
   - `method: activation`
   - `modes: [counterfactual]`

   所以它真正执行的是：

   - 对 negative image 的 forward
   - 在某一层，把 image token positions 上的 hidden states
   - 替换成 positive image 在同一层同一位置的 hidden states

   也就是说，这里 patch 的单位是：

   - 某一层
   - 某些 token positions
   - 整个 hidden vector

   它不是 block 内部结构分解，也不区分 attention / MLP。

### 2.3 测量指标

当前 patching 实验只保留一个主指标：

- **`probability_change`**

本质上，它记录的是每一层 patch 后的 `P(Yes)` 曲线，并用：

- clean baseline（positive image 的 `P(Yes)`）
- corrupted baseline（negative image 的 `P(Yes)`）

作为参考线。

因此你看图时，最直观的问题是：

- patch 某一层之后，`P(Yes)` 有没有往 clean baseline 靠近？

### 2.4 实验结果形式

这个实验会输出：

- `results/summary.json`
- `results/{concept}_results.json`
- `results/{concept}_results.csv`
- `plots/probability_change/{concept}_{mode}_p_yes.png`

因为当前 config 只跑 `counterfactual`，所以每个 concept 主要会对应一张：

- `{concept}_counterfactual_p_yes.png`

### 2.5 当前配置中的关键参数

- `dataset.num_pairs: 0`
  - 表示使用全部 matched pairs
- `patching.method: activation`
  - 指明这不是 component patching
- `patching.stage: decoder`
  - 只在 decoder 上做
- `patching.modes: [counterfactual]`
  - 当前只跑 counterfactual 替换

### 2.6 适合回答什么研究问题

- decoder 的哪些层在利用 image-token activations 做属性判断？
- 如果只替换图像对应 token 的 hidden states，模型判断能被改动多少？
- affective 与 non-affective concepts 的 decoder 因果敏感层是否不同？

## 3. `patching_encoder_component.yaml`

### 3.1 实验目的

这个实验的目标是把 encoder block 内部的贡献进一步分解开，而不是只问“整层有没有作用”。

它关心的问题是：

- encoder block 的整体 residual contribution 是否重要？
- 在一个 block 内，attention 和 MLP 谁更关键？
- 在 attention 内部，q / k / v / proj 哪部分更关键？

因此它是一个**结构分解型的 causal intervention 实验**。

### 3.2 实验方法

这个实验属于 component patching，当前配置跑的是 encoder 版本，并且 levels 1-3 全开。

先看共同的基础流程：

1. 构造 positive / negative pairs。

   对 component patching，当前实现使用的是 **class-aware pairing**：

   - 优先在相同 `class_label` 内配 positive / negative
   - 如果不够，再回退到更宽松的匹配方式

   当 `num_pairs: 0` 时，会尽量用完所有 matched pairs。

2. 对每对样本，positive image 作为 clean，negative image 作为 corrupted。

3. 对指定层先收集 clean component cache。

   在 encoder block 内，会缓存：

   - `attn_only`
   - `mlp_only`
   - `both`
   - `qkv`
   - `proj`

   这里的含义分别是：

   - `attn_only`：attention 模块本身输出的 delta
   - `mlp_only`：MLP 模块本身输出的 delta
   - `both`：整个 block output - block input，也就是 full block residual delta
   - `qkv`：fused `attn.qkv` 模块输出
   - `proj`：attention output projection 输出

4. 再在 corrupted run 上做 patching。

   不同 level 对应不同 patching 对象：

#### Level 1：Block-level residual baseline

- patch `both`
- 含义是：把整个 block 的 residual delta 用 clean 的版本替换掉
- 这是最粗粒度的 baseline，也可以理解为 block-level counterfactual patching

#### Level 2：Component decomposition

- 分别 patch：
  - `attn_only`
  - `mlp_only`
- 目的是比较同一层中 attention 和 MLP 各自的贡献

#### Level 3：Attention submodule decomposition

- patch：
  - `q`
  - `k`
  - `v`
  - 图中额外显示一条 full attention reference 虚线

对于 encoder，这里有一个非常重要的实现细节：

- encoder attention 的 `q/k/v` 不是三个独立模块
- 而是一个 fused `attn.qkv` 线性层一次性输出
- 因此当前实现是：
  - 先拿到 `attn.qkv` 的输出
  - 再沿最后一维三等分
  - 分别把前 1/3、中间 1/3、后 1/3 当作 `q / k / v`

因此 encoder 的 `q/k/v` 是“从 fused output 中切出来的概念子分量”。

虽然内部 patching cache 仍然会收集 `proj`，但当前结果导出和可视化里不会单独展示 `proj`。原因是：

- 在当前 Qwen3-VL attention 实现里，`proj` 的 patching 结果与 full attention reference 完全重合
- 如果同时画出来，灰色虚线会被 `proj` 实线完全盖住，反而降低可读性

### 3.3 测量指标

主指标仍然是：

- **`probability_change / P(Yes)`**

### 3.4 实验结果形式

这个实验的输出比 activation patching 更复杂，会按 level 分目录保存：

- `plots/probability_change/level_1_block_residual/`
- `plots/probability_change/level_2_component_decomposition/`
- `plots/probability_change/level_3_attn_submodule_decomposition/`

同时还会输出：

- `results/summary.json`
- `results/{concept}_results.json`
- `results/{concept}_results.csv`

### 3.5 当前配置中的关键参数

- `patching.stage: encoder`
- `patching.method: component`
- `patching.levels: [1, 2, 3]`
- `patching.components: [attn_only, mlp_only, both]`
- `patching.attention_submodules: [q, k, v, proj]`
- `patching.selected_layers: [0..26]`
  - 表示当前 encoder 各层都跑

### 3.6 适合回答什么研究问题

- encoder 中 affective information 主要来自 attention 还是 MLP？
- encoder attention 中 q/k/v 哪一部分最相关，以及它们与 full attention 的关系如何？
- non-affective control concepts 是否呈现不同的 component structure？

## 4. `patching_decoder_component.yaml`

### 4.1 实验目的

这个实验与 encoder component patching 的总体目标一致，只是把分析对象换成了 decoder block。

它关心：

- decoder 哪些层的 residual baseline 更重要？
- decoder block 内 attention 与 MLP 谁贡献更大？
- decoder attention 内部的 q/k/v 哪些部分对属性判断影响更强，以及它们与 full attention 的关系如何？

### 4.2 实验方法

它的整体框架与 encoder component patching 是同构的：

1. 构造 positive / negative pairs
2. 对每层收集 clean component cache
3. 在 corrupted run 上做 component-level patching
4. 统计 `P(Yes)` 变化
Levels 1-3 的定义与 encoder 版本相同：

- Level 1：`both`
- Level 2：`attn_only` vs `mlp_only`
- Level 3：`q / k / v`，并以 full attention 作为 reference

但真正重要的是实现差异：

#### 与 encoder 的关键差异

对于 decoder：

- `q/k/v/proj` 通常来自独立模块
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj` / `proj`

因此 decoder component patching 中：

- `q/k/v` 可以直接对对应 projection module 的输出做 patch
- 不需要像 encoder 那样从 fused `qkv` 中再手工切分

与 encoder 一样，`proj` 虽然仍会参与内部 cache 收集，但不会单独显示在 level 3 图和结果导出中，因为它与 full attention reference 完全重合。

这意味着：

- encoder 的 q/k/v patching 更像“fused 模块内部切片”
- decoder 的 q/k/v patching 更像“真实子模块级 intervention”

另外：

- `both` 仍然是 block output - block input 得到的 full residual delta
- 其作用仍然是 block-level baseline

### 4.3 测量指标

与 encoder component 一样，主指标是：

- `probability_change / P(Yes)`

### 4.4 实验结果形式

输出目录结构与 encoder component 相同，共 3 个层级目录：

- `level_1_block_residual/`
- `level_2_component_decomposition/`
- `level_3_attn_submodule_decomposition/`

并配套输出：

- `results/summary.json`
- `results/{concept}_results.json`
- `results/{concept}_results.csv`

### 4.5 当前配置中的关键参数

- `patching.stage: decoder`
- `patching.method: component`
- `patching.levels: [1, 2, 3]`
- `patching.selected_layers: [0..35]`
  - 表示当前 decoder 各层都跑
- `num_pairs: 0`
  - 表示使用全部 matched pairs

### 4.6 适合回答什么研究问题

- decoder block 中是 attention 还是 MLP 更主导 affective judgement？
- decoder 的 q/k/v 哪个子模块最敏感？
- 与 encoder 相比，decoder 的组件分布是否更偏向某类模块？

## 5. `attention_analysis_decoder.yaml`

### 5.1 实验目的

这个实验不做 patching，而是做 decoder attention heads 的观测型分析。

它关心的问题是：

- 哪些 decoder heads 对 affective concepts 最敏感？
- 这些 heads 与 non-affective control concepts 相比，有没有明显不同？
- 最后一个 token 在不同层到底把多少 attention 分配给 image tokens、text tokens、keyword tokens？

因此它本质上是 **observational head analysis**，不是 intervention experiment。

### 5.2 实验方法

实现流程如下：

1. 对每个 concept，分别收集 positive 与 negative 样本。

   当前实现中：

- `dataset.num_samples`
  - 解释为每类最多 N 张
  - 如果 `num_samples: 0`，则该类全部使用

2. 对每张图构造与 patching 相同风格的问句 prompt。

   当前配置中：

   ```text
   Does this image contain the attribute {attribute}? Answer with yes, or no.
   ```

3. 对 decoder 每层 self-attention 提取两类信息：

   #### (a) attention weights

   - 目标是拿到 last token position 的注意力分布
   - 即 `attn_scores[:, -1, :]`

   当前实现优先尝试：

   - 在 forward 中打开 `output_attentions=True`

   如果当前模型/transformers 版本没有把 attentions 直接放进输出里，则实验内部会：

   - 临时包装 `self_attn.forward`
   - 从输出 tuple 中抓取 attention weights 作为 fallback

   这层 fallback 是为了兼容不同版本的 Qwen3-VL / transformers 实现。

   #### (b) weighted values

   实现上，当前代码不会直接取最终混合后的 `attn_output`，而是：

   - hook `v_proj`
   - 得到 value states
   - 再结合 last-token attention weights 计算：

   ```text
   attn_weights @ V
   ```

   得到的是 **pre-output-projection per-head context**

   这一步的意义是：

   - 它更接近每个 head 在写入 residual stream 之前准备写的内容
   - 比最终过了 `o_proj` 的混合结果更适合做单 head 分析

4. 处理 GQA。

   如果 decoder 使用的是 GQA，那么：

- query heads 数量
- key/value heads 数量

可能不一致。

当前实现统一将其展开到 **query-head 粒度**：

- 如果 attention weights 或 value states 只有 kv-head 粒度
- 就按 group mapping 重复展开
- 这样最后所有 heatmap 和 top-head 排名都用统一的 query-head 编号

5. 构造 token groups。

当前实验支持：

- `image`
- `text`
- `keyword`

其中：

- `image` mask
  - 优先复用 patching 中的 hidden-diff heuristic
  - 不行再 fallback 到 `image_token_id`
- `text` mask
  - 定义为非 image positions
- `keyword` mask
  - 通过把 concept phrase tokenized 以后，在 `input_ids` 里做完整子序列匹配得到
  - 对多词 concept（例如 `natural light`）同样生效

6. 计算 head importance。

对于每个 `(layer, head)`：

- 收集所有 positive images 的 weighted values
- 收集所有 negative images 的 weighted values
- 计算：

```text
importance = || mean_pos - mean_neg ||_2
```

这个值越大，说明这个 head 对 positive / negative 的响应差异越大。

7. 从 heatmap 中选 top-k heads 做 detail analysis。

对于 top heads，会进一步统计：

- 它在 positive images 上对 `image/text/keyword` 分配了多少注意力
- 它在 negative images 上对这些 token groups 分配了多少注意力

### 5.3 测量指标

这个实验的指标主要有三类：

1. **head importance score**

   - 定义为 `||mean_pos - mean_neg||_2`
   - 用来判断哪些 heads 最 affective-relevant

2. **layer-wise attention allocation**

   - 统计每一层 last-token attention 对 image tokens 与 text tokens 的平均分配
   - 分 positive / negative 两组

3. **top-head token-group attention summary**

   - 对 top-k heads，比较其在 positive / negative 上对 image / text / keyword 的注意力分配

### 5.4 实验结果形式

这个实验会输出三大类图：

1. `plots/attention_allocation/`

- `{concept}_layer_attention_allocation.png`

2. `plots/head_importance/`

- `{concept}_head_importance_heatmap.png`
- `affective_vs_control_heatmap.png`

3. `plots/top_heads_detail/`

- `{concept}_top_head_summary.png`
- `{concept}_top_head_L{layer}_H{head}_attention_detail.png`

此外还会保存：

- `results/summary.json`
- `results/summary.csv`
- `results/{concept}_results.json`
- `results/{concept}_results.csv`

### 5.5 当前配置中的关键参数

- `dataset.num_samples: 0`
  - 表示每类样本都尽量使用全量
- `attention_analysis.top_k_heads: 10`
  - 对每个 concept 选 top-10 heads 做 detail
- `attention_analysis.token_groups`
  - 当前是 `image / text / keyword`

### 5.6 适合回答什么研究问题

- 哪些 decoder heads 最可能是 affective-relevant heads？
- affective heads 与 control heads 的热图模式是否不同？
- top heads 更关注图像区域、普通文本，还是属性关键词本身？

## 如何选择实验

如果你现在的研究问题是下面这些，可以这样选：

### 想看“概念信息能不能被读出”

用：

- `probing_concepts.yaml`

它回答的是：

- 这个概念信息是否存在于某层表征里
- 它在哪些层最容易被线性分开

### 想看“哪一层、哪一部分对判断有因果作用”

用：

- `patching_decoder_activation.yaml`
- `patching_encoder_component.yaml`
- `patching_decoder_component.yaml`

区别是：

- 如果你只关心 decoder image-token activations 本身是否重要，用 `patching_decoder_activation.yaml`
- 如果你想把 block 内部结构拆开分析，用 component patching

### 想看“哪些 decoder heads 最相关”

用：

- `attention_analysis_decoder.yaml`

它特别适合做：

- head ranking
- affective vs control 对照
- top heads 的注意力分配分析

## 最后补充：几个容易混淆的点

1. probing 不是 causal intervention  
   probing 只能说明“信息在不在表征里”，不能说明“模型是不是在用它”。

2. patching 是 intervention  
   patching 改变了模型中间激活，因此更接近因果分析。

3. attention analysis 不是 patching  
   它主要是在观察 head 的 attention pattern 和 weighted values 差异，不直接改模型内部状态。

4. encoder 和 decoder component 的 `q/k/v` 含义并不完全一样  
   - encoder：来自 fused `qkv` 的切分
   - decoder：来自独立 projection modules
