# Agent驱动的面向自动驾驶长尾场景的免训练场景感知系统

## 摘要

自动驾驶感知系统在常规场景上已经取得了显著进展，但在低频、高风险、语义开放的长尾场景中仍然缺乏稳定性。近两年视觉语言模型（Vision-Language Models, VLMs）开始被用于自动驾驶场景理解与问答，但现有工作仍主要集中在离线基准、集中式推理或模型训练范式，尚未充分回答三个更贴近部署的问题：第一，边缘端小模型在长距离感知上的退化能否被系统性补救；第二，在车端实时约束下，云端能力如何选择性接入而不是全量替代；第三，在冻结主模型权重的条件下，系统是否能够通过测试时反思与外部技能注入实现可复用的能力扩展[1-23]。

本文提出一种面向自动驾驶长尾场景的 training-free hierarchical agent framework。该框架以边缘端小型 VLM 作为高频基础感知器，以云端大模型作为低频反思器，并以结构化 skill 作为外部知识单元，构成“边缘感知 - 不确定性触发 - 云端纠正 - 技能沉淀 - 再感知”的闭环。与直接采用云端输出不同，本文强调将反思结果编译为可检索、可约束、可复用的 skill，从而使系统具备部署条件下的测试时自适应能力，而不必频繁更新模型权重[24-33,48-53]。

在实验上，本文以 DTPQA 作为主要定量评测基准，并结合真实系统 artifact 对 skill refinement 机制进行案例分析。清洗后的 `DTP-Synth / Category 1` 三路对比结果显示：`edge_only`、`cloud_only` 与 `hybrid` 的准确率分别达到 `90.0%`、`96.0%` 和 `98.0%`；在 far-distance 子集上三者分别为 `73.3%`、`86.7%` 和 `93.3%`。相对于 `edge_only`，`hybrid` 修复了 `80%` 的边缘错误且未引入新的正确样本退化；相对于 `cloud_only`，`hybrid` 将云调用比例降低 `82%`，平均延迟降低 `74.9%`，并在当前 clean subset 上略优于 `cloud_only`。此外，系统能够把云端反思结果编译为结构化 skill，对关注区域、问题树、输出约束和回退标签进行显式表示。本文认为，面向自动驾驶部署的有效路径，不应简单理解为“把更大模型接到系统后面”，而应理解为“在边缘实时性约束下，以选择性云协同和结构化 skill 沉淀实现测试时能力增强”。

**关键词**：自动驾驶；长尾场景；视觉语言模型；边缘-云协同；training-free；skill refinement；DTPQA

## 1. 引言

自动驾驶系统的核心难点，已经逐渐从常规目标检测和车道线识别，转向对低频、高风险、语义开放场景的稳定理解。相关研究已经表明，真实道路中的关键失败不只是“看不见常见目标”，更常表现为对 corner case、异常交互、尺度变化、距离变化和复杂上下文的理解失效[7-21]。在这一背景下，VLM 因其将视觉感知与语言语义耦合到统一推理框架中的能力，成为自动驾驶场景理解的重要候选方案[11-23,34-47]。

但将 VLM 真正用于自动驾驶部署，仍存在三个没有被充分解决的问题。首先，模型能力与部署约束之间存在明显张力。大量工作展示了大模型在多模态问答和驾驶推理中的潜力，但这些结果通常默认有充分中心化算力，而车端实际部署受限于时延、显存、热设计与稳定性，难以直接承载大模型常驻推理[8,11-23]。其次，已有工作往往强调训练、更大的数据和更复杂的模型，而对“冻结权重条件下如何在测试时增强小模型能力”讨论不足[1,8,24-33,48-53]。第三，尽管 DTPQA 等工作已经揭示小型 VLM 在距离敏感任务上的显著退化，但如何在不破坏实时性的情况下，对这类退化进行系统级补救，仍缺少成熟答案[1,2]。

基于以上观察，本文的核心问题可以表述为：**能否在不更新主模型权重的前提下，构建一个既满足边缘实时性要求，又能利用云端大模型进行定向补救，并能将补救经验沉淀为可复用外部技能的自动驾驶长尾场景感知系统？**

围绕这一问题，本文做出三点贡献。

1. 提出一种 training-free hierarchical agent framework，将边缘端小模型、云端反思模型和结构化 skill store 组织为统一闭环。
2. 提出面向自动驾驶场景的 skill refinement 机制，将云端反思结果从自由文本转化为关注区域、动态问题树、输出约束和回退标签等结构化组件。
3. 在 DTPQA 上给出三路对比结果，并进一步以系统级指标说明：相比纯边缘方案，混合架构可以明显提升长距离感知；相比纯云方案，混合架构可以显著降低云调用成本和平均延迟。

需要特别说明的是，本文将“benchmark-faithful evaluation”和“deployment-oriented adaptation”明确区分。在 DTPQA 的主定量实验中，为避免 benchmark contamination，本文采用不持久化 skill 的干净协议，主评估对象是 selective cloud routing 的增益；而在系统案例部分，再使用真实 artifact 展示 skill refinement 的编译与复用能力。这种双协议设计有助于分别回答“系统是否有效”和“知识是否能够被沉淀”为两个不同问题。

## 2. 相关工作

### 2.1 自动驾驶场景理解与 VQA 基准

自动驾驶 VLM 研究的一条重要主线是构建适合交通场景的问答与理解基准。LingoQA、DriveLM、RoadTextVQA、TB-Bench、DriveMLLM/SURDS、DTPQA 等数据集或任务框架，分别从图结构问答、文本理解、时空交通行为、多模态场景问答和距离敏感感知等角度评估模型能力[2,7-15]。这类工作推动了自动驾驶多模态研究从“是否能回答一个问题”转向“模型到底在什么类型的驾驶场景中失败”。

其中，与本文最相关的是 DTPQA。DTPQA 将问题限定为感知型而非推理型问题，并显式提供距离标注，使得研究者可以直接观察模型随目标距离增长而出现的性能退化[1,2]。这为本文研究边缘端小模型在长距离 pedestrian-presence 问题上的失效模式提供了可操作、可解释的 benchmark 入口。

### 2.2 自动驾驶多模态大模型与系统化驾驶理解

另一条主线关注将大模型直接引入自动驾驶理解与决策过程，例如 SENNA、SimpleLLM4AD、LaVIDA Drive、OmniDrive、V3LMA、LightEMMA、DynRSL-VLM 等工作尝试把场景理解、语言交互、三维信息和规划约束统一在端到端或者准端到端框架中[16-23]。这些工作表明，多模态大模型具有很强的开放语义建模能力，但也暴露出部署成本高、系统链路长和评测协议复杂等问题。

本文与这些工作不同。我们的目标不是设计一个更大的统一驾驶模型，而是探索在**冻结主模型权重**条件下，是否能通过系统级组织方式把小模型、云端能力和结构化技能库组合起来，从而更接近真实部署场景中的“快速适配”需求。

### 2.3 VLM 感知能力与视觉短板

近年来，不少研究指出，即便是能力很强的 VLM，也常在基础视觉感知上表现不稳定。Vision Language Models are Blind、Eyes Wide Shut、How Well Can Vision Language Models See Image Details、VisOnlyQA、What’s in the Image?、HallusionBench 等研究系统分析了 VLM 在几何理解、细节识别、视觉幻觉和局部视觉证据利用上的短板[24-30]。与此同时，MMBench、MMMU、VLMEvalKit 等工作从更广泛的基准层面对多模态模型进行评测[31-33]。

这些研究对本文有两点启发。第一，VLM 的失败并不一定来自“不会推理”，很可能来自对关键视觉证据的编码和利用不足。第二，部署时真正需要的不是一个“在所有任务上都强”的模型，而是一个能够在关键交通视觉任务上稳定、可控工作的系统。因此，本文更关注 selective cloud assistance 与 uncertainty-triggered correction，而不是盲目堆叠更复杂的 chain-of-thought。

### 2.4 多模态基础模型与模型规模演化

CLIP、Flamingo、LLaVA、InternVL、InternVL3、Qwen-VL、Qwen2-VL、Qwen2.5-VL、Ovis、DeepSeek-VL、DeepSeek-VL2、DINOv2、Cambrian-1 等模型构成了本文系统设计的模型背景[34-46]。这些模型的发展说明了两个趋势：一是 VLM 正在快速增强跨模态表示与 instruction-following 能力；二是模型性能提升往往伴随着更大的计算与部署成本。

对于车端应用，这意味着“更强的模型”与“可部署的模型”未必是同一个模型。因此，本文采用分层式方案：边缘端使用较小模型维持低时延主链路，云端使用更强模型承担低频纠正，而不是试图把所有能力压缩到同一个常驻模型中。

### 2.5 Agent、测试时自适应与外部工具

在自然语言和通用 agent 研究中，ReAct、Reflexion、Self-Refine、Toolformer、RAG、REALM、Chain-of-Thought 等工作已经表明，模型能力可以在不更新权重的情况下，通过外部记忆、工具调用、自我反思和测试时迭代得到增强[48-54]。从工程角度看，这些工作提供了一个重要视角：**能力演化不必总是通过参数更新发生，也可以通过上下文组织、工具链接和外部记忆来发生。**

本文借鉴了这一思想，但将其从自然语言任务迁移到自动驾驶场景理解系统中。与通用 agent 不同，自动驾驶系统要求更严格的实时性、结构化输出和失败可解释性，因此本文进一步引入结构化 skill 表示和 selective routing 机制，以适应安全关键应用场景。

### 2.6 基础架构与系统实现背景

从模型构件和系统实现角度，ViT、RoFormer、MoE、FlashAttention-2、Transformers 库等工作为现有多模态模型与推理系统提供了基础设施[55-59]；而在统计显著性与实验设计方面，Noreen 的经典工作为非参数重采样与假设检验提供了方法学参考[60]。这些工作并不是本文的主要贡献来源，但构成了本文方法与实验实现的技术底座。

综上所述，现有文献已经分别讨论了自动驾驶多模态理解、VLM 感知短板、数据基准和测试时工具增强，但仍缺少一个同时面向**长尾场景泛化、边缘-云协同、training-free 适配和结构化 skill 沉淀**的统一系统框架。本文正是试图在这个交叉点上给出一个部署导向的回答。

## 3. 问题定义与研究假设

本文考虑如下问题设定：给定一张来自自动驾驶前视相机的图像 $I$，以及与之对应的任务问题 $q$，系统需要输出结构化场景理解结果 $y$。在工程实现上，$y$ 不仅包含最终问答标签，还包括场景三元组、区域级目标、动作建议和不确定性代理量。与纯问答不同，本文要求系统能够在输出正确性的同时，提供后续评测、审计和技能复用所需的结构化中间表示。

本文围绕三个研究假设展开。

**H1：** 边缘端小模型在 DTPQA 类距离敏感任务上存在显著的 long-range degradation，尤其在 pedestrian presence 这类远距离正样本上容易产生假阴性。

**H2：** 纯云端模型虽然有更高的上限，但若把云端作为默认主链路，其时延与成本难以满足车端实时约束，因此更合理的方式是选择性调用云端而不是全量替代。

**H3：** 在冻结主模型权重条件下，系统仍然可以通过 agent 式测试时增强获得有效收益；这种收益既可以体现在 selective cloud routing 的即时纠错上，也可以体现在反思结果向结构化 skill 的外化上。

围绕以上假设，本文的整体研究问题可以概括为：**如何在训练冻结条件下，用最小的云调用比例和可接受的系统时延，获得对边缘端感知失败样本的定向补救，并进一步把补救知识转化为可复用技能？**

## 4. 方法

### 4.1 总体框架

本文提出的系统是一个分层式 agent 架构，由边缘层、云端层和 skill store 三部分组成。边缘层负责高频基础感知；云端层负责低频反思和纠正；skill store 负责将反思经验沉淀为可检索外部知识。整个系统围绕一个统一的 replay orchestrator 运行，其核心闭环为：

1. 边缘端模型对输入图像执行第一次感知，生成 baseline 结构化结果。
2. 系统根据候选分布熵、fallback 标记和任务规则判断当前结果是否稳定。
3. 若匹配到已有 skill，则将 skill 注入 prompt，对边缘端模型执行约束化再感知。
4. 若仍不稳定，或命中特定长尾触发条件，则调用云端多模态模型进行反思。
5. 云端返回纠正结果，并可进一步编译为新的 skill。
6. 系统将最终结果和中间状态写入统一的 run artifact，供评测与复盘使用。

与“把云端结果直接当最终答案”的简单方案相比，本文的关键区别在于：云端不仅是一个更强的推理器，还是一个**技能编译器**。它把单次纠错转化为结构化知识，从而为后续样本提供测试时复用基础。

### 4.2 边缘层：小模型结构化感知

边缘层采用较小的多模态模型执行直接图像推理。当前实现中，边缘模型为 `Qwen/Qwen3.5-9B`。模型输入为图像、问题与任务上下文，输出统一格式的 `EdgePerceptionResult`，其中包含：

- `general_perception`：全局场景描述；
- `regional_perception`：局部对象与边界框；
- `triplets`：结构化场景图关系；
- `qa_report`：问题对应的问答结果；
- `top_k_candidates`：候选标签与分布；
- `entropy`：由候选分布计算的不确定性代理；
- `recommended_action`：动作建议；
- `latency_ms` 与 `vision_tokens`：成本记录。

这一设计的目的，是把单次推理结果从“自然语言答案”提升为“可操作的结构化状态”。后续的技能检索、云端反思和实验评估都依赖这一结构化表示。

### 4.3 云端层：反思与定向纠错

云端层使用更强的多模态模型对失败或不稳定样本进行低频深推理。当前实现使用 `Kimi-K2.5`。与早期版本的纯文本反思不同，本文最终采用了**direct cloud re-perception** 作为 DTPQA `category_1` 的主要混合路径：即在满足触发条件时，由云端模型直接读取原图像重新做视觉感知，而不是被边缘模型错误解释所锚定。

这一修改来自实验诊断。早期 hybrid 路径中的 MCP 文本反思会把云端模型限制在边缘结果的错误语义上，导致本可被 `cloud_only` 修正的 far false negative 在 hybrid 中仍旧失败。将其改为 direct cloud re-perception 后，系统能够在保持低触发率的同时，更直接地发挥云端视觉能力。

### 4.4 Skill refinement：从反思到结构化技能

本文将 skill 定义为一种可检索、可约束、可复用的外部知识单元。与自由文本 prompt 不同，skill 由 `manifest.json` 和 `SKILL.md` 组成，其中包含如下核心字段：

- `trigger_tags`
- `trigger_embedding_text`
- `focus_region`
- `dynamic_question_tree`
- `output_constraints`
- `fallback_label`

这种设计使得一次反思结果不再只是对当前样本的修正，而能被转化为对未来样本可执行的外部约束。系统中的 `SkillMatcher` 不仅进行 embedding 相似度检索，还结合标签重叠、关键词一致性、天气一致性和同族去重策略进行重排，以避免 skill store 扩张后出现大量语义相近但任务不匹配的技能污染。

### 4.5 两种评测协议

为兼顾 benchmark 公平性和系统完整性，本文区分两种协议：

1. **Benchmark-faithful protocol**：在 DTPQA 主实验中禁用 skill 持久化，避免 benchmark contamination，主评估对象为 `edge_only`、`cloud_only` 与 `hybrid` 的即时纠错效果。
2. **Deployment-oriented protocol**：在系统案例分析中保留 skill 编译与存储链路，验证反思结果是否能被成功编译为可复用结构化技能。

这种分离是必要的。如果在 benchmark 主实验中直接持久化 skill，系统很容易因为历史 skill 污染而高估 hybrid 的真实性能；但如果完全不保留 skill，又无法体现 training-free adaptation 的系统意义。因此本文采用“主实验看干净增益，案例分析看技能沉淀”的双协议设计。

## 5. 实验设置

### 5.1 数据集

本文主要采用 DTPQA 作为定量评测基准。DTPQA 包含 DTP-Synthetic 与 DTP-Real 两部分，并带有对象距离标注，可以用来研究模型在不同距离条件下的视觉感知退化[1,2]。本文当前的主定量结果使用 `DTP-Synth / Category 1`（pedestrian presence）上的 clean 50-sample shared subset。选择这一子集有两个原因：

1. 该任务直接对应边缘端远距离假阴性这一核心失败模式；
2. 当前系统在该任务上的混合路径已经过协议清洗与逻辑修复，可作为 thesis 的最可信定量证据。

此外，本文还使用已有的系统 artifact 作为 deployment-oriented case study，用于展示 skill refinement 的结构化编译结果。

### 5.2 模型与运行环境

当前系统配置如下：

- 边缘模型：`Qwen/Qwen3.5-9B`
- 云端模型：`Pro/moonshotai/Kimi-K2.5`
- 评测裁判模型：`Pro/moonshotai/Kimi-K2.5`
- 嵌入模型：`BAAI/bge-m3`

需要说明的是，虽然本文将其称为“edge”与“cloud”，但当前原型仍通过统一 API 调用模型服务。因此，这里的“edge/cloud”更准确地说是一种**角色分层与推理路径分层**，而不是真实车载硬件上的物理部署切分。这一点在论文讨论部分会进一步说明。

### 5.3 对比模式

本文比较三种执行模式：

- `edge_only`：仅使用边缘模型完成单次感知；
- `cloud_only`：使用云端模型直接完成同一任务；
- `hybrid`：以边缘模型为主链路，仅在触发条件满足时调用云端模型。

在当前 clean 50-sample 实验中，`hybrid` 的触发率为 `18%`，其中未发生历史 skill contamination，所有样本的 `matched_skill_ids` 均为空。

### 5.4 评测指标

为对齐你的毕设主线，本文采用三类指标。

**(1) 任务效果指标**

- `Exact Match Accuracy`
- `Yes Recall`
- `No Specificity`
- `Far Accuracy`

**(2) 系统代价指标**

- `Mean / P50 / P95 Pipeline Latency`
- `Reflection Rate`
- `Cloud-call Reduction vs Cloud-only`
- `Latency Reduction vs Cloud-only`

**(3) 系统机制指标**

- `Hybrid Rescue Rate over Edge Errors`
- `Hybrid Harm Rate over Edge Correct Cases`
- `Oracle(edge, cloud) Accuracy`
- `Reflection Precision`
- `Skill Match Rate`
- `Unique Matched Skill Count`

这些指标分别对应你的 PDF 中提出的“泛化性能”“车云协同实时性”和“测试时加强/skill refinement”三条主线。

## 6. 实验结果

### 6.1 与已发表 DTPQA baseline 的对比

DTPQA 原始数据集论文主要描述 benchmark 构建，而模型 baseline 来自 Theodoridis 等人的评测论文[1,2]。该论文显示，已发表小型 VLM 在全 DTPQA 上的平均表现仍然较弱，最佳小模型平均值仅为 `59.4%`；在与本文最相关的 `DTP-Synth / Cat.1` 上，最佳已发表小模型为 `Ovis2-2B`，准确率为 `71.5%`[1]。

表 1 给出已发表 baseline 与本文当前 clean subset 结果的对比。需要强调的是，本文的结果基于 `Cat.1-Synth` 的 clean 50-sample subset，而非完整 DTPQA 全量结果，因此这张表更适合作为**任务定向对比**而不是 full benchmark 排名。

| 方法 | 评测范围 | Cat.1-Synth Acc. | Cat.1-Synth Neg. Spec. | 备注 |
| --- | --- | ---: | ---: | --- |
| Human | DTPQA 原论文 | 95.7 | 100.0 | 已发表 |
| Ovis2-2B | DTPQA 原论文 | 71.5 | 100.0 | 已发表最佳小模型 |
| InternVL2.5-2B-MPO | DTPQA 原论文 | 71.0 | 100.0 | 已发表 |
| Qwen2-VL-2B | DTPQA 原论文 | 61.4 | 100.0 | 已发表 |
| Qwen2.5-VL-3B | DTPQA 原论文 | 17.3 | 100.0 | 已发表 |
| Ours (`edge_only`) | Clean 50-sample subset | 90.0 | 100.0 | 当前实现 |
| Ours (`cloud_only`) | Clean 50-sample subset | 96.0 | 100.0 | 当前实现 |
| Ours (`hybrid`) | Clean 50-sample subset | 98.0 | 100.0 | 当前实现 |

可以看到，在当前 clean subset 上，本文系统的三种模式都超过了已发表小模型 baseline，其中 `hybrid` 达到了最好的整体效果。这说明：对于 pedestrian presence 这类距离敏感任务，training-free 的系统级改造比单纯更换一个小模型更有潜力。

### 6.2 三路实验的整体对比

在当前 clean 50-sample shared subset 上，三路实验的核心结果如表 2 所示。

| 模式 | Baseline Acc. | Final Acc. | Yes Recall | No Specificity | Far Acc. | Mean Latency (ms) | Reflection Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `edge_only` | 0.9000 | 0.9000 | 0.8837 | 1.0000 | 0.7333 | 13,765.54 | 0.0000 |
| `cloud_only` | 0.9600 | 0.9600 | 0.9535 | 1.0000 | 0.8667 | 101,053.60 | 0.0000 |
| `hybrid` | 0.9000 | 0.9800 | 0.9767 | 1.0000 | 0.9333 | 25,330.26 | 0.1800 |

这一结果说明了三点。

第一，`edge_only` 已经比 DTPQA 论文中的多数小模型更强，但在 far-distance 上仍然明显退化，说明当前失败不在于整体场景理解完全崩溃，而在于远距离正样本识别不足。

第二，`cloud_only` 给出了更高的上限，证明更强模型在该任务上的确拥有补救潜力。但 `cloud_only` 的平均端到端延迟超过 `100s`，显然不适合作为车端默认主链路。

第三，`hybrid` 实现了本文最关心的 trade-off：它只在少量样本上调用云端，却获得了高于 `edge_only` 甚至略高于当前 `cloud_only` 的最终准确率。这说明 selective routing 是有效的，而且不是“全面转向云端”的伪提升。

### 6.3 系统级收益分析

为了更直接回答“整个系统是否有效”，本文进一步报告选择性路由与系统收益指标：

- `Hybrid rescue rate over edge errors = 0.80`
- `Hybrid harm rate over edge-correct cases = 0.00`
- `Far rescued case count = 3`
- `Oracle(edge, cloud) accuracy upper bound = 0.98`
- `Hybrid cloud-call reduction vs cloud-only = 0.82`
- `Hybrid latency reduction vs cloud-only = 0.7493`
- `Hybrid accuracy delta vs edge = +0.08`
- `Hybrid accuracy delta vs cloud = +0.02`

这些结果表明，`hybrid` 不是靠扩大云调用换来的准确率提升，而是在相对有限的云调用比例下，定向修复了边缘端最脆弱的样本。特别地，`hybrid harm rate = 0` 说明当前 routing 机制没有明显破坏边缘端本来已经正确的样本，这对安全关键应用尤为重要。

### 6.4 为什么早期 hybrid 会变差，而现在 hybrid 会变好

早期实验中，hybrid 路径曾出现“最终效果反而下降”的现象。经过代码审计和对照实验，本文将原因归结为两类。

第一类是**实验控制问题**：批处理脚本会吞掉失败样本、MCP 健康检查与实际 client 端口不一致、历史 skill store 污染 benchmark 运行。这些问题会使实验看起来“完成了”，但实际上混入了历史状态和部分失败样本。

第二类是**架构逻辑问题**：旧版 MCP 文本反思会把云端模型锚定在边缘端错误解释之上，导致本来 `cloud_only` 可以修正的 far false negative，在 `hybrid` 中仍然失败。本文将其替换为 direct cloud re-perception 后，`hybrid` 才真正释放出 selective cloud assistance 的价值。

因此，本文认为，“旧 hybrid 变差”并不能证明边缘-云混合架构无效，反而说明：**在自动驾驶场景中，混合架构的有效性高度依赖于协议洁净性和反思路径设计。**

## 7. Skill refinement 的系统证据

### 7.1 为什么 benchmark 主实验里看不到大量 skill 增益

这里需要诚实说明：在当前 thesis 主定量实验中，DTPQA 的 clean benchmark-faithful protocol 禁用了 skill persistence，因此 `hybrid_clean_v3` 的 `skill_match_rate = 0`、`unique_matched_skill_count = 0`。这不是 skill refinement 无效，而是为了避免 benchmark contamination，主动把“技能学习”从主定量评测中拿掉。

换言之，当前 DTPQA clean 结果主要证明的是**selective cloud routing 有效**，而不是“大规模 skill accumulation 已被完整验证”。如果把两者混为一谈，会导致论文论证链条失真。

### 7.2 真实 artifact 中的 skill 编译结果

尽管 DTPQA 主实验不持久化 skill，当前系统已经在其他真实 run 中实现了完整 skill 编译链路。以 `lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c` 为例，系统成功生成了如下结构化 skill：

- `trigger_tags`: `lead_vehicle_ahead`, `pedestrian_near_roadside`, `worker_at_curb`, `narrowed_lane_margin`, `daylight_clear`
- `focus_region`: `center_lane_20_40m_ahead_plus_right_curb`
- `dynamic_question_tree`: “前方是否存在直接约束自车的车辆”“路侧是否有行人或作业人员”“右侧边界是否可通行”“最终动作是什么”
- `output_constraints`: 强制输出 `Lead_Vehicle`、`Pedestrian`、`slow_down` 等关键约束
- `fallback_label`: `Critical_Unknown_Obstacle`

这一 artifact 表明，系统已经能够把一次失败案例中的云端理解转化为可执行、可审计、可复用的结构化知识单元，而不是仅仅生成一段“人类看得懂但系统用不了”的解释文本。对于你的毕设主线来说，这一点非常关键，因为它支撑了“training-free 不是不学习，而是把学习从权重空间转移到了外部技能空间”这一论点。

### 7.3 本文对 skill refinement 的定位

基于当前证据，本文将 skill refinement 定位为两层贡献：

1. **系统设计贡献**：提出并实现了从反思到结构化 skill 的编译路径；
2. **下一阶段定量扩展方向**：在干净 adaptation/evaluation split 上验证 skill 的覆盖率、精度和可迁移性。

因此，在当前论文版本中，skill refinement 最适合作为“系统创新点 + 真实 artifact 证据 + 后续可扩展实验方向”来论述，而不宜过度夸大为“已经在大规模 benchmark 上被完整量化验证”的结论。

## 8. 讨论

### 8.1 本文系统为什么适合部署导向研究

本文方法之所以有意义，不在于它达到了某个单一 benchmark 的更高分数，而在于它更接近真实部署所关心的三个问题：

1. 当边缘端模型失败时，系统是否能做定向补救？
2. 这种补救是否会破坏实时性与成本边界？
3. 补救经验是否能被沉淀，而不是每次都从零开始？

从当前结果看，这三个问题至少得到了部分肯定回答。`hybrid` 修复了大部分边缘端错误；云调用比例被压缩在 `18%`；系统已经具备 skill 编译基础设施。虽然离完整部署还有距离，但 thesis 所需要证明的“方法路径合理性”已经基本建立。

### 8.2 本文与“直接上更大模型”的差异

一个自然问题是：既然 `cloud_only` 已经比 `edge_only` 更强，为什么不直接使用云端大模型？

答案在于部署代价。当前 `cloud_only` 的平均延迟是 `101,053.60 ms`，约为 `edge_only` 的 7 倍以上；即便其准确率较高，也难以作为车端常驻方案。本文的贡献就在于证明：**更合理的系统形式不是“纯小模型”或“纯大模型”，而是“以小模型为主、以大模型为选择性补偿”的混合结构。**

### 8.3 本文与传统 training-based adaptation 的差异

本文并不否认训练与微调的重要性，但在自动驾驶长尾场景中，频繁回流数据、重新标注和重新验证模型权重，往往意味着更长的闭环周期和更高的系统风险。相比之下，training-free agent 方法把适配能力迁移到推理时上下文组织、外部工具和结构化技能库，能更快地面向新场景作出工程响应。这一思路尤其适合本科毕业设计中的“系统方法验证”范式，因为它强调可实现性、可解释性与部署约束下的有效性。

## 9. 局限性与未来工作

本文也存在清晰的局限。

第一，当前最强的定量证据仍来自 `DTP-Synth / Cat.1` 的 clean 50-sample subset，而不是 DTPQA 全量同日三路评测。因此，本文目前更适合作为“prototype thesis with strong targeted evidence”，而不是一个已经完整封闭的大规模 benchmark paper。

第二，当前 skill refinement 在 benchmark-faithful 协议中被主动关闭，因此本文尚未给出“大规模 skill accumulation 明确提升 benchmark 指标”的强定量结果。下一步应当构建独立 adaptation/evaluation split，在不污染 benchmark 的前提下测试 skill 覆盖率、precision、复用收益和学习曲线。

第三，当前所谓 edge/cloud 仍然是逻辑角色分层，而不是真正车载端侧硬件与远程云资源之间的物理部署切分。未来需要在真实边缘硬件上重新测量端到端时延、吞吐和带宽消耗。

第四，当前系统主要在 pedestrian presence 这一距离敏感问题上验证了 selective cloud routing 的价值。后续可以扩展到方向判断、blinker 识别、交通标志和更复杂的开放语义 long-tail scene。

## 10. 结论

本文围绕“自动驾驶长尾场景下如何在冻结权重条件下增强小模型感知能力”这一问题，提出了一种 training-free hierarchical agent framework。系统以边缘小模型为主链路，以云端大模型为选择性纠错器，以结构化 skill store 作为外部知识容器，形成了边缘感知、云端反思和知识沉淀相结合的闭环。

当前 clean 实验表明，混合架构能够同时获得比纯边缘方案更高的准确率、比纯云方案更低的系统代价，并在 far-distance 子集上显著缓解边缘小模型的假阴性问题。进一步地，真实系统 artifact 证明，云端反思结果可以被编译为具有触发条件、关注区域和输出约束的结构化 skill，为后续面向部署的测试时适配提供了可行路径。

因此，本文的核心结论不是“更大的模型一定更好”，而是：**在自动驾驶长尾场景中，一种以选择性云协同与结构化技能外化为核心的 training-free agent 系统，是比纯模型规模扩张更贴近部署约束、也更具工程可执行性的方案。**

## 参考文献

[1] Nikos Theodoridis, Tim Brophy, Reenu Mohandas, Ganesh Sistu, Fiachra Collins, Anthony Scanlan, and Ciaran Eising. *Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception*. IEEE Open Journal of Vehicular Technology, 2025.

[2] Nikos Theodoridis, Tim Brophy, Reenu Mohandas, Ganesh Sistu, Fiachra Collins, Anthony Scanlan, and Ciaran Eising. *Descriptor: Distance-Annotated Traffic Perception Question Answering (DTPQA)*. CoRR abs/2511.13397, 2025. [ArXiv](https://arxiv.org/abs/2511.13397)

[3] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. *nuScenes: A Multimodal Dataset for Autonomous Driving*. CVPR, 2020.

[4] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio López, and Vladlen Koltun. *CARLA: An Open Urban Driving Simulator*. CoRL, 2017.

[5] Ming Liu, Erfan Yurtsever, J. Fossaert, Xiaotong Zhou, Walter Zimmer, Yunlong Cui, Bence L. Zagar, and Alois C. Knoll. *A Survey on Autonomous Driving Datasets: Statistics, Annotation Quality, and a Future Outlook*. IEEE Transactions on Intelligent Vehicles, 2024.

[6] P. Kaur, S. Taghavi, Z. Tian, and W. Shi. *A Survey on Simulators for Testing Self-Driving Cars*. MetroCAD, 2021.

[7] Ana-Maria Marcu, L. Chen, J. Hünermann, A. Karnsund, B. Hanotte, P. Chidananda, S. Nair, V. Badrinarayanan, A. Kendall, J. Shotton, E. Arani, and O. Sinavski. *LingoQA: Visual Question Answering for Autonomous Driving*. ECCV, 2024.

[8] Shijie Xie, L. Kong, Y. Dong, C. Sima, W. Zhang, Q. A. Chen, Z. Liu, and L. Pan. *Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives*. CoRR abs/2501.04003, 2025. [ArXiv](https://arxiv.org/abs/2501.04003)

[9] G. Tom, M. Mathew, S. Garcia, D. Karatzas, and C. V. Jawahar. *Reading Between the Lanes: Text VideoQA on the Road*. ICDAR, 2023.

[10] K. Chen, Y. Li, W. Zhang, Y. Liu, P. Li, R. Gao, L. Hong, M. Tian, X. Zhao, Z. Li, D.-Y. Yeung, H. Lu, and X. Jia. *Automated Evaluation of Large Vision-Language Models on Self-Driving Corner Cases*. WACV, 2025.

[11] Chen Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, P. Luo, A. Geiger, and H. Li. *DriveLM: Driving with Graph Visual Question Answering*. ECCV, 2024.

[12] X. Ding, J. Han, H. Xu, X. Liang, W. Zhang, and X. Li. *Holistic Autonomous Driving Understanding by Bird’s-Eye-View Injected Multi-Modal Large Models*. CVPR, 2024.

[13] K. Charoenpitaks, V.-Q. Nguyen, M. Suganuma, K. Arai, S. Totsuka, H. Ino, and T. Okatani. *TB-Bench: Training and Testing Multi-Modal AI for Understanding Spatio-Temporal Traffic Behaviors from Dashcam Images/Videos*. CoRR abs/2501.05733, 2025. [ArXiv](https://arxiv.org/abs/2501.05733)

[14] X. Guo, R. Zhang, Y. Duan, Y. He, C. Zhang, S. Liu, and L. Chen. *DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving*. CoRR abs/2411.13112, 2024. [ArXiv](https://arxiv.org/abs/2411.13112)

[15] T. Choudhary, V. Dewangan, S. Chandhok, S. Priyadarshan, A. Jain, A. K. Singh, S. Srivastava, K. M. Jatavallabhula, and K. M. Krishna. *Talk2BEV: Language-Enhanced Bird’s-Eye View Maps for Autonomous Driving*. CoRR abs/2310.02251, 2023. [ArXiv](https://arxiv.org/abs/2310.02251)

[16] A. Gopalkrishnan, R. Greer, and M. Trivedi. *Multi-Frame, Lightweight and Efficient Vision-Language Models for Question Answering in Autonomous Driving*. CoRR abs/2403.19838, 2024. [ArXiv](https://arxiv.org/abs/2403.19838)

[17] B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang, W. Liu, and X. Wang. *SENNA: Bridging Large Vision-Language Models and End-to-End Autonomous Driving*. CoRR abs/2410.22313, 2024. [ArXiv](https://arxiv.org/abs/2410.22313)

[18] P. Zheng, Y. Zhao, Z. Gong, H. Zhu, and S. Wu. *SimpleLLM4AD: An End-to-End Vision-Language Model with Graph Visual Question Answering for Autonomous Driving*. CoRR abs/2407.21293, 2024. [ArXiv](https://arxiv.org/abs/2407.21293)

[19] S. Jiao and Y. Fang. *LaVIDA Drive: Vision-Text Interaction VLM for Autonomous Driving with Token Selection, Recovery and Enhancement*. CoRR abs/2411.12980, 2024. [ArXiv](https://arxiv.org/abs/2411.12980)

[20] S. Wang, Z. Yu, X. Jiang, S. Lan, M. Shi, N. Chang, J. Kautz, Y. Li, and J. M. Alvarez. *OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning*. CoRR abs/2405.01533, 2024. [ArXiv](https://arxiv.org/abs/2405.01533)

[21] J. Lübberstedt, E. Rivera, N. Uhlemann, and M. Lienkamp. *V3LMA: Visual 3D-Enhanced Language Model for Autonomous Driving*. CVPR, 2025.

[22] Z. Qiao, H. Li, Z. Cao, and H. X. Liu. *LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving*. CoRR abs/2505.00284, 2025. [ArXiv](https://arxiv.org/abs/2505.00284)

[23] X. Zhou, L. Shan, and X. Gui. *DynRSL-VLM: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models*. CoRR abs/2503.11265, 2025. [ArXiv](https://arxiv.org/abs/2503.11265)

[24] P. Rahmanzadehgervi, L. Bolton, M. R. Taesiri, and A. T. Nguyen. *Vision Language Models Are Blind*. ACCV, 2024.

[25] S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, and S. Xie. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs*. CVPR, 2024.

[26] C. Gou, A. Felemban, F. F. Khan, D. Zhu, J. Cai, H. Rezatofighi, and M. Elhoseiny. *How Well Can Vision Language Models See Image Details?* CoRR abs/2408.03940, 2024. [ArXiv](https://arxiv.org/abs/2408.03940)

[27] R. Kamoi, Y. Zhang, S. S. S. Das, R. H. Zhang, and R. Zhang. *VisOnlyQA: Large Vision Language Models Still Struggle with Visual Perception of Geometric Information*. CoRR abs/2412.00947, 2024. [ArXiv](https://arxiv.org/abs/2412.00947)

[28] O. Kaduri, S. Bagon, and T. Dekel. *What’s in the Image? A Deep-Dive into the Vision of Vision Language Models*. CVPR, 2025.

[29] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, and F. Zhao. *Are We on the Right Way for Evaluating Large Vision-Language Models?* NeurIPS, 2024.

[30] T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen, F. Huang, Y. Yacoob, D. Manocha, and T. Zhou. *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models*. CVPR, 2024.

[31] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, K. Chen, and D. Lin. *MMBench: Is Your Multi-Modal Model an All-Around Player?* ECCV, 2024.

[32] X. Yue, Y. Ni, T. Zheng, K. Zhang, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, C. Wei, B. Yu, R. Yuan, R. Sun, M. Yin, B. Zheng, Z. Yang, Y. Liu, W. Huang, H. Sun, Y. Su, and W. Chen. *MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*. CVPR, 2024.

[33] H. Duan, J. Yang, Y. Qiao, X. Fang, L. Chen, Y. Liu, X. Dong, Y. Zang, P. Zhang, J. Wang, et al. *VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models*. ACM Multimedia, 2024.

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.

[35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. *Visual Instruction Tuning*. NeurIPS, 2023.

[36] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karl Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. *Flamingo: A Visual Language Model for Few-Shot Learning*. NeurIPS, 2022.

[37] Zhe Chen, Jiarui Wu, Wenhai Wang, Wei Su, Guanzhong Chen, Shaohua Xing, et al. *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks*. CVPR, 2024.

[38] Jiahao Zhu, Wenhai Wang, Zhe Chen, et al. *InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models*. CoRR abs/2504.10479, 2025. [ArXiv](https://arxiv.org/abs/2504.10479)

[39] Shuai Bai, Kuan Chen, Xingxuan Liu, et al. *Qwen2.5-VL Technical Report*. CoRR abs/2502.13923, 2025. [ArXiv](https://arxiv.org/abs/2502.13923)

[40] Peng Wang, Shuai Bai, Shijie Tan, et al. *Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution*. CoRR abs/2409.12191, 2024. [ArXiv](https://arxiv.org/abs/2409.12191)

[41] Jinze Bai, Shuai Bai, Shusheng Yang, et al. *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*. CoRR abs/2308.12966, 2023. [ArXiv](https://arxiv.org/abs/2308.12966)

[42] Shichao Lu, Yiyang Li, Q.-G. Chen, et al. *Ovis: Structural Embedding Alignment for Multimodal Large Language Model*. CoRR abs/2405.20797, 2024. [ArXiv](https://arxiv.org/abs/2405.20797)

[43] Zhihong Wu, Xinyu Chen, Zhen Pan, et al. *DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding*. CoRR abs/2412.10302, 2024. [ArXiv](https://arxiv.org/abs/2412.10302)

[44] Haotian Lu, Wenjie Liu, Baiming Zhang, et al. *DeepSeek-VL: Towards Real-World Vision-Language Understanding*. CoRR abs/2403.05525, 2024. [ArXiv](https://arxiv.org/abs/2403.05525)

[45] Maxime Oquab, Timothée Darcet, Theo Moutakanni, et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR, 2024.

[46] Shengbang Tong, Ellis Brown, Pengchuan Wu, et al. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs*. NeurIPS, 2024.

[47] Siyuan Chen, Tianhang Zhu, Rui Zhou, et al. *Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas*. ICML, 2025.

[48] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR, 2023. [ArXiv](https://arxiv.org/abs/2210.03629)

[49] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS, 2023. [ArXiv](https://arxiv.org/abs/2303.11366)

[50] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023 Workshop / CoRR abs/2303.17651, 2023. [ArXiv](https://arxiv.org/abs/2303.17651)

[51] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS, 2023. [ArXiv](https://arxiv.org/abs/2302.04761)

[52] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-Tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS, 2020. [ArXiv](https://arxiv.org/abs/2005.11401)

[53] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. *REALM: Retrieval-Augmented Language Model Pre-Training*. ICML, 2020. [ArXiv](https://arxiv.org/abs/2002.08909)

[54] Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS, 2022. [ArXiv](https://arxiv.org/abs/2201.11903)

[55] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR, 2021. [ArXiv](https://arxiv.org/abs/2010.11929)

[56] Jianlin Su, Yu Lu, Shengfeng Pan, et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Neurocomputing, 2024. [ArXiv](https://arxiv.org/abs/2104.09864)

[57] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR, 2017. [ArXiv](https://arxiv.org/abs/1701.06538)

[58] Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR, 2024.

[59] Thomas Wolf, Lysandre Debut, Victor Sanh, et al. *Transformers: State-of-the-Art Natural Language Processing*. EMNLP System Demonstrations, 2020.

[60] Eric W. Noreen. *Computer-Intensive Methods for Testing Hypotheses*. Wiley, 1989.
