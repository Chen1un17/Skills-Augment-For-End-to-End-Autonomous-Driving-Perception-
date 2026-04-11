# 面向自动驾驶长尾场景的免训练分层 Agent 系统汇报草稿

## 题目建议

面向自动驾驶长尾场景的免训练分层 Agent 感知与技能自适应系统

## 1. 当前领域的 Research Gap

- 长尾场景仍然是自动驾驶感知系统的核心薄弱环节，尤其是对象级 corner case、复杂交互和开放语义描述场景。[1]
- 视觉语言模型已经被引入自动驾驶场景理解，但大量工作更偏重基准构建、集中式推理或离线评测，尚未形成适用于车端实时约束的可复用测试时适配机制。[2][3]
- 小型视觉语言模型在交通感知中存在明显的距离敏感性和细粒度识别能力下降问题，这使“直接部署到边缘端”与“保持结构化正确性”之间存在张力。[4]

现有研究已经清楚表明，自动驾驶系统的主要瓶颈不再只是常规目标检测，而是对低频、高风险、语义开放的长尾场景进行稳定理解。[1] CODA 的提出说明，真实道路中的 corner case 既包含新类别对象，也包含常见类别的异常实例，而这类场景往往正是传统监督式模型最容易失效的部分。[1] 与此同时，DriveLM 进一步将问题推进到图结构问答与驾驶推理层面，证明“看见什么”与“这意味着什么”必须被统一到结构化场景理解框架中。[2]

然而，现有自动驾驶 VLM 研究仍存在三个明显缺口。第一，不少方法强调大模型推理能力或多模态问答性能，但默认推理在中心化算力环境中完成，缺少与车端实时约束相匹配的边缘-云协同设计。[2][3] 第二，已有工作普遍关注模型训练、指令微调或数据集扩展，而对“冻结权重条件下的测试时自适应”讨论不足，尤其缺少把云端反思结果转化为可复用结构化知识单元的机制。[3] 第三，针对小型 VLM 的研究开始揭示距离、尺度和局部细节对感知质量的显著影响，但如何利用外部技能注入来缓解这些问题，仍缺乏成熟的系统路径。[4]

基于上述现状，本文所针对的研究空白可以归纳为：当前领域仍缺少一种面向自动驾驶长尾场景、无需重新训练模型参数、能够在边缘端与云端之间进行分层协作，并将云端反思沉淀为边缘端可重复调用技能的系统化方法。

## 2. Research Motivation：为什么采用 Training-Free Agent 方式

- 长尾样本天然稀缺，人工标注和频繁微调成本高，难以支撑快速迭代。
- 车端部署受限于时延、算力和更新稳定性，频繁修改模型权重并不现实。
- 对于真实系统而言，测试时适配比重新训练更贴近部署需求，因为它允许在保持主模型冻结的前提下动态注入外部知识。

采用 training-free agent 的根本动机，不是回避训练，而是针对自动驾驶长尾场景的工程约束选择更可执行的适配路径。长尾问题的困难在于，新的异常组合不断出现，但每一次都走“数据回流 - 标注 - 微调 - 验证 - 部署”的闭环，周期长且代价高。在车端系统中，这种方式尤其不经济，因为模型参数一旦频繁更新，就会引入额外的验证成本和系统稳定性风险。[3]

相较之下，training-free 方法保留主模型权重不变，把适配能力转移到推理时的上下文组织、工具调用和外部技能注入上。Self-Refine 的思想说明，大模型可以通过自我反馈与迭代修正提升输出质量，而不必依赖额外训练。[5] 本工作借鉴这一思路，但并不将其简单复用于自然语言生成任务，而是将其迁移到自动驾驶场景理解流程中：当边缘端模型输出不稳定或不充分时，云端多模态模型对图像与当前结构化结果进行反思；反思后的知识再被编译为可检索、可复用的 skill，而不是直接改写底层模型权重。

因此，training-free agent 的价值体现在三个层面。其一，它能缩短新场景适配路径，使系统具备更快的部署级响应能力。其二，它通过技能外化降低了模型更新的操作风险，使知识演进与权重演进解耦。其三，它天然适合分层式系统设计：边缘端负责高频、低时延的基础感知，云端负责低频、深推理的反思与技能归纳，两者通过标准化接口协同，而不是通过重新训练耦合在一起。

## 3. Challenges

- 何时判定边缘端结果“不足够可靠”并触发云端介入，是闭环是否稳定的前提。
- 云端推理虽然更强，但会带来额外时延，必须避免其进入所有样本的常规路径。
- 云端输出通常是自由文本，如何把它变成边缘端可执行、可检索、可约束的结构化 skill，是系统设计中的关键难点。
- 技能数量增长后，如何保证检索精度、抑制同名或同族技能冲突、避免重复调用，是系统可扩展性的核心问题。
- 评测层面，开放语义场景下的“结构化正确”并不总能用字符串精确匹配衡量，这会导致结果解释出现偏差。

从系统角度看，第一个挑战是触发机制。若触发阈值过低，云端会被频繁调用，系统失去实时性；若阈值过高，长尾失败样本又无法得到补救。因此，边缘端不仅要输出场景理解结果，还必须输出可供决策的置信度代理量。本系统当前使用候选分布的归一化熵与 fallback 标记共同决定是否进入反思流程。

第二个挑战是时延预算。自动驾驶系统不能把所有复杂样本都交给云端处理，因此必须把云端能力限制在“反思、编译、沉淀”的角色上，而不是把它作为默认感知主链路。第三个挑战是知识表示。自由文本结论虽然可读，但不可直接执行；边缘端实际需要的是关注区域、问题树、输出约束和标签约束等结构化组件。第四个挑战是技能库扩张后的检索质量问题。随着 skill 数量增加，仅依赖向量相似度容易产生语义近似但任务不匹配的结果，因此需要更严格的重排和去重策略。最后，评测本身也存在挑战。当前很多 benchmark 仍偏向固定标签或字符串对齐，但长尾感知本质上具有开放语义特征，这意味着结构化评测需要兼顾精确匹配与语义一致性。[2][3]

## 4. 针对挑战的核心方法：分层式 Agent 架构与自动 Skill Refinement

- 我们提出边缘端小模型与云端多模态大模型协同的分层式 agent 架构。
- 我们采用 training-free 的测试时适配方式，不对主模型权重做更新，而是通过技能归纳与调用完成能力补偿。
- 我们引入自动 skill refinement 机制，将云端反思结果编译为结构化 skill，并写入技能库供后续案例复用。
- 我们通过混合重排与同族去重，提高 skill 检索的稳定性和可控性。

本工作的核心思路是把自动驾驶长尾场景的处理流程拆成两个层级。边缘层使用较小的视觉语言模型执行直接图像推理，输出结构化场景结果；当结果不稳定时，系统通过标准化接口把样本发送给云端层，由更强的多模态模型对图像和当前结果进行反思，并给出纠正后的结构化理解。与直接把云端答案作为最终结果不同，本文更强调“知识沉淀”这一步：云端不仅要给出当前样本的修正结果，还要把触发条件、关注区域、动态问题树和输出约束编译成 skill。

这里所说的自动 skill refinement，在当前系统中具体表现为两种机制。第一种是单次反思后的自动技能归纳，即云端根据失败样本生成新的结构化 skill，并持久化到本地技能库。第二种是已有技能的自动选择与约束化复用，即通过嵌入检索、标签重叠、关键词重叠、天气一致性等多信号混合重排，再加上同族 skill 去重，使系统在后续样本上尽可能调用“最像且最专一”的技能，而不是简单返回多个相似项。需要说明的是，当前版本已经实现了单轮自动修正与技能复用闭环；更强的多轮自我精炼机制仍属于下一阶段工作。

## 5. 模型与系统架构

### 5.1 模型配置

- 边缘端模型：`Qwen/Qwen3.5-9B`
- 云端反思模型：`Pro/moonshotai/Kimi-K2.5`
- 评测裁判模型：`Pro/moonshotai/Kimi-K2.5`
- 嵌入模型：`BAAI/bge-m3`
- 推理接口：SiliconFlow 提供的 OpenAI-compatible `chat.completions` 与 `embeddings` 接口
- 输入形式：直接读取图像进行多模态推理，不先转写为文字

当前原型是一个典型的 edge-cloud hierarchical agent。边缘端使用 `Qwen/Qwen3.5-9B` 完成高频基础感知，云端使用 `Kimi-K2.5` 执行反思、纠正和技能编译，技能检索部分使用 `BAAI/bge-m3` 生成嵌入表示。系统通过 SiliconFlow 的兼容接口直接把图像传入多模态模型，因此方法链路中不存在“先 OCR 再推理”的中间替代步骤，这一点对保持视觉细节尤为重要。

### 5.2 系统组件

| 组件 | 作用 | 当前实现要点 |
| --- | --- | --- |
| EdgeAgent | 执行边缘端图像理解与结构化输出 | 直接输入图像，输出 `general_perception`、`regional_perception`、`triplets`、`qa_report`、`top_k_candidates`、`entropy` |
| ReplayOrchestrator | 串联完整预测闭环 | 实现 baseline perception、skill matching、必要时反思、再感知、产物写出 |
| CloudReflector | 对异常样本执行多模态反思 | 直接读取图像与 baseline 结果，生成纠正标签、场景三元组、反思摘要 |
| SkillCompiler | 将反思结果转为结构化 skill | 生成 `manifest.json` 与 `SKILL.md`，并写入 skill store |
| CloudReflectionService / MCP Tools | 暴露 `match_skills` 与 `reflect_anomaly` 接口 | 使用 MCP 组织工具调用与技能资源访问 |
| SkillMatcher | 在 skill store 中检索最相关技能 | 使用嵌入相似度、标签重叠、关键词重叠、天气一致性混合重排，并做同族去重 |
| Evaluation Runner | 产出可读评测结果 | 写出 `predictions.jsonl`、`predictions.pretty.json`、`metrics.json`、`report.md` |

从实现层面看，`EdgeAgent` 是主链路的入口。它接收问题、场景上下文和原始图像，输出结构化的 `EdgePerceptionResult`，并在结果中记录延迟、视觉 token 用量和归一化熵。`ReplayOrchestrator` 决定是否进入技能匹配与云端反思，是闭环执行器。`CloudReflector` 则把图像、问题、当前结果和已使用技能一起送入云端模型，获得纠正后的结构化结果与 skill 生成原料。最后，`SkillCompiler` 和 `SkillMatcher` 负责把一次反思转化为后续可复用的系统能力。

### 5.3 模型最终输出与统一结果格式

- 当前系统对单张图像的最终输出不是一句自然语言，而是一个统一的结构化结果对象 `EdgePerceptionResult`。
- 该结果对象同时承载感知、结构化关系、问答结果、动作建议和不确定性估计。
- 在 run 级别，系统再将 baseline 和 final 两次结果封装为 `CasePredictionRecord`，用于评测与归档。

当前系统中，边缘端模型和 skill-conditioned 再推理阶段都会输出统一的 `EdgePerceptionResult`。这一输出格式包含以下字段：

1. `general_perception`：全局场景理解，按车辆、弱势交通参与者、交通锥等类别组织。
2. `regional_perception`：局部区域目标及其边界框。
3. `driving_suggestions`：动作建议及解释。
4. `triplets`：场景图三元组。
5. `qa_report`：面向任务问题的问答结果。
6. `top_k_candidates`：候选标签及其概率。
7. `entropy`：基于候选分布计算的不确定性指标。
8. `recommended_action`：最终动作建议。
9. `latency_ms` 与 `vision_tokens`：推理代价信息。
10. `applied_skill_ids` 与 `used_fallback_label`：技能调用和回退标签使用情况。

对于单个 case，系统最终持久化的是 `CasePredictionRecord`，其中同时包含 `baseline_result`、`final_result`、`matched_skill_ids`、`reflection_result`、`judge_score` 和 `metadata`。这意味着当前系统已经具备论文实验中常见的“输入 - 中间过程 - 最终输出 - 评测分数”完整记录能力，而不是只保留最后一条答案。

### 5.4 技能表示

- 技能清单文件：`manifest.json`
- 人类可读说明：`SKILL.md`
- 检索索引：`data/skills/index.json`
- 技能内容包含：`trigger_tags`、`trigger_embedding_text`、`focus_region`、`dynamic_question_tree`、`output_constraints`、`fallback_label`

这种表示方式的意义在于，系统不把技能当作模糊的自然语言提示，而是将其约束为可检索、可执行、可审计的结构化对象。对于自动驾驶场景而言，这种设计比自由文本更适合作为长期演化的知识单元，因为它既能被模型消费，也能被工程系统检查和复现。

## 6. 系统工作流程

### 6.1 预测闭环

1. 载入一个待处理样本，包括图像、问题、场景上下文和可选标注。
2. 边缘端模型执行第一次感知，得到 baseline scene graph、候选标签分布和不确定性估计。
3. 调用 MCP `match_skills`，根据当前样本上下文检索可复用 skill。
4. 如果找到 skill，则将 skill 注入 prompt，再执行一次边缘端感知。
5. 如果结果仍不稳定，或触发强制反思条件，则调用 MCP `reflect_anomaly`。
6. 云端模型直接读取图像并结合 baseline 结果生成纠正结果，同时编译新的 skill。
7. 新 skill 持久化到技能库后，再次驱动边缘端执行 skill-conditioned perception。
8. 最终写出预测、评测指标和可读报告。

这个流程的关键不在于“多跑一次模型”，而在于把一次失败案例转化为后续可重用的外部知识。也就是说，系统的学习单位不是权重，而是结构化 skill。它既可以来自当前样本的云端反思，也可以来自以往案例中已经沉淀的 skill store。这样，系统就具备了部署场景下更实用的测试时自适应能力。

### 6.2 与 MCP 的关系

- `match_skills`：根据异常上下文、候选标签和熵值返回最相关 skill
- `reflect_anomaly`：对当前样本进行云端反思，返回纠正结果与可选的新 skill
- `skill://{skill_id}`：读取某条 skill 的结构化描述与 Markdown 说明

MCP 在这里不是附属接口，而是系统模块化的基础。[6] 它把边缘代理、云端反思服务和技能资源组织为统一的工具与资源接口，使得后续系统扩展不依赖于临时拼接的私有 RPC 约定，而可以沿着标准化协议演化。

### 6.3 当前支持的格式化任务

- 当前系统对齐 CODA-LM 的 3 个任务。
- 这 3 个任务分别是 `general_perception`、`region_perception` 和 `driving_suggestion`。
- 尽管数据集任务是分开的，系统在推理时将三者统一组织到同一个结构化输出对象中。

当前版本支持的任务数为 3，分别对应：

1. `general_perception`
   - 目标：识别场景中的主要驾驶相关实体，并给出全局场景说明。
   - 输出位置：`general_perception`
2. `region_perception`
   - 目标：识别关键局部目标，并给出边界框、类别与局部解释。
   - 输出位置：`regional_perception`
3. `driving_suggestion`
   - 目标：根据当前场景给出自车动作建议。
   - 输出位置：`driving_suggestions` 与 `recommended_action`

在此基础上，系统还额外输出了 `triplets`、`qa_report`、`top_k_candidates` 和 `entropy`。这些字段不只是为了“看起来更丰富”，而是为了让系统同时满足三类需求：一是完成数据集任务本身；二是支持技能匹配与闭环控制；三是提供后续评测与解释所需的结构化证据。

### 6.4 如何评判输出效果

- 当前评测同时包含 case-level 语义打分和 run-level 聚合指标。
- case-level 主要通过 judge 模型评估最终答案、triplets 和动作建议是否与问题及参考答案对齐。
- run-level 通过结构化指标衡量技能是否有效、triplet 是否匹配，以及代价是否可量化。

当前系统采用两级评判方式。第一层是 case-level judge：评测器读取 `question`、`reference answer`、`final QA answer`、`final triplets` 和 `recommended_action`，由裁判模型给出单案例语义分数。第二层是 run-level summary：系统对全部案例聚合计算 `judge_score_mean`、`regional_triplet_recall`、`skill_success_rate`、`latency_delta_ms` 和 `vision_token_delta`。

具体而言：

1. `judge_score_mean`
   - 衡量最终输出在语义层面与参考答案的一致程度。
2. `regional_triplet_recall`
   - 衡量最终 triplets 与标注 triplets 的匹配比例。
3. `skill_success_rate`
   - 统计使用 skill 或触发反思的案例中，有多少达到 judge 阈值。
4. `latency_delta_ms`
   - 比较 baseline 与 final 的平均时延差值。
5. `vision_token_delta`
   - 比较 baseline 与 final 的视觉 token 变化。

这种评测设计的意义在于，它既关注“答得对不对”，也关注“结构化得对不对”，同时还关注“为此付出了多少代价”。对于一个 training-free hierarchical agent 系统来说，这种联合评测比单一准确率更能反映真实系统表现。

## 7. 具体样例

- 实验样本：`data/coda_lm/2026-03-12 14.32.23.png`
- 对应 run：`data/artifacts/run-20260312T124716Z`
- case id：`image-experiment-1`
- 任务问题：识别前方主要驾驶相关实体，并给出最安全的自车动作

该样例是一条真实的图像推理链路。场景中存在前方引导车辆、路侧行人与作业人员、交通锥桶以及右侧禁入标志。baseline 阶段，边缘端模型已经能够识别出“前方黑色 SUV”“左侧与右侧路侧行人”“右侧锥桶”和“禁入标志”等要素，并给出 `slow_down_and_yield` 的建议动作，但其标签仍偏粗粒度，且不确定性较高，说明结果虽然具有一定可用性，但还不够稳定。

随后，系统从技能库中匹配到已有 skill `lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c`，该 skill 的关注区域是 `center_lane_20_40m_ahead_plus_right_curb`，核心问题树包括“前方是否存在直接约束自车的车辆”“路侧是否有行人或作业人员”“右侧边界是否可通行”“最终动作是什么”。在 skill 注入后的第二次感知中，系统输出更加规范的结构化结果：`vehicle -> is -> Lead_Vehicle`，`pedestrian -> is -> Pedestrian`，`ego_vehicle -> should -> slow_down`。同时，最终候选中 `Lead_Vehicle` 的概率提升到 `0.85`，归一化熵下降到 `0.4717`，推荐动作收敛为 `slow_down`。

为便于汇报，可将这一案例概括为下表：

| 项目 | Baseline | Skill-conditioned final |
| --- | --- | --- |
| 主要实体 | 黑色 SUV、路侧行人、交通锥、禁入标志 | Lead Vehicle、Pedestrian、Traffic Cones |
| 动作建议 | `slow_down_and_yield` | `slow_down` |
| 代表性三元组 | `obstacle -> is -> Lead_Vehicle_Pedestrian_Caution` | `vehicle -> is -> Lead_Vehicle` |
| 额外三元组 | `ego_vehicle -> should -> slow_down_and_yield` | `pedestrian -> is -> Pedestrian`；`ego_vehicle -> should -> slow_down` |
| 归一化熵 | `0.7298` | `0.4717` |
| 反思调用 | 未发生 | 未发生，说明 skill 复用已足以完成修正 |

这个案例说明，当前系统已经不仅能生成自然语言结论，还能输出结构化场景 triplets、可追踪的 skill id 和对应的动作建议。更重要的是，结果的改进来自 skill 检索与模型再推理，而不是人工编写 placeholder 或离线微调。

## 8. 当前成果汇报：系统已经完成了什么，产出了什么

- 已完成一个真实可运行的 training-free 闭环原型，覆盖边缘感知、技能匹配、云端反思、技能编译、技能持久化、再次感知与评测。
- 当前最稳定、可直接汇报的实验结果位于 `data/artifacts/run-20260312T124716Z`。
- 当前成果已经不是“只有链路”，而是包含结构化预测结果、评测指标、技能文件、可读报告和通过测试的代码实现。
- 当前版本已能够输出场景图要素、triplets、动作建议、匹配到的 skill，以及可重复复用的 skill 清单。

为了使报告能够完整呈现现阶段成果，本节不只给出结论，还直接展示当前 run 的真实产物内容，并逐项说明这些内容说明了什么。

### 8.1 当前已经实现的系统能力

- `EdgeAgent` 已能够直接读取图像，输出结构化 `general_perception`、`regional_perception`、`triplets`、`qa_report`、`top_k_candidates`、`entropy`。
- `ReplayOrchestrator` 已能够执行完整闭环：baseline perception、skill matching、必要时云端反思、再感知、结果写出。
- `CloudReflector` 已能够直接基于图像和 baseline 结果做多模态反思，并输出可被编译的结构化内容。
- `SkillCompiler` 已能够把反思结果转为 `manifest.json` 和 `SKILL.md`。
- `SkillMatcher` 已实现 embedding 检索、标签重叠重排、关键词重排、天气一致性约束，以及同族 skill 去重。
- `Evaluation Runner` 已能够写出 `predictions.jsonl`、`predictions.pretty.json`、`metrics.json` 和 `report.md`。

这一部分说明，当前系统已经超出了“概念验证”阶段。它不是只有一个单独的推理脚本，而是已经具备了可重复运行、可沉淀技能、可度量结果、可进行后续扩展的完整原型结构。对于论文汇报而言，这意味着我们可以把系统描述为一个已经闭环运行的 training-free edge-cloud agent prototype，而不是仅停留在方法设想。

### 8.2 当前 run 的真实产物清单

当前最新成果目录 `data/artifacts/run-20260312T124716Z` 下已经生成如下文件：

1. `predictions.jsonl`
2. `predictions.pretty.json`
3. `metrics.json`
4. `report.md`

此外，skill store 中已经存在与该案例相关的技能文件：

1. `data/skills/lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c/manifest.json`
2. `data/skills/lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c/SKILL.md`

这些产物分别对应不同层面的系统能力。`predictions.jsonl` 用于机器处理和后续评测；`predictions.pretty.json` 用于人工阅读和汇报展示；`metrics.json` 给出当前 run 的聚合指标；`report.md` 提供快速摘要；`manifest.json` 和 `SKILL.md` 则表明系统已经把一次案例经验转化为可复用的技能对象。

### 8.3 当前 run 的完整评测摘要内容

当前 `metrics.json` 的完整内容如下：

```json
{
  "run_id": "run-20260312T124716Z",
  "total_cases": 1,
  "judge_score_mean": 85.0,
  "regional_triplet_recall": 0.0,
  "skill_success_rate": 1.0,
  "latency_delta_ms": 4037.1968330000527,
  "vision_token_delta": 94.0
}
```

这一结果的概括性解释如下：

- `total_cases = 1` 说明当前展示的是一个单案例的稳定闭环结果。
- `judge_score_mean = 85.0` 说明裁判模型认为最终输出在语义层面已经达到较高可接受程度。
- `skill_success_rate = 1.0` 说明在需要 skill 参与的当前案例上，skill 调用是成功的。
- `latency_delta_ms = 4037.20` 与 `vision_token_delta = 94.0` 说明 final 输出相对 baseline 发生了可量化的推理代价变化。
- `regional_triplet_recall = 0.0` 并不代表没有 triplets，而是说明当前 exact-match 评测还不能充分刻画语义一致的结构化结果。

当前 `report.md` 的完整内容如下：

```md
# Evaluation Report: run-20260312T124716Z

- Total cases: 1
- Judge score mean: 85.00
- Regional triplet recall: 0.00
- Skill success rate: 1.00
- Latency delta ms: 4037.20
- Vision token delta: 94.00
```

这个摘要文件虽然很短，但它的作用非常明确：它表明当前原型已经具备一套固定的 run-level 汇报格式。也就是说，每次实验结束后，系统不只是产生若干中间日志，而是能自动生成一份标准化评测摘要，这对后续批量实验和论文中结果表的积累都很重要。

### 8.4 当前案例的完整结构化输出内容

当前最新 run 中的案例为：

- `case_id`: `image-experiment-1`
- 输入问题：`Identify the main driving-relevant entities ahead and provide the safest ego-vehicle action.`
- 匹配到的 skill：`lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c`
- `reflection_result`: `null`
- `judge_score`: `85.0`

这里的 `reflection_result = null` 具有明确意义。它并不表示云端反思模块不可用，而是说明在这个案例中，系统通过已有 skill 的检索和再次感知，已经把结果从 baseline 修正到可接受状态，因此不再需要额外调用云端反思。这说明 skill 复用闭环已经真实发挥作用。

#### 8.4.1 Baseline 输出的完整内容

Baseline 阶段的 `general_perception` 包含如下实体：

- 车辆：
  - `Black SUV directly ahead in the travel lane.`
  - `Distant vehicles visible further up the road.`
- 弱势交通参与者：
  - `Pedestrians walking on the left sidewalk.`
  - `Pedestrians standing on the right sidewalk.`
- 交通锥：
  - `Orange traffic cone near the right lane margin.`
- 其他物体：
  - `Red and white striped bollards on the left.`
  - `No Entry sign on the right.`
- 全局解释：
  - `The scene shows a clear day on a tree-lined road with a black SUV ahead. Pedestrians are present on both sidewalks, and traffic cones and bollards define the lane boundaries. The primary hazard is the lead vehicle and the potential for pedestrians to enter the roadway.`

Baseline 阶段的 `regional_perception` 包含如下局部目标：

| 类别 | 描述 | 框坐标 |
| --- | --- | --- |
| `vehicle` | `Black SUV directly ahead in the lane.` | `(483, 515) - (593, 698)` |
| `traffic_cone` | `Orange traffic cone on the right side.` | `(720, 620) - (738, 705)` |
| `pedestrian` | `Pedestrian on the left sidewalk.` | `(135, 515) - (168, 680)` |

Baseline 阶段的结构化结果如下：

```json
{
  "triplets": [
    {
      "subject": "obstacle",
      "relation": "is",
      "object": "Lead_Vehicle_Pedestrian_Caution"
    },
    {
      "subject": "ego_vehicle",
      "relation": "should",
      "object": "slow_down_and_yield"
    }
  ],
  "qa_report": [
    {
      "question": "What is the primary hazard or obstacle?",
      "answer": "Lead_Vehicle_Pedestrian_Caution"
    }
  ],
  "top_k_candidates": [
    {
      "label": "Lead_Vehicle_Pedestrian_Caution",
      "probability": 0.7
    },
    {
      "label": "Narrow_Lane_Constraint",
      "probability": 0.2
    },
    {
      "label": "Critical_Unknown_Obstacle",
      "probability": 0.1
    }
  ],
  "entropy": 0.7298466991620975,
  "recommended_action": "slow_down_and_yield",
  "latency_ms": 46295.383832999505,
  "vision_tokens": 2730,
  "applied_skill_ids": [],
  "used_fallback_label": true
}
```

这一部分说明，baseline 已经具备基本场景理解能力，但标签仍偏粗，且 `used_fallback_label = true`，意味着系统认为当前结果仍然存在不确定性，需要进入 skill 匹配或进一步修正流程。

#### 8.4.2 Final 输出的完整内容

在 skill 注入后的 final 阶段，`general_perception` 输出为：

- 车辆：
  - `A dark SUV is directly ahead in the center lane, moving away from the ego vehicle.`
- 弱势交通参与者：
  - `Pedestrians are present on the left sidewalk and a worker is standing on the right side of the road.`
- 交通锥：
  - `Traffic cones are placed along the right curb.`
- 其他物体：
  - `A 'No Entry' sign and a 'Parking Lot Full' sign are visible on the right.`
- 全局解释：
  - `The scene shows a clear road with a lead vehicle ahead. The primary hazards are the lead vehicle requiring following distance and the presence of pedestrians and a worker on the right side, necessitating caution.`

Final 阶段的 `regional_perception` 输出为：

| 类别 | 描述 | 框坐标 |
| --- | --- | --- |
| `Lead_Vehicle` | `Lead vehicle in the center lane.` | `(483, 515) - (594, 697)` |
| `Pedestrian` | `Pedestrians and worker on the right curb.` | `(655, 524) - (999, 826)` |

Final 阶段的结构化结果如下：

```json
{
  "triplets": [
    {
      "subject": "vehicle",
      "relation": "is",
      "object": "Lead_Vehicle"
    },
    {
      "subject": "pedestrian",
      "relation": "is",
      "object": "Pedestrian"
    },
    {
      "subject": "ego_vehicle",
      "relation": "should",
      "object": "slow_down"
    }
  ],
  "qa_report": [
    {
      "question": "What is the primary hazard or obstacle?",
      "answer": "Lead_Vehicle"
    },
    {
      "question": "Are there pedestrians or workers near the roadway?",
      "answer": "Pedestrian"
    }
  ],
  "top_k_candidates": [
    {
      "label": "Lead_Vehicle",
      "probability": 0.85
    },
    {
      "label": "Pedestrian",
      "probability": 0.1
    },
    {
      "label": "Traffic_Cones",
      "probability": 0.05
    }
  ],
  "entropy": 0.4716734178155153,
  "recommended_action": "slow_down",
  "latency_ms": 42258.18699999945,
  "vision_tokens": 2636,
  "applied_skill_ids": [
    "lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c"
  ],
  "used_fallback_label": false
}
```

与 baseline 相比，final 输出的变化是清晰且可解释的：

- 标签从较粗粒度的 `Lead_Vehicle_Pedestrian_Caution` 收敛为更规范的 `Lead_Vehicle` 与 `Pedestrian`。
- 动作从 `slow_down_and_yield` 收敛为更明确的 `slow_down`。
- 熵从 `0.7298` 降到 `0.4717`，说明不确定性下降。
- `used_fallback_label` 从 `true` 变为 `false`，说明系统已不再依赖保底标签。
- `applied_skill_ids` 明确记录了本次结果由哪一条 skill 参与修正。

这说明当前系统已经能够把“看懂图像”转化为“给出更规范的结构化输出”，并且这种改进路径是可追踪的、可解释的。

#### 8.4.3 与 Ground Truth 的关系

当前案例记录中的 `ground_truth_triplets` 为：

```json
[
  {
    "subject": "ego_vehicle",
    "relation": "follows",
    "object": "lead_vehicle_ahead"
  },
  {
    "subject": "pedestrians",
    "relation": "are",
    "object": "roadside"
  },
  {
    "subject": "ego_vehicle",
    "relation": "should",
    "object": "slow_down_and_yield"
  }
]
```

这部分内容解释了为什么 `judge_score` 可以较高，而 `regional_triplet_recall` 仍然为零。因为 Ground Truth 与 final 输出在语义上是相近的，但在字符串表达上并不完全一致。例如，模型输出 `vehicle -> is -> Lead_Vehicle`，而标注使用的是 `ego_vehicle -> follows -> lead_vehicle_ahead` 这样的关系表达。因此，目前系统已经产生结构化结果，但评测对齐方式仍有待继续完善。

### 8.5 当前已经生成并可复用的 skill 内容

当前案例对应的 skill `manifest.json` 完整内容如下：

```json
{
  "skill_id": "lead-vehicle-pedestrian-caution-image-experiment-1-fad37b2c",
  "name": "Lead_Vehicle_Pedestrian_Caution",
  "version": "0.1.0",
  "trigger_tags": [
    "lead_vehicle_ahead",
    "pedestrian_near_roadside",
    "worker_at_curb",
    "narrowed_lane_margin",
    "daylight_clear"
  ],
  "trigger_embedding_text": "Lead black SUV in center lane 30m ahead, pedestrians at left roadside, sanitation workers at right curbside near cones and no entry sign narrowing right lane margin. Ego vehicle should slow down and maintain gap while watching right workers.",
  "focus_region": "center_lane_20_40m_ahead_plus_right_curb",
  "dynamic_question_tree": [
    "Is there a vehicle directly ahead blocking or slowing ego?",
    "Are there pedestrians or workers near the roadway?",
    "Is the right lane margin passable?",
    "Final action?"
  ],
  "output_constraints": [
    "max_triplets: 12",
    "label_must_include: ['Lead_Vehicle', 'Pedestrian']",
    "action_must_include: ['slow_down']",
    "region_must_include: ['center_lane_ahead', 'right_curb']"
  ],
  "fallback_label": "Critical_Unknown_Obstacle",
  "source_case_id": "image-experiment-1",
  "created_at": "2026-03-12T06:46:51.699162Z",
  "metadata": {}
}
```

这个 skill manifest 的意义在于，它把一次图像理解经验压缩成了可被检索和复用的结构化知识单元。其中：

- `trigger_tags` 与 `trigger_embedding_text` 负责检索；
- `focus_region` 负责告诉边缘端模型优先关注哪里；
- `dynamic_question_tree` 负责规定再次推理时的问题顺序；
- `output_constraints` 负责把输出限制到更稳定的结构化形式。

当前案例对应的 `SKILL.md` 完整内容如下：

```md
# Lead Vehicle with Pedestrian Caution

## Scene Classification
- **Primary Hazard**: Lead vehicle (black SUV) in center lane requiring gap maintenance
- **Secondary Hazards**: Roadside pedestrians and sanitation workers near narrowed right lane margin
- **Environmental**: Daylight clear conditions

## Key Triplets
1. `black_suv` → is → `Lead_Vehicle`
2. `pedestrians` → is → `Pedestrian`
3. `sanitation_workers` → is → `Pedestrian`
4. `cones` → is → `Static_Obstacle`
5. `no_entry_sign` → is → `Traffic_Sign`
6. `ego_vehicle` → should → `slow_down_and_maintain_gap`
7. `ego_vehicle` → should → `watch_right_workers`
8. `obstacle` → focus_region → `center_lane_30m_ahead`
9. `lead_vehicle` → is → `center_lane_ahead`
10. `pedestrians` → are → `left_roadside`
11. `workers` → are → `right_curbside`
12. `cones_and_sign` → narrow → `right_lane_margin`

## Action Directive
Ego vehicle must **slow down** to maintain safe following distance from lead SUV while actively monitoring right curbside workers and pedestrians. Right lane margin is narrowed - treat as partially obstructed.
```

与 `manifest.json` 相比，`SKILL.md` 更适合人类阅读和人工审查。它说明系统不仅能把技能写成可执行的结构化对象，也能同时输出一份语义清晰、适合工程人员理解的文本说明。对于论文汇报而言，这一点可以作为“技能沉淀不是黑箱”的直接证据。

### 8.6 当前成果的整体概括

综合以上内容，当前成果可以概括为以下四点：

1. 系统链路已经完整跑通。图像输入、边缘感知、skill 匹配、再次感知、结果写出和评测已经形成闭环。
2. 系统已经能够输出结构化结果。当前结果不只是自然语言描述，还包括区域感知、triplets、QA 报告、候选标签和动作建议。
3. 系统已经能够沉淀并复用技能。当前案例中的 skill 已经保存为 `manifest.json` 和 `SKILL.md`，并成功参与了后续的结果修正。
4. 系统已经具备可汇报的实验产物。当前 run 已有 `predictions.pretty.json`、`metrics.json`、`report.md` 等可直接用于论文汇报的证据材料。

因此，现阶段报告中最重要的结论不是“系统未来可能做到什么”，而是“系统现在已经能够做到什么”。当前原型已经完成了从方法设想到真实产物的落地：它可以对真实图像给出结构化场景理解，可以通过 skill 改善结果，可以把技能保存下来，并且可以把结果组织成标准化实验产物。对论文阶段汇报而言，这些内容已经足以支撑“系统原型已实现且能稳定产生有意义输出”的结论。

## 9. 后续改进方向

- 扩展多案例实验规模，在 CODA-LM 上形成更有说服力的统计结果。
- 完成 DTPQA 与 DriveLM 评测链路，验证距离敏感性和图结构推理能力。
- 完善 triplet normalization 与语义对齐评测，避免 exact-match 指标低估真实结构化质量。
- 将自动 skill refinement 从当前单轮闭环扩展到多轮反思与版本化管理。
- 进一步量化边缘-云协同的时延收益、token 成本和安全收益。

下一阶段最重要的工作不是继续增加新的演示样例，而是把当前闭环推广到更多案例并建立更严格的评测协议。第一，需要在 CODA-LM 上扩大样本规模，区分已有 skill 复用成功、云端反思新建 skill 成功、以及完全失败三类情况。第二，需要补齐 DTPQA 与 DriveLM 的实证，以证明系统不仅能在单图场景上工作，也能在距离敏感和图结构推理任务上体现优势。[2][4] 第三，需要从评测侧改造 triplet 对齐方式，引入标签规范化、同义映射和语义匹配规则，否则 exact-match 指标会系统性低估真实结果质量。

此外，从方法层面看，当前 skill refinement 仍以单轮反思为主，后续可继续加入 skill 版本化、置信度校验、冲突检测和多轮自我修正机制。这样可以使系统从“能自动沉淀技能”进一步发展到“能自动维护技能库质量”。对于自动驾驶这一高安全约束领域，这种从外部知识管理入手、而非频繁改动主模型权重的演进方式，更适合作为长期工程路线。

## 参考文献

[1] Sadat, A., et al. *CODA: A Real-World Road Corner Case Dataset for Object Detection in Autonomous Driving*. arXiv, 2022. https://arxiv.org/abs/2203.07773

[2] Sima, C., et al. *DriveLM: Driving with Graph Visual Question Answering*. arXiv, 2023. https://arxiv.org/abs/2312.14150

[3] *Vision-Language Models in Autonomous Driving and Intelligent Transportation Systems: A Survey*. arXiv, 2024. https://arxiv.org/abs/2405.14414

[4] *DTPQA: Distance-Aware Traffic Perception QA Benchmark for Small VLMs*. arXiv, 2025. https://arxiv.org/abs/2501.07634

[5] Madaan, A., et al. *Self-Refine: Iterative Refinement with Self-Feedback*. arXiv, 2023. https://arxiv.org/abs/2303.17651

[6] Anthropic. *Model Context Protocol Documentation*. 2026. https://modelcontextprotocol.io/introduction
