# 一、痛点（整体陈述）

在 ASVspoof5 条件下，评测集引入多样化的编解码与带宽设定，并与不同深伪生成方式交织，形成显著的分布偏移。常规判别范式主要在“整体相似度/可疑度”层面建模，却缺少对**语音的内容（语言/共振峰）与声门激励（基频、相位、谐噪结构）之间物理耦合关系**的显式检验。当语音经过不同 codec、重采样或转码链路时，这种耦合在自然语音中具有相对稳定性；而在深伪或加工不当的信号中，耦合常被破坏。缺乏对这一“**内容–激励一致性**”的直接建模，会放大“攻击×codec”的联合作用，使得开发集与评测集之间的指标落差难以收敛，同时也削弱了分数的可解释性与可校准性。

## 痛点（细分要点）

**P1 评测分布偏移的核心来自“攻击×codec”的联合作用。**
不同生成方式与窄带/强压缩/转码链路叠加后，统计特性产生非线性扭曲，导致开发集选择的决策边界在评测集失效。

**P2 缺乏“内容–激励耦合”的显式检验。**
自然语音可由源–滤波器模型刻画：在给定内容（滤波器/共振峰）条件下，激励（基频与相位/谐噪结构）的统计分布存在稳定形态。深伪链路往往在该耦合上留下缺口，但常规得分未把“耦合是否成立”作为一条独立判别轴。

**P3 增广覆盖的上限与未见 codec 的不可预知性。**
仅依赖经验性增广难以确保对“未见 codec/带宽”的泛化；缺少“同一内容在不同 codec 下应等价”这一结构性先验。

**P4 可解释性与可校准性不足。**
分数难以定位到具体时刻与具体失配类型，导致 actDCF 与 minDCF 存在显著落差，且难以针对性修正。

**P5 工程简单性与复现性要求。**
追求鲁棒性不应以高复杂度和多模型堆叠为代价；需要一条**统一、轻量、可复现**的路线，以便稳定部署与严谨验证。

---

# 二、创新性（整体陈述）

提出**C²S：内容–激励一致性评分（Content–Excitation Coherence Scoring）**。该方案以源–滤波器视角解构语音：以**内容表征 $C$** 作为锚点，以**激励特征 $E$** 描述声门与相位等源端属性，在**仅基于 bona fide** 的样本上学习“**自然语音的条件分布 $p(E\mid C)$**”，并在测试时以**负对数似然（NLL）**度量“观测激励是否与内容相容”。同时，将不同 codec/带宽/转码视作“**同一内容的等价观测**”，在学习阶段以多视图的方式融入 $p(E\mid C)$，使一致性评分天然具备跨 codec 的不变性。最终在**有声帧门控**与**鲁棒统计汇聚**下得到句级分数，实现对“攻击×codec”条件的稳定判别，并提供可分解到时间轴与特征维度的**可解释与可校准**证据。方案从原理到实现保持单一主题：**以一致性先验为核心、以等价观测为支撑、以统计评分为落地**。

## 创新性（细分要点）

**C1 一致性先验：显式建模自然语音的 $p(E\mid C)$。**

* **内容 $C$**：时间对齐的内容表征（例如来自允许条件下的 SSL 中层或等价的内容特征），反映音素/共振峰层面的信息。
* **激励 $E$**：从一次 STFT/相位张量派生的低维、可解释特征（如 $\log F0$/voicing、抖动/颤动、HNR、谱倾斜、改进组延迟统计、LPC 残差信号能量与平坦度等）。
* **条件密度**：在内容空间做离散化（聚类成 $K$ 个子区），针对每个子区拟合**线性–高斯族** $E=A_k C + b_k + \varepsilon, \ \varepsilon\sim\mathcal N(0,\Sigma_k)$，协方差采用对角形式并作稳健下界处理。
  → 结果：把“内容–激励耦合”转化为**可计算的条件似然**，为后续评分提供物理与统计上的统一依据。

**C2 等价观测：把 codec/带宽当作“同一内容的多视图”。**

* 对同一条自然语音生成若干 codec/采样率视图，仅重算 $E$（内容锚 $C$ 保持一致）；
* 在拟合 $p(E\mid C)$ 时合并多视图样本，**不引入新损失**，仅以最大似然统一学习；
  → 结果：在学习阶段注入“跨 codec 不变”的结构性先验，使一致性评分对未见 codec/转码链具备外推能力，从根源抑制“攻击×codec”的联合作用。

**C3 评分与汇聚：voiced-gated 的 NLL 与鲁棒统计。**

* **帧级评分**：仅在有声帧计算 $\text{NLL}_t$，避免静音/噪声干扰：

  $$
    \text{NLL}_t=\tfrac12\sum_j\Big(\frac{E_{t,j}-(A_kC_t+b_k)_j}{\sigma_{k,j}}\Big)^2
                  +\tfrac12\sum_j\log\sigma_{k,j}^2 .
  $$
* **窗级/句级汇聚**：对固定时长窗做截尾均值或逆方差加权，再做句级平均；可配合简单的 z 归一化与逻辑回归校准。
* **解释与诊断**：$\text{NLL}_t$ 可视化为时间热图，并按特征分组定位“不一致”的来源（基频相关、相位相关、残差相关等）。
  → 结果：得到**稳定可校准**的句级分数，并能对异常区间给出可读解释。

**C4 面向痛点的直接对齐（P1–P5 ↔ C1–C3 的映射）。**

* **应对 P1/P3（分布偏移与未见 codec）**：等价观测使 $p(E\mid C)$ 学得“跨 codec 不变”的映射；评分对“攻击×codec”联合作用不敏感。
* **应对 P2（缺乏耦合检验）**：一致性先验把“是否合乎源–滤波器规律”上升为**一条独立判别轴**。
* **应对 P4（解释与校准）**：NLL 可分解到时间与特征维度，便于定位失配与进行操作点校准。
* **应对 P5（工程与复现）**：一次 STFT、向量化特征、线性–高斯拟合与对角协方差的评分，**计算与实现均轻量**，便于复现与部署。

**C5 统一叙事的完整性。**
所有组件——内容锚、激励特征、等价观测、条件密度、voiced 门控与鲁棒汇聚——均服务于**同一主题**：在统计与物理两层面**确立并检验“内容–激励一致性”**。这些环节并非相互独立的技巧集合，而是从“先验 → 学习 → 评分 → 解释/校准”的单线逻辑顺次展开，形成自洽的整体方案。

---

**一句话概括**
C²S 以“自然语音的内容–激励耦合”为统一先验，借助等价观测学习得到跨 codec 稳定的 $p(E\mid C)$，并以有声门控的 NLL 进行时–句两级评分；在不增加系统复杂度的前提下，直接针对“攻击×codec”的核心失配给出可解释、可校准的判别信号。


# 实现总体思路

- 先把 C²S 做成与现有训练解耦的“离线管线”（提E→拟合→打分→校准），不改你主训练循环；待验证收益后，再决定是否在训练期
联动读取 E。
- 所有实现按 v1.1 文档：20ms 对齐、ragged memmap + index.jsonl、自描述元数据、同分片聚合读取。

TODO 阶段 0（准备）

- 配置开关: 在 config/config.py 加 c2s 命名空间（enable/components/use_voiced_only/use_buckets/agg/calibration 等，
见文档“极简消融”示例）。
- 目录约定: data/ASVspoof5/features/E/{train,dev}、models/c2s/、results/c2s/。
- 验收清单: 20ms 对齐检查（E.stride_ms==C.stride_ms）、index.jsonl 字段一致、设备 I/O 风险提示记录。

TODO 阶段 1（E/V 离线提取与写盘）

- 新模块 src/features/excitation.py:
    - GPU STFT 一次（n_fft=512, win=400, hop=320, center=True）；由该 STFT/相位派生全部 E 分量；
    - 组合 E=32 维（logF0/voicing/jitter/vibrato/HNR/谱倾斜/组延迟/LPC 残差等，见文档“维度建议”）。
    - 数值稳健性: logF0（unvoiced=0）、逐维 mean/std 统计、异常帧裁剪策略占位。
- 写盘器 src/data/e_writer.py:
    - ragged memmap 分片写入 E（float16）与 V（uint8），生成 index.jsonl；
    - 元数据: {utt_id, shard, offset_elems, elem_count, T, D, L=1, sr=16000, win_ms=25, stride_ms=20, dtype,
layout:“TD”, version:“E.v1.1”}；
    - 统计输出: 总样本数/总元素/平均帧数/总 GiB。
- 脚本 scripts/c2s_extract_E.py:
    - 对 train/dev 批量提取，默认仅 train/dev（eval 可后续在线）；支持 --test 小样本。

TODO 阶段 2（读取与 I/O 聚合）

- src/data/feature_loader.py 扩展或新建 EFeatureLoader:
    - 复用 ragged memmap 读取逻辑，支持 E 与 V 的 index；
    - 返回 (E[T, D], V[T])。
- DataLoader/Collate:
    - 保持主训练不受影响；新建 src/data/dataloader_c2s.py 仅用于 C²S 打分；
TODO 阶段 3（内容桶与条件模型）

- 桶化拟合:
    - scripts/c2s_fit_buckets.py: IncrementalPCA(64) + MiniBatchKMeans(K=128/200)，对 train-bonafide 抽样 20–30%
帧；保存随机种与模型到 models/c2s/；
    - src/models/c2s_bucket.py: assign_bucket(C_288)（PCA→KMeans）。
- 线性-高斯条件:
    - src/models/c2s_model.py: per-bucket Ridge 回归（E = A_k*C64 + b_k），对角方差估计（σ²≥1e-4）；序列化到 models/
c2s/linGauss_k*.pkl。

TODO 阶段 4（打分与校准）

- scripts/c2s_score.py:
    - 对 dev/progress 逐句打分：帧级 NLL（voiced-only）、窗内聚合（trimmed/mean）、句级平均，输出 results/c2s/
score_c2s.jsonl；
    - 规则: 无声>80% 的窗降权/跳过（按文档）。
- scripts/c2s_calibrate.py:
    - BOSARIS 逻辑回归校准（dev-cal），输出 C2S_cal 分数 + 指标（minDCF/EER/CLLR/actDCF）。
- 可选 scripts/c2s_fuse.py:
    - 与 AASIST3 融合（仅对照，不作为主线）。

TODO 阶段 5（极简消融 S0–S5）

- 加最小开关并驱动脚本/配置:
    - S0 基线（all/voiced/buckets/trimmed/calib）
    - S1 去 voiced
    - S2 去分桶
    - S3 聚合 mean
    - S4 仅 F0+HNR
    - S5 F0+HNR+Slope
- 结果归档:
    - 统一命名 C2S_S{ID}_<key>=<val>；汇总成一张表（minDCF/EER/CLLR/actDCF、吞吐、读速、磁盘 GiB）。

TODO 阶段 6（文档/诊断/回滚）

- 文档: 在 C²S.md 添加“结果表模板/命名规范/指标定义与工具链”；
- 诊断: per-bucket NLL 均值/方差与帧占比的 quick report 脚本；
- 回滚: 若 I/O 抖动，先停多视图/仅跑单视图；必要时将 target_frames 降到 320/256。

先做哪些（不占 GPU/当前不执行）

- 配置与接口占位（阶段 0）
- 模块/脚本骨架（阶段 1–4 的文件与函数签名、docstring、索引/元数据字段定义）
- 结果表模板与命名规范（阶段 6）
- 增补 C²S.md 的“运行指南”小节