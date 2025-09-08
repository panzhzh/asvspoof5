太好了—在已稳定的 AASIST/AASIST3 基线上，下面给出 C²S 最终版方案（v1.1），对齐你当前的数据形态与硬件约束，重点优化存储与 I/O：

- 与现有内容特征 C 完全对齐（stride=20 ms）；
- E 采用与 C 相同的“分片 ragged memmap + index.jsonl”格式（避免海量小文件）；
- 给出可复现实测的磁盘/吞吐预算与回滚点。

---

# 方案总览（v1.1）

前提与约束
- C 的时间轴：由你当前特征决定（已确认 index.jsonl: `stride_ms: 20`），D=288。
- NVMe 在 WSL2 下顺序带宽有限，随机小块读会导致 GPU/CPU 低利用；因此 E 必须用与 C 相同的分片内存映射（ragged memmap）组织，并在 DataLoader/Collate 聚合连续段，尽量顺序读。
- Track 1（closed）不使用外部预训练；本方案所有“平坦化优化/校准/增强”均不依赖外部语料。

落地目标
- 离线缓存 E（FP16）+ voiced mask，与 C 同步对齐 → 训练/推理由缓存读取。
- 仅用 bonafide 拟合 p(E|C)，推理以 NLL 做窗/句级评分，最后做 BOSARIS 逻辑回归校准；可选与 AASIST3 融合。

磁盘/IO 预算（关键数字）
- 若 E 采用 20 ms 对齐、32 维 FP16：
  - 每小时 ≈ 50 fps × 3600 × (2×32 + 1) bytes ≈ 11.7 MB/h（含 1-byte voiced mask），全量 4242 h ≈ 49–50 GiB（与 C 同分片、顺序写）；
  - 多视图 E（仅 train-bonafide）每视图 ≈ 2–5 GiB（取决于训练总时长、平均切片长度）。
- 训练阶段 I/O 字节量增加 ≈ 10–12%（E 相对 C），但采用分片 memmap + 批内顺序聚合后，SSD 队列深度与 readahead 能发挥作用，整体吞吐保持稳定。

回滚点
- 若 E 的读取造成吞吐显著下降，先行仅跑“单视图 E + 线性高斯模型”，暂不落盘多视图；或将 `target_frames`/`io_train_layers` 降低，先控 I/O 体量。

---

# 任务清单（按重要性 & 依赖关系排序）

## 1)（独立）离线提取并缓存激励特征 E（GPU-first）

**目标**：一次性为所有音频生成帧级 E 与 voiced mask，FP16 存盘，后续所有训练/推理只读缓存。
**输入**：原始 `wav`（16 kHz）、你已有的 `C` 帧特征（用于对齐参数参考，不参与本步计算）。
**统一时轴**：与 C 保持一致（你当前为 **20 ms stride**；win 可用 25 ms）。若 C 未来改动，E 必须同步修改。
**GPU-first 实现要点**

* 单次 `torch.stft(..., n_fft=512, win_length=400@16k, hop_length=160, return_complex=True, device='cuda')`；
* 从 STFT/相位派生全部特征（**不**用 CPU 的逐帧 `pyin`）：

  * **F0/voicing**（倒谱/频域自相关选峰，周期范围 80–500 Hz）；
  * **jitter/vibrato**（对 F0 序列 1D 卷积求 Δ与 4–8 Hz 带通幅度/速率）；
  * **HNR**（谐波/残差能量比，dB）；
  * **谱倾斜**（0.3–3 kHz 线性回归斜率，dB/八度）；
  * **组延迟统计（MGD-lite）**（1–4 kHz 均值/方差/分位数）；
  * **LPC 残差能量比 & 平坦度**（两带：0–1k、1–4k）。
* 拼成 `E[t]`（**建议 32 维**），并生成 `voiced_mask[t]`（voicing>0.5）。
  **存储规范（与你的 C 同分片/格式）**

```
data/ASVspoof5/features/E/
  train/
    data_000.npy, data_001.npy, ...           # ragged memmap shards (float16)
    index.jsonl                                # {utt_id, shard, offset_elems, elem_count, real_len, L=1, D=32, stride_ms=20, dtype="float16", layout:"TD"}
  dev/
    ...
  eval/   # 可选（若在线抽取评测已足够，可不落盘）
```

说明：E 以“逐样本扁平化”的方式写入分片，与 C 一致（L=1，TD 布局）；voiced mask 建议以并行的 `V/` 目录落盘（uint8 ragged memmap，同一份 index.jsonl 复用 `elem_count`/`offset_elems`）。

**产物**：完整的 E/V 分片 + `index.jsonl`；统计表（总样本数/总元素/平均帧数）。
**验收**：随机抽 100 条，检查 `E.shape[0] == C.shape[0]`（允许 ±1 帧），全量磁盘增量≈ **49–50 GiB/4242h（20ms, 32 维, FP16）**。

---

## 2)（独立）DataLoader 对齐升级（C + E + wav 段）

**目标**：训练/推理阶段从缓存**直接读** C 与 E，并提供 4s 滑窗切片。
**要做**

* `__getitem__` 返回：`C_frames[T,288]`, `E_frames[T,32]`, `voiced_mask[T]`, `wav_path`, `segments_idx[List[(start,end)]]`, `utt_id`；
* 统一 4s 滑窗（步长 2s），窗内同步切片 C/E/voiced\_mask（**不需要**先做建模）。读盘端优先做“同分片聚合读取”。
  **产物**：`dataloader_c2s.py`。
  **验收**：跑一个 epoch 的 I/O smoke test，确保内存常驻稳定、吞吐 OK。

---

## 3)（独立）AASIST3 基线回归测试（保证不受新管线影响）

**目标**：确认加入 E 的读取不会拖慢/破坏现有基线。
**要做**

* 用新 DataLoader 仅把 C 和 wav 喂给你现有的 WavLM+AASIST3，记录吞吐对比；
* 这一步**不需要**任何 C²S 计算，只是回归测试。
  **产物**：基线吞吐/显存日志（before / after）。

---

## 4) 内容分桶（PCA64 + KMeans）

**目标**：把内容流 C 离散成 K 个桶，便于条件回归稳健。
**要做**

* 对 `C` 再做 **PCA→64**（仅建模用，不替换你 288 维原件）；
* 在 **train-bonafide** 全部帧（可 20–30% 抽样）上做 **KMeans（K=128 & 200 两套）**；
* 序列化：`content/pca64.pkl`, `content/kmeans_k128.pkl`（含均值/方差）。
  **产物**：分桶模型；`assign_content_bucket()` API。
  **验收**：随机帧分布均匀度检查（各桶样本数无极端失衡）。

---

## 5)（可并行于 4）Codec 等价视图的 E 批量缓存

**目标**：为 **train-bonafide** 生成 2–3 个 codec 视图的 E 缓存，供后续“等价观测”学习。
**要做**

* 视图集合（建议）：

  * **16k**：MP3(96/32kbps)、AAC(64kbps)、Opus(24kbps)、G.722（任选其二）；
  * **8k**：下采→AMR-NB(12.2)/GSM/μ-law（选 1–2 个）→上采回 16k。
* 对这些视图重复**第 1 步**流程，仅输出 **E、V、meta**（不重算 C）。
* 存储：与 E 相同的 ragged memmap 组织，建议仅对 **train-bonafide** 落盘（目录如 `features/E_views/train/<view>/...`）。
  **磁盘预算**：每新增 1 个视图（仅 train）≈ **2–5 GiB**（视训练总时长而定），远小于“全量 4k+ 小时”时的 90 GiB 估算。
  **产物**：多视图 E 缓存 + `manifest_views.csv`。

---

## 6) 条件密度 p(E|C) 拟合（只用 bonafide）

**目标**：学“内容→激励”的线性–高斯条件族，具备 **codec-不变性**。
**要做**

* 对每个桶 `k`：拟合 `E = A_k * C64 + b_k + ε`（**Ridge 回归**，α=1e-2 起）；
* 残差估计 **对角协方差 Σ\_k**（加 ϵ=1e-4 防奇异）；
* 训练集用 **原始 E + 视图 E** 合并（把视图当“等价观测”）一并拟合；
* 序列化：`models/c2s_linGauss_k128.pkl`（含 {A\_k,b\_k,σ\_k}）。
  **产物**：C²S 条件模型（K=128 & 200 两套）。
  **验收**：在 bonafide 上测残差分布（QQ plot/均值接近 0，方差稳定），视图间 NLL 无系统偏移。

---

## 7) 推理打分（窗级 NLL → 句级汇聚）

**目标**：生成每条语音的 **C²S(raw)** 分数。
**要做**

* 帧级：只在 `voiced_mask==1` 上计算

  ```
  nll_t = 0.5 * Σ_j ((E_j - (A_k C64 + b_k)_j) / σ_kj)^2 + 0.5 * Σ_j log σ_kj^2
  ```
* 窗级：trimmed-mean（截尾 10%）或逆方差权平均；
* 句级：对所有窗平均 → `C2S_raw`；输出到 `score_c2s.jsonl`。
  **产物**：分数文件（dev-eval / progress）。
  **验收**：分数直方图可分、bonafide 均值低于 spoof。

---

## 8) 校准与评测

**目标**：与挑战指标接轨，并与 AASIST3 做轻量对照。
**要做**

* 用 **dev-cal** 做 BOSARIS 逻辑回归校准（单变量），得到 `C2S_cal`；
* 在 **dev-eval / progress** 报：EER、minDCF、Cllr、actDCF；
* **可选融合**：`S = α·z(C2S_cal) + (1-α)·z(AASIST3)`，在 dev-cal 网格选 α。
  **产物**：三行（C²S / AASIST3 / 融合）× 四列指标表。

---

## 9)（增强）高精度激励：GCI/QCP & 相位特征

**目标**：在窄带/低码率下进一步稳固。
**要做**

* GCI 估计 + 闭合相位窗内统计（周期能量变化、相位坡度、简化 R\_d proxy）；
* 把这些维度追加进 E，重训第 6 步（可做小规模）。
  **产物**：`features_excitation_advanced.py`；对比指标。

---

## 10)（论文素材）消融与可解释化

**目标**：证明“核心思想”而非堆料；提供图表。
**要做**

* 消融：无视图训练 / 仅 F0+HNR / 不同 K；
* 难例切片：**8k/AMR/GSM × YourTTS** 的窗级 nll\_t 热图（叠 F0/voiced）。
  **产物**：2–3 张核心对比图 + 2 张热图。

---

## 附：3 个即可拷的函数签名（方便你落地）

```python
# features/excitation.py
def extract_excitation_gpu(wav: torch.Tensor, sr: int, win_ms: float, hop_ms: float
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    wav: (T,) float32 on CPU/GPU, sr=16000
    return:
      E: (n_frames, 32) float16  # excitation features
      V: (n_frames,) uint8       # voiced mask (0/1)
    """

# content/bucketing.py
def fit_pca_kmeans(C_iter, pca_out: str, kmeans_out: str, k: int = 128):
    """C_iter yields chunks of (n_frames, 288) float32; save pca(64) + kmeans(k)."""

def assign_bucket(C_frame_288: np.ndarray, pca, kmeans) -> int: ...

# models/c2s.py
def fit_lin_gauss(buckets, C64_ds, E_ds, alpha=1e-2) -> dict:
    """
    Fit per-bucket: E = A_k @ C64 + b_k, diag var sigma_k^2
    return {k: (A_k, b_k, sigma_k)}
    """
```

---

# 现在就能开的 3 个螺丝（与你的现状最贴合）

1. **执行第 1 步**：跑离线 E/V 提取（仅对 train/dev；eval 可在线抽取），以 ragged memmap 落盘，出 `index.jsonl` 与统计。
2. **执行第 2 步**：把 DataLoader 升级到能同时读 `C+E+V+segments`，并做“分片内聚合”读取的小测（吞吐/显存）。
3. **执行第 4 步**：PCA64 + KMeans（K=128/200），保存模型；随后直接做第 7–8 步出第一版 C²S 曲线。

> 若要“codec 等价”，再做第 5 步补视图的 E（仅 train）；不影响第 1–4 步产物与模型接口。

---

# 风险与缓解
- I/O 抖动：仅使用 ragged memmap + 分片内聚合读取；避免“每 utt 一个 .npy”的海量小文件设计。
- Voiced mask 鲁棒性：在 dev 上小规模对比倒谱/自相关两种 voicing 判定；阈值搜索（0.4–0.6）。
- 桶化边界效应：如硬分配不稳，可尝试 soft top-2 加权（简化实现：两桶 NLL 最小值的凸组合）。
- 校准偏移：除 BOSARIS 单变量逻辑回归外，备份 z-norm/shift 以控制 actDCF–minDCF gap。

---

# 需要进一步讨论的点
- E 的维度与构成：当前建议 32 维（F0/HNR/谱倾斜/组延迟统计/LPC 残差等）；是否需要加入相位/GCI（第 9 步）视 dev 收益决定。
- 多视图的性价比：优先做 1–2 个典型 codec（MP3 32/96kbps 或 Opus 24kbps）；若收益边际小，可停在单视图。
- 评测与融合：与 AASIST3 的轻融合是否作为提交主线，还是仅作对照曲线。
