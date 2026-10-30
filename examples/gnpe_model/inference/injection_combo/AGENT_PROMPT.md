# 发给 agent 的 prompt：生成 overlapping-signal coverage heatmap

> 用法：复制下面分隔线之间的全部内容发给 agent，只需改写「我这次的需求」一节。

---

你是一个帮我在 HPC 集群上生成引力波重叠信号研究热图的 agent。请先完整读完本 prompt，再读 `injection_combo/HEATMAP_LOGIC.md` 和 `injection_combo/README.md`，然后执行我的需求。

## 我这次的需求

<!-- 在这里写清楚。例子：
- 用现有数据出 snr_ratio sweep 的三面板图，面板 = chirp_mass / mass_ratio / luminosity_distance
- 重新汇总 separation 和 separation_clone（它们的 npz 是补跑前算的），再出 4×3 矩阵图
- 新跑一个 mass_ratio sweep，跑完出图
-->

## 研究设计（背景）

- 一个**固定的目标信号 signal 1**：ℳ₁ = 57.66 M⊙，q₁ = 0.896，d_L,1 = 894.07 Mpc，网络 SNR₁ = 25.0，seed 1，IMRPhenomPv2，H1+L1，O2 PSD，模型 `1M_7params_O2`（GNPE，推断 7 个参数：chirp_mass, mass_ratio, luminosity_distance, geocent_time, ra, dec, psi）。signal 1 在所有 sweep 中永远不变。
- 叠加一个**干扰信号 signal 2**。每个热图是 10×10 网格：
  - **纵轴永远是 |Δt_c|**：0.05–1.0 s 对数等分 10 档，signal 2 比 signal 1 **更早**并合（已实测，inference.py 里 `--shift` 的 help 写"later"是错的）。
  - **横轴**是 signal 2 被控制的某个属性。
- **每格 100 次注入**，每次注入 = 一段 h₁ + shift(h₂) + 噪声 → 一个后验。
- **格子颜色** = 这 100 次中，目标参数真值落在 90% 中心可信区间内的比例（覆盖率），色标固定 0–1、红→绿。

## 环境（必须三步都做，否则 `import numpy` 就会失败）

```bash
cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/injection_combo
module load GCCcore/12.3.0 Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export MPLBACKEND=Agg
```

- `/tmp` 是节点本地的，SLURM 日志和中间文件都必须放在 BeeGFS 上。
- **不要修改 `inference/inference.py`**，所有新代码放在 `injection_combo/`，从 inference.py import 函数。

## 脚本

| 文件 | 作用 |
|---|---|
| `run_grid.py` | 跑注入（GPU），一个 SLURM 数组任务 = 一个格子，已存在的 realization 会跳过 |
| `submit_grid.sh` | 数组作业包装：`sbatch --array=0-99 submit_grid.sh <轴名>` |
| `missing_cells.py` | 列出缺失/不满的格子，并打印补交命令 |
| `summarize_grid.py` | 读 HDF5 → 覆盖率网格 `results/<sweep>/coverage_*.npz`（CPU，秒级） |
| `plot_heatmap.py` | npz → `results/figures/*.png`（CPU，秒级） |
| `verify_cell.py` | 从原始后验样本独立重算某一格，用于审计 |
| `select_target.py` | 选 signal 1、测时移方向、检查网格可达性（CPU） |
| `grid_axes.py` | 各横轴的定义 |

## 已有数据（每个 100/100 格、10000 次注入）

| sweep | signal 2 怎么构造 | 结论 |
|---|---|---|
| `snr_ratio` | 反解 d₂ 让 SNR₁/SNR₂ 精确等于 bin 值（0.2–5），其余先验随机 | 强结构，阈值在比值≈1 |
| `luminosity_distance` | d₂ 受控 100–2000 Mpc，SNR₂ 不控 | 平滑梯度 |
| `chirp_mass` | ℳ₂ 受控 10–65，q₂ = q₁，SNR₂ 钉在 25 | 只在 Δt≤0.1 s，沿质量非单调 |
| `population` | 纯先验抽样，只控 Δt；事后按 `--bin-by snr_ratio` 分箱 | 损伤只在最低十分位 |
| `separation` | d₂ = d₁，天区角 γ 受控（D = d_L·γ，0–2809 Mpc），质量随机 | **平坦**（干扰源太弱） |
| `separation_matched` | 同上 + ℳ₂ = ℳ₁、q₂ = q₁ | **平坦** |
| `separation_clone` | signal 2 是 signal 1 的完整克隆，**只有 ra/dec 不同** | 有结构，ra 两端高中间低 |
| `reference` | 无干扰信号，100 次对照 | 基线 |

`mass_ratio` 轴已实现但**从未跑过**。`separation` 的 npz 是补跑前算的（9900 次），`separation_clone` 的 npz 是 9998 次时算的——出图前重跑一次 `summarize_grid.py`。

## 标准流程

```bash
# 1. 跑注入（只有新 sweep 才需要；提交前先征得我同意）
sbatch --array=0-99 submit_grid.sh <轴名>
python missing_cells.py <轴名>              # 跑完后查缺，按打印的命令补交

# 2. 汇总
python summarize_grid.py <sweep> --mean
#   --mean-over chirp_mass luminosity_distance ...  自选平均的参数
#   --bin-by sky_separation                          按派生量重新分箱，免费
#   --bin-mode quantile|log                          派生分箱方式
#   --level 0.9                                      可信水平

# 3. 出图（三种布局）
python plot_heatmap.py results/<sweep>/coverage_<轴>.npz --parameters mean --ncols 1
python plot_heatmap.py results/<sweep>/coverage_<轴>.npz --panels parameter \
    --parameters chirp_mass mass_ratio luminosity_distance --ncols 3
python plot_heatmap.py <npz1> <npz2> <npz3> <npz4> --panels matrix \
    --parameters chirp_mass mass_ratio luminosity_distance --out results/figures/<名字>.png
#   --counts 格内显示 命中/总数   --chrome 恢复副标题和说明   --cmap cividis 色盲友好
```

## 必须遵守的规则（都是踩过的坑）

**关于资源**
1. **提交任何 GPU 作业前先问我**，说明 sweep 名、注入数、预计 GPU·h（每次注入约 6.4 s，一个 sweep ≈ 18 GPU·h）。只汇总和出图不用问。
2. `submit_grid.sh` 里必须保留 `--gres=gpu:1,VramPerGpu:20G`。集群上 VramPerGpu 是 `no_consume`，不加的话多个任务挤在一张 10 GB 卡上，会在加载模型时 CUDA OOM。
3. **不要对仍在运行的任务号补交**——两个进程同时写同一个 HDF5 会损坏文件。先 `squeue` 确认。
4. 流里的 `assert (discriminant >= 0)` 崩溃**不是按种子确定性的**（float32 GPU 临界情况），直接补交就能过。`run_grid.py` 已会跳过单次失败并记录到 `failed_realizations`。

**关于新 sweep 设计**
5. **新 sweep 提交前，先在 CPU 上抽 200–300 次测 SNR₁/SNR₂ 分布**，报告落入损伤区（比值 < 1.2）的比例。低于约 5% 的话图基本会是平的——两版 separation 就是这样浪费了 36 GPU·h。
6. signal 2 的 `geocent_time` 必须绑定到 signal 1（先验 ±0.1 s 会把 Δt 轴抹开）。
7. SNR 必须用**先投影再时移之后的纯信号**计算，不能用 context/waveform（那是 h+n）。
8. 质量约束：两个分量都在 [10, 80] M⊙，所以 ℳ ≤ 69.64，q = 0.125 只是一个点。

**关于解读**
9. **比较基准不是 0.90**。固定单一真值时，即使完美采样器也达不到名义覆盖率。无干扰对照组：chirp_mass 0.77、luminosity_distance 0.77、dec 0.80、geocent_time 0.93、ra 0.91、mass_ratio 1.00、psi 1.00，7 参数平均 0.88。热图读的是**相对基线的退化**。对照组只有 100 次，误差 ±0.042。
10. **每格误差 ±0.03**（100 次）。`mean` 的分母是 700（100×7），但 7 个判定来自同一后验、强相关，**有效样本量仍是 100**。
11. `psi` 覆盖率恒为 ~1.00（全网格最小 0.990），是平均里的死重；**`mass_ratio` 不是**——它安全区饱和在 1.00，但损伤区会塌到 0.03，动态范围 0.97。
12. 平坦的图要说明原因：separation 平是因为**干扰源太弱没触发**，不能写成"分离度不重要"。

**关于出图**
13. **面板标题永远命名"面板之间变化的那个东西"**：面板之间变的是参数，标题就是参数；变的是横轴，标题就是横轴。
14. 文件名格式 `<sweep>__<参数>_vs_<横轴>.png`，必须带 sweep 名（population 和 snr_ratio 可以同轴不同源）。
15. 默认图只保留标题 + 热图 + 色标（我要求的）。基线、每格样本数等信息靠 `--chrome` 或写进 caption。
16. **出完图必须打开看一眼**：检查标题/脚注是否溢出画布、文字是否重叠、有无灰格（无数据）或斜纹格（样本不足）。

## 完成后请交付

- 生成的图片路径列表
- 每张图的关键数字：各面板 min / max、沿横轴的散布（`nanmean(nanstd(coverage, axis=1))`）、对应基线
- 有无缺格、失败任务、异常，以及你做了什么处理
- 如果结果和预期不同（比如图是平的），说明原因，不要只描述现象
- 用中文回复

---
