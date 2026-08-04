# 證據

每一節都附可重跑的指令。全部在 `conda activate pangu_env` 下,工作目錄
`/wk2/pc/AI_models/LLAT_manifold_experiments`。

---

## E1 — 決定性檢驗:$f = 0$ 的空跑

模式是決定性的,所以零強迫必須給零擾動。這是整件事的驗收條件。

| $n$ | 修正前 $\lvert\delta T\rvert_{\max}$ | 修正後 |
| ---: | ---: | ---: |
| 1 | 0.823 K | $1.53\times10^{-5}$ K |
| 2 | 1.957 K | $1.88\times10^{-5}$ K |
| 3 | 6.224 K | $4.08\times10^{-5}$ K |
| 4 | 8.812 K | $1.21\times10^{-4}$ K |
| 8 | 5.537 K | $4.78\times10^{-4}$ K |
| 16 | 10.763 K | $8.61\times10^{-4}$ K |
| 40 | 16.645 K | $2.81\times10^{-2}$ K |

**修正後不是嚴格 0,而是 float32 的 machine epsilon 等級**:
$\varepsilon \cdot 300\,\mathrm{K} \approx 3.6\times10^{-5}$。ONNX 的 CPU 推論不保證兩次
呼叫的規約次序相同,所以同樣的輸入會差在最後一位。這是實作層的雜訊,不是方法的缺陷。

它**會被迭代放大**(每步約 ×1.2,40 步累積約 ×1800),所以 0 K 空跑保留下來當
**雜訊底噪**,但不再相減。

```bash
python scripts/run_amp_sweep.py --amps 0 --inits 2025092000 --mode snapshot \
    --steps 40 --forcing-steps 8 --ic axisym --tag-prefix tseries --family heating_moist
```

---

## E2 — 舊資料壞到什麼程度:0 K 與 5 K 幾乎是同一個場

這是最有說服力的一張表。把**舊的**空跑與**舊的** 5 K run 並排:

| $n$ | 舊 0 K $\lvert\delta T\rvert_{\max}$ | 舊 5 K $\lvert\delta T\rvert_{\max}$ | 差 |
| ---: | ---: | ---: | ---: |
| 1 | 0.823 | 0.825 | 0.2% |
| 2 | 1.957 | 1.948 | 0.5% |
| 4 | 8.812 | 8.207 | 7% |
| 8 | 5.537 | 5.872 | 6% |
| 16 | 10.763 | 11.086 | 3% |
| 24 | 15.064 | 14.799 | 2% |
| 40 | 16.645 | 16.640 | **0.03%** |

**δ 有約 98% 與強迫無關。** 上一版我用「扣掉 0 K 空跑」把訊號撈回來 —— 數值上可行,
但撈回來的東西是「兩條已經分岔的完整非線性軌跡之差」,那是 twin experiment,
**不是強迫響應**,而且在 $n$ 大時它量的是混沌發散。

舊 run 保留在 `outputs/diabatic_heating_axisym/_archive_snapshot_pre_u0fix/`(28 個,
4.7 GB),當證物不當資料。

---

## E3 — 修正後的訊噪比

同一組 init(0920, Category 1),$\lvert\delta T\rvert_{\max}$:

| $n$ | 0 K(底噪) | 1 K | 5 K | 10 K | 5K / 0K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | $1.5\times10^{-5}$ | 0.084 | 0.416 | 0.831 | $2.7\times10^{4}$ |
| 2 | $1.9\times10^{-5}$ | 0.126 | 0.632 | 1.293 | $3.4\times10^{4}$ |
| 4 | $1.2\times10^{-4}$ | 0.282 | 1.038 | 1.878 | $8.6\times10^{3}$ |
| 8 | $4.8\times10^{-4}$ | 0.808 | 2.752 | 5.016 | $5.8\times10^{3}$ |
| 16 | $8.6\times10^{-4}$ | 2.298 | 7.669 | 8.309 | $8.9\times10^{3}$ |
| 24 | $2.4\times10^{-3}$ | 7.417 | 8.533 | 9.192 | $3.6\times10^{3}$ |
| 40 | $2.8\times10^{-2}$ | 3.480 | 4.830 | 5.039 | $1.7\times10^{2}$ |

訊號比底噪高 2–4 個數量級。**$n=1$ 的振幅標度幾乎完美線性**:
0.084 / 0.416 / 0.831 對 1 / 5 / 10 K → 每 K 是 0.0832 / 0.0832 / 0.0831。

窗內方位平均 ΔPV 的 RMS 每 K(0920):

| | 1 K | 2 K | 5 K | 10 K | 10K 相對 1K |
| --- | ---: | ---: | ---: | ---: | ---: |
| $n=1$ | 0.0048 | 0.0048 | 0.0047 | 0.0044 | −8% |
| $n=8$ | 0.1377 | 0.1342 | 0.1170 | 0.0950 | **−31%** |

$n=8$ 的 31% 下垂是**真的飽和**(底噪在該處只有 $1.3\times10^{-4}$ PVU)。

---

## E4 — prescribed 通道:$u_0$ vs $M(u_0)$

```bash
python - <<'PY'
import numpy as np, sys; sys.path.insert(0,'src')
from llat_manifold import io, layout
from llat_manifold.idealized_vortex.background import load_background
R='outputs/diabatic_heating_axisym/heating_moist/tseries_5K_iter40_init2025092000/data/background.npz'
st=load_background('outputs/axisymmetric_ic/axisym_202518W_2025092000.npz')
u0=np.float32(st.surface); _up,ub=io.load_delta_bundle(R)
for n in ("sst_filled","solar","hgt","landmask","diurnal_sin"):
    i=layout.surface_index(n)
    print(f"{n:12s} u0=({u0[:,:,i].min():8.3f},{u0[:,:,i].max():8.3f})  "
          f"ubar=({ub[:,:,i].min():8.3f},{ub[:,:,i].max():8.3f})")
PY
```

**修正後**這段會印出兩邊完全一致(因為 `background.npz` 現在鎖到 $u_0$)。
要看未鎖的原始模式輸出,得直接呼叫 `m_operator(..., advance_time=False)` 不帶 `lock_ref`。
缺陷 2 那張表(地形 1899 → 1041 m 等)就是這樣量出來的。

---

## E5 — 離線迴歸測試

```bash
python tests/test_offline.py        # ALL OFFLINE TESTS PASSED
python tests/test_vortex_offline.py # ALL VORTEX OFFLINE TESTS PASSED
```

新增三個(見 [01_the_bug.md](01_the_bug.md) 末節)。它們不需要 GPU、不需要真實模式,
用 `_DriftingDLAMPty`(`up → 0.9·up + 0.1`、`sfc → sfc + 1`)就能區分對錯。
`test_snapshot_recursion_uses_u0_as_base` 檢查的閉式解:

$$\delta_i = 0.9(\delta_{i-1} + f) \;\Rightarrow\; \delta_3 = 2.439\,f$$

實測 `iter 03/3 | |δT|max=2.439`。

---

## E6 — domain 確實沒動,而且沒動在**對的**地方

```bash
cd experiments/diabatic_heating/src/myplot
python fig_MP7_domain_drift.py --mode snapshot
```

輸出:每個 $n$ 都是 `centre 16.50N 129.50E   core land 0.0 %   core SST 302.57 K`,
`total centre displacement n=1 -> n=40: 0.00 deg`。

**修正前這張圖也會給 0.00°** —— 因為它確實凍結了,只是凍結在 $\bar u$ 上,
也就是位移 0.079° 之後、地形壓平之後的世界。所以「位移 0.00°」這個檢查
**不足以**證明設定正確,必須同時檢查 lock reference 的值。這是這次除錯最容易被忽略的一點。
