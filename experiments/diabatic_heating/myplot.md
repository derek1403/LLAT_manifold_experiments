# myplot — 補充圖組(semi-linear / snapshot 版)

> **這份文件已整份重做。** 先前的版本建立在 `continuous` mode 的 run 上 —— 那個 mode 讓
> control 軌跡逐步前進、TC-following domain 跟著颱風走,結果 0920 在第 5 天核心有 70% 是陸地。
> 那不是這套研究要問的問題。現在全部改用 **`snapshot` mode(semi-linear power iteration)**:
> 背景 $\bar u = M(u_0)$ 只算一次、每次迭代都被拉回它,**位置與環境永遠不動**。
>
> **時間軸的意思也跟著變了:$n$ 是模式算子的迭代次數,不是預報時效。** 全文不再出現「第幾小時」。
> 強迫在 $n \le 8$ 注入(每次 amp/8),之後不再加。
>
> 程式在 [`src/myplot/`](src/myplot/),圖在 [`figs/myplot/`](figs/myplot/)。個案:202518W RAGASA。
>
> **§0-2 那個 bug 的完整交接文件在 [`handoff/`](handoff/)** —— 程式碼位置、代數推導、
> 全部實測證據、撤回的結論、剩下的工作。

---

## 0. 兩個必須先講的方法學修正

### 0-1 mode 跟錯了 → 已重跑

`driver._run_continuous` 的 control 那次 `m_operator` **沒有傳 `lock_ref`**,而且 `I ← M(I)`
每步覆寫,所以 domain 會漂。`driver._run_snapshot` 才是位置釘死的那個。詳細比對與證據見
[`my_comment_reply.md`](my_comment_reply.md)。

`scripts/run_amp_sweep.py` 已加上 `--mode snapshot`,重跑了 **13 個振幅 × 2 成員 = 26 個 run**
(外加 2 個 0 K 雜訊底噪 run,見 §0-2)。

![MP7 snapshot](figs/myplot/mp7_domain_drift_snapshot.png)

**所有虛線框完全重合、核心陸地與 SST 全平 —— domain 位移 0.00°。** 對照
`mp7_domain_drift_continuous.png` 是舊 mode 的同一張圖,框往西北走了 16.9°。

### 0-2 `_run_snapshot` 的 base state 放錯 → 已修程式碼並全部重跑

上一版我發現 0 K 空跑不是零,判斷成「$\bar u$ 不是 $M$ 的不動點,所以要事後扣掉」。
**那個判斷是錯的 —— 那不是物理,是 bug,而且是兩個。**

`_run_snapshot` 原本寫成

$$u'_1 = M(u_0 + f) - \bar u,\qquad u'_i = M(\bar u + u'_{i-1} + f) - \bar u \quad (i \ge 2)$$

**問題一:$i \ge 2$ 的 base 用 $\bar u$。** 由於 $u'_{i-1}$ 是從 $\bar u$ 量起的,

$$\bar u + u'_{i-1} = \bar u + (u_{i-1} - \bar u) = u_{i-1},$$

**餵進模式的其實就是前一次的輸出**。這個迴圈根本不是同一個算子的迭代,而是一條自由的
非線性積分 $u_i = M(u_{i-1}+f)$;而且線性近似下多出一個常數項
$(\mathbf J - \mathbf I)\bar u = M(\bar u) - \bar u$,它**與 $f$ 無關、因此與 $f$ 的正負號無關**,
±A 反對稱檢驗會把它整包讀成「非線性」。

**問題二:11 個 prescribed 通道鎖到 $\bar u$ 而不是 $u_0$。** DLAMPty 會連同其他變數一起
「預測」這些通道,而且預測得很糟 —— 一步之後地形被壓平 45%(1899 → 1041 m)、land mask
不再是二元(−0.06…1.05)、經緯度位移 0.08°、日夜編碼前進 3 h。原本的程式把這些採納成
往後每次迭代的下邊界,所以 `advance_time=False` 實際上凍結在 $t_0+3$h 而不是 $t_0$。

改成 base 恆為 $u_0$、lock 恆對 $u_0$:

$$u'_i = M(u_0 + u'_{i-1} + f) - M(u_0)$$

線性極限下正好是 $u'_i = \mathbf J u'_{i-1} + \mathbf J f$ —— 也就是 `docs/perturbation_method.md`
§3.1 一直宣稱、但程式碼沒做到的 forced power iteration。**模式是決定性的,所以 $f=0$ 必須給
$u' \equiv 0$**,這就是驗收條件:

| $n$ | 修正前 $\lvert\delta T\rvert_{\max}$ | 修正後 |
| ---: | ---: | ---: |
| 1 | 0.823 K | $1.5\times10^{-5}$ K |
| 2 | 1.957 K | $1.9\times10^{-5}$ K |
| 4 | 8.812 K | $1.2\times10^{-4}$ K |
| 40 | 16.645 K | $2.8\times10^{-2}$ K |

**舊資料壞到什麼程度**:把舊的 0 K 與舊的 5 K 並排,$n=40$ 時是 16.645 vs 16.640 K ——
**兩者幾乎是同一個場,δ 有 ~98% 是與強迫無關的漂移**。事後扣空跑確實能撈回訊號,但撈回來的
是「兩條已經分岔的完整非線性軌跡之差」,是 twin experiment,不是強迫響應。

修正後 $1.5\times10^{-5}$ K 是 float32 對 $T\approx300$ K 的 machine epsilon
($\varepsilon\cdot300 \approx 3.6\times10^{-5}$);ONNX CPU 推論的規約次序不保證逐次重現。
它會被迭代放大(每步約 ×1.2,$n=40$ 到 $2.8\times10^{-2}$ K),所以 **0 K 空跑保留下來當
雜訊底噪**,但**不再相減**(`_snap.dpv_field` 的 `subtract_null` 預設已改為 `False`)。

程式碼:[`driver.py`](../../src/llat_manifold/driver.py) `_run_snapshot`;方法學:
[`docs/perturbation_method.md`](../../docs/perturbation_method.md) §3.1.1;
迴歸測試:`tests/test_offline.py::test_snapshot_null_run_is_exactly_zero` 等三個
(舊的 `test_snapshot_math` 用 identity $M$,此時 $\bar u = u_0$ —— 正是這個巧合讓兩個 bug 藏住)。
舊 run 保留在 `outputs/diabatic_heating_axisym/_archive_snapshot_pre_u0fix/`,當證物不當資料。

---

## 圖 1 — 0920 / 0921 軸對稱絕對 PV(未變)

![MP1](figs/myplot/mp1_axisym_pv.png)

這張只用 $n=0$ 的 IC,**不受 mode 影響,數字與先前完全相同**:
PV 塔最大值 4.45 PVU @ 500 hPa(0920)→ 7.25 PVU @ 300 hPa(0921),
850 hPa 核心 2.20 → 3.25 PVU,RMW 173 → 147 km。程式
[`fig_MP1_axisym_pv.py`](src/myplot/fig_MP1_axisym_pv.py),caveat 見程式 docstring。

---

## 圖 2 — 同一份加熱打進兩個強度,逐迭代

程式:[`fig_MP2_pv_evolution.py`](src/myplot/fig_MP2_pv_evolution.py)
圖:`mp2-01` … `mp2-10_pv_evolution.png`、`mp2-m01/m02/m05`(降溫)、`mp2-05_early`

![MP2 5 K](figs/myplot/mp2-05_pv_evolution.png)

上列 Category 1(0920)、下列 Category 4(0921),欄位是 **迭代 $n$ = 1/2/4/8/16/24/32/40**,
前四欄仍在注入。兩列的環境**逐位元相同**,而且在整個實驗中都不變。

**窗內方位平均 ΔPV 的正峰值(PVU):**

| 振幅 | 0920 $n$=1 | 0921 $n$=1 | 比值 | 0920 $n$=8 | 0921 $n$=8 | 比值 | 0920 $n$=40 | 0921 $n$=40 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 K | 0.019 | 0.018 | 0.99 | 0.61 | 0.45 | 0.73 | 0.91 | 2.78 |
| 3 K | 0.055 | 0.055 | 1.01 | 1.70 | 1.20 | 0.70 | 0.77 | 1.43 |
| 5 K | 0.088 | 0.092 | 1.04 | 2.52 | 1.78 | 0.71 | 0.86 | 0.78 |
| 8 K | 0.133 | 0.149 | 1.12 | 3.55 | 2.86 | 0.81 | 0.72 | 1.82 |
| 10 K | 0.158 | 0.188 | 1.19 | 4.11 | 3.51 | 0.85 | 0.64 | 1.25 |

1. **$n=1$ 是唯一乾淨的線性點**:窗內 RMS 每 K 的響應是 0.0048 / 0.0048 / 0.0047 / 0.0044
   (1/2/5/10 K),10 K 才掉 8%。到 $n=8$ 已經是 0.138 / 0.134 / 0.117 / 0.095,掉 31% ——
   **這是真正的飽和,不是雜訊**(底噪 RMS 在 $n=8$ 只有 $1.3\times10^{-4}$ PVU)。
2. **$n=1$ 兩成員幾乎同響應(比值 0.99–1.19,隨振幅單調上升)**;到 $n=8$ 反轉成
   0.70–0.85(Category 1 響應較大)。窗內 RMS 的比值也給同一個轉折:
   $n$=1 時 1.41、$n$=8 時 0.92、$n$=16–24 時 0.78 / 0.69。
   **強弱渦旋的差別不在瞬時響應,而在強迫停掉之後誰放大得快。**
3. **$n=40$ 的數字不是振幅的乾淨函數**(0920:0.91/0.77/0.86/0.72/0.64;0921 更亂)。
   power iteration 到那裡已經進入非線性放大,單一數值不可讀,只能讀趨勢。
4. **兩個先前結論都已撤回**:「強渦旋留得住、弱渦旋留不住」是 `continuous` mode 下
   0920 登陸造成的;而上一版「$n=8$ 比值穩定 0.81–0.91、較弱渦旋響應略大」則是
   §0-2 那批壞資料的產物(當時 δ 有 98% 是漂移)。

---

## 圖 3 — $q\,\partial\dot\theta/\partial\theta$ 的預測 vs 模式

程式:[`fig_MP3_pv_tendency.py`](src/myplot/fig_MP3_pv_tendency.py)
圖:`mp3-05_n001_pv_tendency.png`(線性區)、`mp3-05_n008_pv_tendency.png`(注入窗末)

![MP3 n=1](figs/myplot/mp3-05_n001_pv_tendency.png)

$\dot\theta$ **不用診斷,是解析已知的**(注入是我們自己加的),推導與物理圖像見程式 docstring。
背景取自**凍結的 $\bar u$**,不是 $n=0$ 的 IC。

| 5 K | 預測 ΔPV | 實測 ΔPV |
| --- | --- | --- |
| **$n=1$** Category 1 | −0.11 … +0.13 | −0.10 … +0.09 |
| **$n=1$** Category 4 | −0.18 … +0.19 | −0.19 … +0.09 |
| $n=8$ Category 1 | −0.90 … +1.05 | −0.23 … +2.52 |
| $n=8$ Category 4 | −1.45 … +1.52 | −1.39 … +1.78 |

**在 $n=1$ 理論站得住**:低層正、高層負,量級也吻合。到 $n=8$ 量級仍同級但結構已經改變 ——
與圖 4 量到的非線性接手時間一致。

---

## 圖 4 — 降溫實驗:±A 反對稱性

程式:[`fig_MP4_antisymmetry.py`](src/myplot/fig_MP4_antisymmetry.py)
圖:`mp4-05_0920/0921_pv_sections.png`、`mp4_antisymmetry_summary.png`

![MP4 0921](figs/myplot/mp4-05_0921_pv_sections.png)

三列共用同一組色階:增溫、降溫、非鏡像殘差。若響應線性,第 2 列是第 1 列的顏色翻轉、
第 3 列全白。

**非線性佔比 RMS(even)/RMS(odd)** —— 不再扣任何東西,底噪比訊號低 3–4 個數量級:

| n | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Category 1 | **7%** | 9% | 14% | 19% | 23% | 70% | 151% | 141% |
| Category 4 | **5%** | 10% | 20% | 27% | 66% | 136% | 259% | 416% |

- **$n \le 2$ 幾乎完美反對稱(5–10%)** —— 你預期的「降溫時上層 PV 變大、下層變小」
  在這裡成立,剖面上第 2 列就是第 1 列的鏡像。
- **非線性單調爬升,沒有突變**:$n$=4 已有 14–20%,$n$=8(注入結束)19–27%,
  $n \ge 24$ 超過 100% 就不能再當「擾動」讀。這比舊資料乾淨得多 —— 舊版在 $n$=2 就跳到
  600%(因為分子整包是背景漂移)。
- **Category 1 全程比 Category 4 更線性**(40 次迭代後 141% 對 416%)。
  **注意這與上一版相反**,上一版寫的是 Category 4 更線性(70% 對 485%);
  那組數字建立在 §0-2 的壞資料上,已作廢。

> **警語**:這張表的分母是 odd 部分;$n$ 很大時 odd 本身會塌掉,比值就會爆大。
> 而且比值 ≈100% 也是兩個**互不相關**的場會給的值,只有明顯高於它才要求真正的同號整流。

---

## 圖 5 — 次級環流與 ΔPV 收支

程式:[`fig_MP5_pv_budget.py`](src/myplot/fig_MP5_pv_budget.py)
圖:`mp5-1_<成員>_secondary_circulation.png`、`mp5-2_<成員>_pv_budget.png`

![MP5-1 0921](figs/myplot/mp5-1_0921_secondary_circulation.png)

$\delta$ 本身就只含強迫響應(不需扣任何東西),所以這是**加熱自己造出來**的環流:
深厚上升柱貫穿整層、低層向心流入、高層輻散外流 —— 平流假說的前提成立。

**收支(五列時間積分到該欄,單位 PVU;空間相關 = 假說 vs 實測):**

| n | 0920 相關 | 0921 相關 |
| ---: | ---: | ---: |
| 1 | +0.31 | **+0.86** |
| 2 | +0.08 | **+0.79** |
| 4 | +0.14 | +0.59 |
| 8 | +0.31 | +0.49 |
| 16 | +0.35 | +0.20 |

**Category 4 在 $n \le 2$ 相關 0.79–0.86,假說量化成立**,而且 $n$=4–8 仍有 0.49–0.59;
**Category 1 從頭到尾只有 0.08–0.35 —— 同一個假說在弱渦旋上不成立**,這是新資料才看得出來的
(舊資料兩者都在 0.16–0.51 之間,分不出來)。之後線性化預算過預測一個數量級
(擾動速度 × 背景梯度一路積分,沒有回饋限制它)。收支「是什麼、不是什麼」的四條界定
(略掉 $-\bar{\mathbf v}\cdot\nabla\delta q$ 與 $-\delta\mathbf v\cdot\nabla\delta q$、
方位平均丟掉渦旋相關項、逐欄色階、`STEP_S` 只是算子步長不是時鐘)見程式 docstring。

在 snapshot 下這個收支比先前**更有意義**:背景凍結,所以 $\partial\bar q/\partial p$ 與
$\partial\bar q/\partial r$ 在每個 $n$ 都相同,微分背景不再混進 domain 滑動。

---

## 圖 6 — BG1 版面,欄位是迭代

程式:[`fig_MP6_state_evolution.py`](src/myplot/fig_MP6_state_evolution.py)
圖:`mp6-05_0920/0921_state_evolution.png`

![MP6 0921](figs/myplot/mp6-05_0921_state_evolution.png)

色階、level 表、row 定義、風場疊加全部從
[BG1](/wk2/pc/seminar/seminar1/figs/src/make_background_state.py) **import**,不是抄的。

**與先前版本最大的差別:每一欄的地圖窗口完全相同。** 颱風不移動、不登陸,
先前那條「domain 是 TC-following,窗口會滑動」的警語已經不需要了 —— 那正是這次修正掉的東西。

畫的是 $\bar u + \delta$。$\delta$ 本身就是可歸因於強迫的量,不需要再扣任何東西。
`--field null` 看 0 K 空跑(雜訊底噪)、`--field control` 看凍結的 $\bar u$(每欄相同)。

---

## 重現

```bash
conda activate pangu_env
cd /wk2/pc/AI_models/LLAT_manifold_experiments

# 0. 先確認 driver 的 snapshot 遞迴是對的(離線,不需要模式)
python tests/test_offline.py

# 1. 跑 run(26 個 + 2 個 0 K 雜訊底噪;snapshot 每迭代只呼叫模式一次,約 35 秒/run)
#    --tag-prefix tseries 不可省:預設是 sweep,而 _snap.py 找的是 tseries_*
python scripts/run_amp_sweep.py --ic axisym --env-from 2025092000 \
    --inits 2025092000 2025092100 --mode snapshot \
    --amps 1 2 3 4 5 6 7 8 9 10 -1 -2 -5 0 \
    --steps 40 --forcing-steps 8 --tag-prefix tseries --family heating_moist

# 2. 出圖
cd experiments/diabatic_heating/src/myplot
python fig_MP1_axisym_pv.py
python fig_MP2_pv_evolution.py                      # 1–10 K
python fig_MP2_pv_evolution.py --amps -1 -2 -5      # 降溫
python fig_MP3_pv_tendency.py --lead 1              # 也可 --lead 8
python fig_MP4_antisymmetry.py                      # 加 --summary 出折線版
python fig_MP5_pv_budget.py
python fig_MP6_state_evolution.py
python fig_MP7_domain_drift.py --mode snapshot      # 也可 --mode continuous 看舊 mode
```

## 還沒做的

1. **`report_intensity_ladder_zh.md` 尚未更新** —— 它整份建立在 `continuous` 的 24 h 掃描上。
   主表(hour 24)不受 domain 漂移影響(該時刻核心陸地 0%),但若要宣稱「這是 semi-linear
   的結果」,措辭與資料都要換成 snapshot。
2. **`continuous` 的舊 run 沒有刪** —— 它們回答的是另一個(也合法的)問題:
   「加熱對一條會移動、會登陸的軌跡有什麼影響」。要不要保留當對照,待決定。
   注意 `_run_continuous` 也有 §0-2 的問題二:control 那次 `m_operator` 沒有 `lock_ref`,
   所以 control 自己的 prescribed 通道逐步劣化,40 步之後地形/land mask 已經失真。
3. **其他 snapshot 家族(qlock / dq_measured 等)尚未重跑** —— §0-2 的修正是在 `driver.py`,
   對所有 snapshot run 一體適用,這批只重跑了 `heating_moist`。
4. **`outputs/idealized_vortex/ring_*` 的 6 個 snapshot run 也是修正前產生的**,
   若還要引用需重跑。
