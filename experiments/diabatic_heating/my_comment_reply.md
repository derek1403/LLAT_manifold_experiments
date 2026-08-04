# 回覆 `my_comment.md` — 你說對了,但它描述的是另一個 mode

> 這份文件**只盤點,不修改**。程式碼、圖、`myplot.md` 一行都沒動。
> 唯一新增的是診斷圖 `figs/myplot/mp7_domain_drift.png` 與它的程式
> [`src/myplot/fig_MP7_domain_drift.py`](src/myplot/fig_MP7_domain_drift.py)。

---

## 1. 你的描述完全正確 —— 對 `snapshot` mode 而言

`driver._run_snapshot`(driver.py:164-201)逐行就是你寫的那套:

```python
ubar_up, ubar_sfc = m_operator(u0_up, u0_sfc, dlampty, advance_time=False)   # 背景只算一次
io.save_delta_bundle(out_dir / "background", ubar_up, ubar_sfc)

for it in range(1, iterations + 1):
    base_up, base_sfc = (u0_up, u0_sfc) if it == 1 else (ubar_up, ubar_sfc)
    A_sfc = base_sfc + d_sfc + f_sfc
    _align_statics(A_sfc, ubar_sfc, active_idx)          # ← 拉回「固定的」ubar
    u_up, u_sfc = m_operator(A_up, A_sfc, advance_time=False,
                             lock_ref=ubar_sfc, lock_idx=active_idx)
    d_up, d_sfc = u_up - ubar_up, u_sfc - ubar_sfc
    _zero_locked(d_sfc, active_idx)
```

`advance_time=False`(凍結時間)、`ubar_sfc` 從頭到尾不變 —— **位置確實被釘死,不可能登陸。**

## 2. 但實際跑的資料是 `continuous` mode

`driver._run_continuous`(driver.py:204-239):

```python
Ip_up, Ip_sfc = m_operator(I_up, I_sfc, advance_time=True, valid_time=t)   # ← 沒有 lock_ref
_align_statics(A_sfc, I_sfc, active_idx)     # ← 拉回「當下的」control,不是最初的 u0
...
I_up, I_sfc = Ip_up, Ip_sfc                   # ← control 變成新基準,繼續前進
```

control 那一次呼叫**沒有傳 `lock_ref`**,而且 `I` 每步被覆寫,所以 control 的靜態通道
(含 lat/lon)是模式輸出的原樣,domain 跟著颱風走。

| 證據 | 結果 |
| --- | --- |
| `config_used.yaml` | `mode: continuous`、`total_steps: 40` |
| 檔名 | `delta_**continuous**_..._**lead120hr**`(snapshot 是 `_snapshot_..._iter040`) |
| 全 repo | `diabatic_heating_axisym`:**944 continuous / 0 snapshot**;`diabatic_heating`:460 / 3 |
| 建立時間 | 既有 5 K run stamped **2026-07-23**,早於本次對話 —— 不是這次引入的 |

**你說對的部分**:δ 的 lat/lon 通道實測**嚴格為 0.0**,perturbed 與 control 的中心逐格完全相同。
鎖定機制有在運作 —— 只是它鎖的是「perturbed → **當下的** control」,不是「→ 最初的 $loc_0$」。

**一個小更正**:被鎖的是 **11 個**通道,不是 10 個 ——
9 個 `static_surface_vars`(sst_filled, f, solar, hgt, landmask, diurnal_sin/cos, doy_sin/cos)
\+ lat + lon。

## 3. 診斷圖:一眼看穿

![MP7](figs/myplot/mp7_domain_drift.png)

**(a) 每個 lead 的 domain 外框。若位置被釘死,所有虛線框會完全重疊 —— 實際上它們往西北走。**
3 h → 120 h 中心位移 **16.9°**(0920)/ **15.3°**(0921)。

**(b)(c) 核心底下是什麼。** 前 48 h 全海洋、SST 平坦;之後 0920 的核心陸地占比衝到 80%、
SST 掉 4.8 K,0921 只到 18%、SST 掉 2.3 K。

**交叉檢查**:用 `--field control` 重畫 MP6,位移**完全一致** —— 證明位移來自 control 軌跡,
不是 `control + delta` 的算術。而且 **control 自己也登陸了**(24.7 → 42.8 m/s,
120 h 剩 20.9 m/s、中心在華東),所以 ΔPV 在 120 h 是「兩個都上岸的 run 相減」。

### 精確分界

| 量 | 0920 | 0921 |
| --- | ---: | ---: |
| 核心陸地首次 > 0 | **n = 51 h** | **n = 57 h** |
| n = 24 h 位移 / 核心陸地 | 1.71° / 0.0% | 1.64° / 0.0% |
| n = 48 h 位移 / 核心陸地 | 4.94° / 0.0% | 4.69° / 0.0% |
| n = 72 h 位移 / 核心陸地 | 9.21° / 7.8% | 8.66° / 7.4% |

兩成員中心距離:24 h **0.03°** → 48 h 0.86° → 72 h 2.13° → 120 h **3.62°**。

> **所以:n ≤ 48 h 核心全是海洋、兩成員 domain 差 < 1°;n ≥ 51 h 開始有陸地。**

## 4. 受影響盤點(依 lead 分級)

| 檔案 | 用到的 lead | 判定 |
| --- | --- | --- |
| `mp1_axisym_pv.png` | n = 0 | ✅ **不受影響** |
| `mp3-05_n003h/n024h_pv_tendency.png` | n = 3 / 24 | ✅ **不受影響** |
| `mp2-05_early_pv_evolution.png` | 3/6/9/12 | ✅ **不受影響** |
| `mp5-1/-2_*(secondary_circulation / pv_budget)` | 3/6/12/24/48 | ✅ **不受影響**(48 h 核心仍 0% 陸地);但 domain 已移 4.9°,「環境完全相同」的措辭要收斂 |
| `mp2-*.png`(12 張) | …48/**72/96/120** | ⚠️ **後兩欄受影響** |
| `mp4-05_*_pv_sections.png`、`mp4_antisymmetry_summary.png` | …**72/96/120** | ⚠️ **受影響** |
| `mp6-05_*.png` | 0/6/12/24/48/**120** | ⚠️ 最後一欄;但這張正是揭露問題的圖 |
| `myplot.md` | 圖 2 結論 2/3(建立在 120 h)、圖 4 長 lead 段、附註 | ⚠️ **要改寫** |
| `report_intensity_ladder_zh.md` | 主表 **hour 24** | ✅ **不受影響**;唯 B2 圖的 48–120 h 欄受影響 |

**最需要修的是 MP2 的頭號結論**(「強渦旋留得住、弱渦旋留不住」)—— 它建立在 120 h,
而該時刻 0920 核心 70% 是陸地、0921 只有 5%。**弱渦旋丟失響應有多少來自「它弱」、
多少來自「它上岸」,continuous 這批資料分不開。**

## 5. 若要改用 snapshot 重跑 —— 三個必須先決定的事

1. **`scripts/run_amp_sweep.py:111` 寫死 `cfg["mode"] = "continuous"`**,沒有 snapshot 選項。
   要加 `--mode snapshot` + `--iterations`,或改走 `run_experiment.py` 逐 YAML。
2. **snapshot 的輸出結構不同**:`background.npz`(單一)+ `delta_*_iterNNN.npz`,
   **沒有逐 lead 的 control**。MP2/MP4/MP5/MP6 目前全部靠 `control_*` 逐 lead 取參考場,
   資料層要改成「單一 background 當所有迭代的參考」。
3. **強迫設計要選**(這是科學選擇,不是技術細節):
   - 既有 snapshot 設定檔 `snapshot_5K_iter20_0920.yaml` 用 `amp_mode: each` + `forcing_steps: 20`
     —— 每步都加,等於**常數強迫的 power iteration**;
   - continuous 這批用 `spread` + 8 步 —— 前 8 步加完就自由演變。
   - **MP2 的頭號故事「加熱停掉之後」需要 `spread` + `forcing_steps: 8`。**

**成本**:snapshot 每迭代只呼叫模式 1 次(背景只算一次),40 迭代 ≈ 41 次
vs continuous 的 80 次 → 約 **1.4 分鐘/run**。13 個振幅(1–10、−1、−2、−5)× 2 成員
= 26 run ≈ **35–40 分鐘**、約 3 GB。

**還有一個敘事層面的後果**:snapshot 是**凍結時間的 power iteration**,$n$ 是迭代次數、
不是預報時效。所以 MP2 / MP6 現在整套「第五天」「隨時間演變」的說法都要改寫成
「第 n 次迭代」。報告本來的鐵律(**n ≠ 物理時間,不可對觀測時鐘**)其實更貼合 snapshot。

---

## 我還不確定、想請你確認的三件事

1. **要不要保留 continuous 這批當對照?** 它們回答的是另一個(也合法的)問題:
   「加熱對一條**會移動、會登陸**的真實軌跡有什麼影響」。刪掉可惜,但擺在一起容易混淆。
2. **snapshot 的 `iterations` 要設多少?** continuous 的 40 步是為了湊 120 h;
   snapshot 沒有物理時間,40 次迭代只是「40 次算子作用」。你要的 n 範圍是多少?
3. **`report_intensity_ladder_zh.md` 要不要一起處理?** 它的主表是 hour 24,不受影響,
   但它整份也是 continuous 資料;若日後要宣稱「這是 semi-linear 的結果」,措辭要一致。
