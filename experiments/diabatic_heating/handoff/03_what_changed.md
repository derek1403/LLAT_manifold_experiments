# 動過什麼、結論變了什麼

---

## A. 程式碼

| 檔案 | 動作 |
| --- | --- |
| `src/llat_manifold/driver.py` | `_run_snapshot`:base 恆為 $u_0$;`_align_statics` 與 `lock_ref` 改對 $u_0$;$\bar u$ 本身也在 lock 下計算。模組 docstring 重寫 |
| `docs/perturbation_method.md` | §3.1 遞迴式改寫;新增 §3.1.1 說明改了什麼、為什麼、驗收數字 |
| `tests/test_offline.py` | 新增 `_DriftingDLAMPty` 與三個測試,並加進 `__main__` |
| `experiments/diabatic_heating/src/myplot/_snap.py` | `dpv_field`/`panels` 的 `subtract_null` 預設 `True → False`;新增 `noise_floor()`;`null_run()` 語意從「要扣掉的漂移」改為「雜訊底噪」;模組 docstring 更新 |
| `.../fig_MP5_pv_budget.py` | 移除環流的 null 相減 |
| `.../fig_MP6_state_evolution.py` | 四個 field 都不再減 `z`;`--field null` 語意改為底噪 |
| `.../fig_MP7_domain_drift.py` | docstring 與 caption 改為「鎖到 $u_0$」,並說明為什麼鎖到模式輸出也會給 0.00° |
| `experiments/diabatic_heating/myplot.md` | §0-2 整節重寫;圖 2/3/4/5 的數字表全部更新;重現指令補上 `--tag-prefix tseries --family heating_moist` |

> `_run_continuous` **沒有動**。但它有缺陷 2 的同型問題:control 那次 `m_operator`
> 沒有 `lock_ref`,所以 control 自己的 prescribed 通道逐步劣化,40 步之後地形與 land mask
> 已經嚴重失真。見 [04_open_items.md](04_open_items.md)。

---

## B. 資料

- **重跑 28 個 run**:14 個振幅(0, ±1, ±2, ±5, 3, 4, 6, 7, 8, 9, 10 K)× 2 成員
  (0920 / 0921,兩者都用 0920 的環境)。每個 40 迭代、強迫 spread 到前 8 步。
  **約 35 秒/run,全部約 17 分鐘。**
- **舊的 28 個 run 移到** `outputs/diabatic_heating_axisym/_archive_snapshot_pre_u0fix/`
  (4.7 GB),附 `README.md` 說明為什麼不能用。
- **MP1–MP7 全部重繪**。

### 一個會咬人的細節

`run_amp_sweep.py` 的 `--tag-prefix` 預設是 `sweep`,但 `_snap.py` 找的是
`tseries_{amp}_iter{iters}_init{init}`。**重跑時一定要加
`--tag-prefix tseries --family heating_moist`**,否則資料會落在
`sweep_*_iter40_*` 而所有繪圖腳本都找不到。

---

## C. 撤回的結論

### C1 「強渦旋留得住響應、弱渦旋留不住」— 早先已撤回

原因是 `continuous` mode 下 0920 的 TC-following domain 走了 16.9°、第 5 天核心 69.9% 是
陸地,而 0921 只有 5.1%。那是登陸的對比,不是渦旋強度的性質。

### C2 「$n=8$ 時 0921/0920 穩定在 0.81–0.91,較弱渦旋響應略大」— 本次撤回

那組數字建立在「舊資料 + 扣 0 K 空跑」上。新資料給的是**不同的、而且更有結構的**答案:

| | $n$=1 | $n$=2 | $n$=4 | $n$=8 | $n$=16 | $n$=24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0921/0920 響應 RMS 比 | **1.41** | 1.27 | 1.47 | 0.92 | 0.78 | 0.69 |

**注入期間(強迫還在)強渦旋響應較大;強迫停掉之後反轉,弱渦旋放大得比較快。**
這是一個轉折,不是一個常數 —— 而舊資料把它抹平成 0.81–0.91。

### C3 「Category 4 全程比 Category 1 更線性」— 本次撤回,方向相反

±A 非鏡像佔比 RMS(even)/RMS(odd):

| $n$ | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Category 1 | 7% | 9% | 14% | 19% | 23% | 70% | 151% | 141% |
| Category 4 | 5% | 10% | 20% | 27% | 66% | 136% | 259% | 416% |

舊版寫的是 Category 4 更線性($n$=40 時 70% 對 485%);新資料是**Category 1 更線性**
(141% 對 416%)。而且新曲線單調爬升、沒有突變,舊版在 $n$=2 就跳到 600%
(因為分子整包是背景漂移)。

---

## D. 新的、只有乾淨資料才看得到的結論

1. **$n=1$ 是乾淨的線性點。** 窗內 RMS 每 K 是 0.0048 / 0.0048 / 0.0047 / 0.0044
   (1/2/5/10 K),10 K 才掉 8%。到 $n=8$ 掉 31% —— 真實飽和。

2. **PV 平流假說在兩個成員上表現完全不同。** MP5 的「假說 vs 實測」空間相關:

   | $n$ | 0920 (Cat 1) | 0921 (Cat 4) |
   | ---: | ---: | ---: |
   | 1 | +0.31 | **+0.86** |
   | 2 | +0.08 | **+0.79** |
   | 4 | +0.14 | +0.59 |
   | 8 | +0.31 | +0.49 |

   **強渦旋上假說量化成立,弱渦旋上不成立。** 舊資料兩者都落在 0.16–0.51,分不出來。

3. **$q\,\partial\dot\theta/\partial\theta$ 的理論預測在 $n=1$ 站得住**:
   0920 預測 −0.11…+0.13 對實測 −0.10…+0.09;0921 預測 −0.18…+0.19 對實測 −0.19…+0.09。

---

## E. 沒有受影響的東西

- **MP1**(絕對 PV 剖面)只用 IC,與 snapshot 遞迴無關。
- **`report_intensity_ladder_zh.md` 的主表**是 `continuous` 的 hour-24 結果,
  不經過 `_run_snapshot`,所以不受這兩個缺陷影響。但它仍受
  [04_open_items.md](04_open_items.md) 第 1 項的敘事問題影響。
- `forward` mode 沒有 base state 的概念,不受影響。
