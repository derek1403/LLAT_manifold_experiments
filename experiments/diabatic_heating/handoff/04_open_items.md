# 還沒做的事

按我認為的優先序排。成本都以本機實測的 **35 秒/run(40 迭代 snapshot)** 換算。

---

## 1. 其他 snapshot 家族尚未重跑 —— 優先

修正在 `driver.py`,**對所有 snapshot run 一體適用**,但這次只重跑了
`diabatic_heating_axisym/heating_moist`。仍是修正前產生的:

| 位置 | 數量 | 說明 |
| --- | ---: | --- |
| `outputs/idealized_vortex/ring_*` | 6 | ring 系列的 snapshot run |
| `outputs/diabatic_heating/snapshot/` | 3 | 非軸對稱的 5K/10K iter20 |

**如果任何論文或報告引用這些,它們也有 98% 是漂移。** 成本很低(9 個 run ≈ 5 分鐘),
建議直接重跑。要注意 ring 系列用的是 `driver_vortex` 路徑,得確認它走的是同一個
`_run_snapshot`(`tests/test_vortex_offline.py::test_driver_vortex_bookkeeping` 顯示是)。

---

## 2. `_run_continuous` —— 查過了,**沒有**缺陷 2,不用改

我原本懷疑 continuous 也有同型問題:control 那次 `m_operator` 沒有 `lock_ref`,而
`I ← Ip` 每步覆寫,所以 prescribed 通道應該逐步劣化。**實測顯示沒有:**

| lead | `hgt` 值域 | `landmask` 值域 | `diurnal_sin` 值域 |
| ---: | --- | --- | --- |
| 3 h | 0 – 1842.4 | 0.000 – 1.000 | −0.000 – 0.500 |
| 24 h | 0 – 1939.8 | 0.000 – 1.000 | 0.707 – 0.966 |
| 60 h | 0 – 1889.2 | 0.000 – 1.000 | −0.966 – −0.866 |
| 120 h | 0 – 2005.2 | 0.000 – 1.000 | 0.866 – 0.966 |

`landmask` 始終**嚴格二元**、`diurnal_sin` 正常走完日夜週期、`hgt` 的變動是
TC-following domain 移到不同地形上方造成的(那是已知的 domain 漂移,不是劣化)。

**原因就是 `advance_time=True` 會呼叫 `changing_additional_information`**,它把 prescribed
通道重新產生 —— 這才是這些通道一直以來被還原的機制。

### 這同時就是缺陷 2 的根因

`operators.m_operator` 對 `advance_time=False` 的處理是「**不**呼叫
`changing_additional_information`」。設計者的本意是「凍結時間」,但實際效果是
**把唯一會還原 prescribed 通道的那一步也一起跳過了**,而 snapshot 路徑沒有任何替代品。
所以模式對這些通道的糟糕輸出,只在 snapshot 下才會外露 —— 這也解釋了為什麼它能存活這麼久。

修正後 `lock_ref=u0_sfc` 就是那個替代品。**若日後新增任何 `advance_time=False` 的路徑,
必須自己帶 lock reference。**

---

## 3. `report_intensity_ladder_zh.md` 尚未更新

整份建立在 `continuous` 的 24 h 掃描上。**主表(hour 24)不受這次兩個缺陷影響**
(它不經過 `_run_snapshot`),而且該時刻核心陸地 0%、domain 位移 < 1°,所以數字本身可用。

但若要宣稱「這是 semi-linear / power iteration 的結果」,措辭與資料都要換成 snapshot。
成本是 11 家族 × 4 強度的完整重掃 —— 這是本清單裡最大的一項,**建議先確認報告要不要用
snapshot 敘事,再決定跑不跑**。

---

## 4. 0 K 底噪只有 `heating_moist` 兩個成員有

其他家族若要引用「訊號高於底噪」的說法,各自需要一個 0 K run(35 秒)。
注意這**不再是為了相減**,只是為了知道底線在哪。

---

## 5. 舊 `continuous` run 的去留

`outputs/diabatic_heating_axisym/heating_moist/` 下仍有大量 `tseries_*_120h_*` 與
`sweep_*_24h_*`。它們回答的是另一個(也合法的)問題:「加熱對一條會移動、會登陸的軌跡
有什麼影響」。要保留當對照還是刪掉,待決定。目錄總計 19 GB。

---

## 6. 值得考慮的方法學延伸(非必要)

- **重正規化。** 現在的 power iteration 沒有每步重正規化 δ,所以 $n$ 大時 δ 會長到
  有限振幅、finite-difference 不再近似 Jacobian。若目標是萃取領先奇異向量,標準做法是
  每步把 $\lVert\delta\rVert$ 拉回固定值並記錄成長率。目前的做法(讓它長)適合回答
  「有限振幅的加熱會演化成什麼」,不適合回答「領先模態長什麼樣」。**兩個問題不同,
  要先決定研究問的是哪一個。**
- **底噪隨 $n$ 成長**(每步約 ×1.2)這件事本身就是一個可報告的量:它是模式在這個狀態附近的
  局部 Lyapunov 成長率的下界估計,而且是用「什麼都不做」量出來的。
