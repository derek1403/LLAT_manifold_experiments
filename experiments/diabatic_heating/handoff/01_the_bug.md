# 缺陷本體

兩個獨立的缺陷,都在 `src/llat_manifold/driver.py::_run_snapshot`,都在
commit `b8c483f` 及更早就存在。

---

## 缺陷 1 — $i \ge 2$ 的 base 是 $\bar u$ 而不是 $u_0$

### 程式碼

原本的 `driver.py:183`:

```python
base_up, base_sfc = (u0_up, u0_sfc) if it == 1 else (ubar_up, ubar_sfc)

A_up = np.ascontiguousarray(base_up + d_up + f_up, dtype=np.float32)
```

這不是筆誤。`driver.py` 的模組 docstring 與 `docs/perturbation_method.md` §3.1 都把它
寫成設計:

$$u'_1 = M(u_0+f)-\bar u, \qquad u'_i = M(\bar u + u'_{i-1} + f) - \bar u \quad (i\ge2)$$

### 為什麼錯 —— 望遠鏡效應

$u'_{i-1}$ 是**從 $\bar u$ 量起**的,也就是 $u'_{i-1} = u_{i-1} - \bar u$。所以

$$\bar u + u'_{i-1} = \bar u + (u_{i-1} - \bar u) = u_{i-1}$$

**餵進模式的其實就是前一次的輸出。** 整個迴圈是

$$u_i = M(u_{i-1} + f_i)$$

也就是一條凍結時間、鎖住靜態通道的**自由非線性積分**。它完全不是「同一個算子的重複迭代」,
而 $u'_i = u_i - \bar u$ 只是「軌跡減掉一個常數參考場」。所謂的 power iteration 從第 2 步起
就不存在了。

### 為什麼錯 —— 常數項

把 $M(x) = \mathbf{J}x$ 代進去比較兩種寫法:

| | 遞迴式 |
| --- | --- |
| 舊 | $u'_i = \mathbf{J}(\bar u + u'_{i-1} + f) - \bar u = \mathbf{J}u'_{i-1} + \underbrace{(\mathbf{J}-\mathbf{I})\bar u}_{=\,M(\bar u)-\bar u} + \mathbf{J}f$ |
| 新 | $u'_i = \mathbf{J}(u_0 + u'_{i-1} + f) - \mathbf{J}u_0 = \mathbf{J}u'_{i-1} + \mathbf{J}f$ |

舊式多出來的 $M(\bar u) - \bar u$ 就是 **$\bar u$ 不是 $M$ 的不動點**這件事。它的性質最毒:

- **與 $f$ 無關** → 也就與 $f$ 的**正負號**無關;
- 所以它原封不動地活進 $\tfrac12[\delta(+A)+\delta(-A)]$,
  **±A 反對稱檢驗會把它整包讀成「非線性」**;
- 而且它會被後續迭代放大,量級遠大於響應本身。

`docs/perturbation_method.md` §3.1 原本用「若 $M$ 是線性的,(3.1) 就是 forced power
iteration $u'_i = \mathbf{J}u'_{i-1} + (\text{const})$」來自辯。這個自辯站不住:它沒說那個
const 裡有 $(\mathbf{J}-\mathbf{I})\bar u$,而該項在實測中比強迫項大得多。
**修正後的程式碼才真的做到文件宣稱的事。**

---

## 缺陷 2 — 11 個 prescribed 通道鎖到 $\bar u$ 而不是 $u_0$

### 程式碼

原本的 `driver.py:171 / 187 / 190`:

```python
ubar_up, ubar_sfc = m_operator(u0_up, u0_sfc, dlampty, advance_time=False)   # 沒有 lock
...
_align_statics(A_sfc, ubar_sfc, active_idx)                                  # 對齊到 ubar
u_up, u_sfc = m_operator(A_up, A_sfc, dlampty, advance_time=False,
                         lock_ref=ubar_sfc, lock_idx=active_idx)             # 鎖到 ubar
```

`active_idx` 是 11 個通道:9 個 `static_surface_vars`(sst_filled, f, solar, hgt,
landmask, diurnal_sin/cos, doy_sin/cos)+ lat + lon。

### 為什麼錯

DLAMPty 會連同其他變數一起「預測」這些通道,而且預測得很糟。實測 $u_0$ vs $\bar u = M(u_0)$:

| 通道 | $u_0$ | $\bar u$ | 後果 |
| --- | --- | --- | --- |
| `hgt` | 0 – **1899 m** | −4.9 – **1041 m** | 地形被壓平 45% |
| `landmask` | 0 / 1 | −0.064 – 1.052 | 不再是二元 |
| `solar` | 772 – 901 | 915 – 953 | 換了一個時刻 |
| `diurnal_sin` | 0.707 – 0.966 | −0.004 – 0.502 | 相位前進 3 h |
| `diurnal_cos` | −0.707 – −0.259 | −1.013 – −0.861 | 同上 |
| `sst_filled` | 297.15 – 303.67 | 296.87 – 303.62 | 最大差 5.7 K |
| `lat` | 6.5 – 26.5 | +0.015° | 網格位移 |
| `lon` | 119.5 – 139.5 | **−0.079°** | 網格位移 |

把這些採納成往後每次迭代的下邊界,等於**整條鏈跑在一個壓平的、海岸線糊掉的、時刻不對的
世界**上。而且 `operators.py` 說 `advance_time=False` 的意思是「凍結在一個 valid time」,
實際上是凍結在 $t_0+3$h 的模式預測值,不是 $t_0$。

### 根因:`advance_time=False` 把還原機制一起跳過了

`m_operator` 對 `advance_time=True` 會呼叫 `changing_additional_information`,
**那才是這些 prescribed 通道一直以來被還原的機制**。實測 continuous 的 control:
120 h 之後 `landmask` 仍嚴格是 0/1、`diurnal_sin` 正常走完日夜週期 —— 完全沒有劣化。

`advance_time=False` 的本意是「凍結時間」,但實際效果是**連唯一會還原這些通道的那一步
也跳過了**,而 snapshot 路徑沒有任何替代品。所以模式對這些通道的糟糕輸出只在 snapshot 下
外露 —— 這解釋了它為什麼能存活這麼久,也是為什麼 continuous 不需要同樣的修正。

**日後新增任何 `advance_time=False` 的路徑,都必須自己帶 lock reference。**

### 這是 $\delta_1 \ne 0$ 的原因

使用者的推導預期 $\delta_1 = 0$(第 1 次迭代 base 本來就是 $u_0$)。實測卻是 0.82 K。
原因就是 `_align_statics(A_sfc, ubar_sfc, ...)` 在 `it == 1` 時**也會執行**,把 $u_0$ 的
11 個正確通道換成 $\bar u$ 的錯誤版本 —— 所以模式看到的輸入從第一步就不是 $u_0$。

**兩個缺陷必須一起修**,只修其中一個都不會讓空跑歸零。

---

## 修正後

```python
ubar_up, ubar_sfc = m_operator(u0_up, u0_sfc, dlampty, advance_time=False,
                               lock_ref=u0_sfc, lock_idx=active_idx)
...
A_up = np.ascontiguousarray(u0_up + d_up + f_up, dtype=np.float32)
A_sfc = np.ascontiguousarray(u0_sfc + d_sfc + f_sfc, dtype=np.float32)
_align_statics(A_sfc, u0_sfc, active_idx)
u_up, u_sfc = m_operator(A_up, A_sfc, dlampty, advance_time=False,
                         lock_ref=u0_sfc, lock_idx=active_idx)
```

$$\boxed{u'_i = M(u_0 + u'_{i-1} + f) - M(u_0)}$$

$\bar u$ 仍然存在、仍然是所有 δ 的參考場、仍然是 `background.npz` —— 只是不再被拿來當
base,也不再被拿來當 lock reference。修正後 $\bar u$ 的 11 個通道與 $u_0$ **逐位元相同**
(最大差 0.0),地形回到 1899 m,經度回到 119.5–139.5。

---

## 為什麼測試沒抓到

`tests/test_offline.py::test_snapshot_math` 用 `_IdentityDLAMPty`(M = identity)。此時:

- $\bar u = M(u_0) = u_0$ → **缺陷 1 不可見**(base 用哪個都一樣);
- 模式輸出的靜態通道 = 輸入 → **缺陷 2 不可見**(鎖到哪個都一樣)。

**一個巧合同時遮住兩個缺陷。** 新增的 `_DriftingDLAMPty`(`up → 0.9·up + 0.1`,
`sfc → sfc + 1`)兩個性質都打破,三個新測試各自釘住一件事:

| 測試 | 釘住什麼 |
| --- | --- |
| `test_snapshot_null_run_is_exactly_zero` | $f=0 \Rightarrow \delta \equiv 0$(缺陷 1+2 的總驗收) |
| `test_snapshot_statics_pinned_to_initial_state` | $\bar u$ 的 prescribed 通道 = $u_0$ 的(缺陷 2) |
| `test_snapshot_recursion_uses_u0_as_base` | $\delta_3 = 2.439f$ 的閉式解(缺陷 1) |
