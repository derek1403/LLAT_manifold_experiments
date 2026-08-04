# 交接:snapshot mode 的 base state 錯誤(2026-08-04)

這個目錄記錄一次方法學層級的除錯:`driver._run_snapshot` 有兩個獨立缺陷,使得
**所有 snapshot run 的 δ 有約 98% 是與強迫無關的漂移**。程式已修、資料已全部重跑、
結論有三條被撤回。

## 一句話版本

`_run_snapshot` 從第 2 次迭代起把 base 從 $u_0$ 換成 $\bar u = M(u_0)$,又把 11 個
prescribed 地表通道鎖到 $\bar u$(而不是 $u_0$)。前者讓迴圈退化成自由非線性積分並在每個
δ 裡留下 $M(\bar u)-\bar u$;後者讓整條鏈跑在 DLAMPty 自己預測出來的、壓平 45% 的地形上。
驗收條件很簡單:**模式是決定性的,所以 $f=0$ 必須給 $\delta \equiv 0$。** 修正前 $n=4$ 是
8.81 K,修正後是 $1.2\times10^{-4}$ K。

## 怎麼讀

| 文件 | 內容 |
| --- | --- |
| [01_the_bug.md](01_the_bug.md) | 兩個缺陷的程式碼位置、代數推導、為什麼舊的自辯不成立 |
| [02_evidence.md](02_evidence.md) | 全部實測數字,每一項都附可重跑的指令 |
| [03_what_changed.md](03_what_changed.md) | 動過的檔案、撤回的結論、新的結論 |
| [04_open_items.md](04_open_items.md) | 還沒做的事,含優先序與成本 |

## 這次除錯的起點

使用者對 $f=0$ 的情況做了純數學推導:若 $u'_{n+1} = M(u'_n + u_0 + f) - \bar u$ 且
$\bar u = M(u_0)$,則 $f=0$ 時 $u'_1 = \bar u - \bar u = 0$,第 2 次迭代的輸入還是 $u_0$,
所以 $u'_2 = 0$,以此類推**每一步都必須嚴格為 0**。既然實測不是 0,程式一定在某處把 base
換掉了。

這個推導完全正確,而且直接命中 [driver.py:183](../../../src/llat_manifold/driver.py)。
順著它往下查才發現第二個缺陷(lock reference),那個是原本的推導沒有預期到的 ——
它讓 $\delta_1$ 就已經不是零(0.82 K),而不是從 $n=2$ 才開始。

**教訓:守恆/零解檢驗要當成迴歸測試寫進去,不要只在懷疑時才手動跑一次。**
`tests/test_offline.py` 原本有 `test_snapshot_math`,但它用 identity $M$ —— 此時
$\bar u = u_0$、模式輸出的靜態通道也等於輸入,**兩個缺陷同時被這一個巧合遮住**。
