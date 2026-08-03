# myplot — 補充圖組

> **這份文件的定位。** [`report_intensity_ladder_zh.md`](report_intensity_ladder_zh.md)
> 從頭到尾畫的都是 **ΔPV(響應)**:同一份加熱打進不同強度的渦旋,每一格看的是
> perturbed − control。這份文件放的是**絕對場** —— 被擾動的那些渦旋自己長什麼樣。
> 兩者相差一個數量級以上(絕對 PV 幾個 PVU,ΔPV 零點幾個 PVU),**數字不可互相比較**。
>
> 用的是同一批軸對稱 IC(`outputs/axisymmetric_ic/`),所以這裡看到的結構就是報告裡
> 那些響應真正作用在上面的底圖。程式在 [`src/myplot/`](src/myplot/),圖在
> [`figs/myplot/`](figs/myplot/)。個案:202518W RAGASA。

---

## 圖 1 — 0920 / 0921 軸對稱絕對 PV 的 r–p 剖面

![MP1 軸對稱絕對 PV](figs/myplot/mp1_axisym_pv.png)

程式:[`src/myplot/fig_MP1_axisym_pv.py`](src/myplot/fig_MP1_axisym_pv.py)

**畫的是什麼。** 兩個階梯成員在 **n = 0**(注入前)的方位平均 Ertel PV,半徑 0–800 km、
壓力 1000–200 hPa,兩 panel 共用同一組色階(0–8 PVU)所以可以直接對比。疊加:

- **青線** = PV 等值線(每 1 PVU,每 2 PVU 標值)—— 就是色階本身的等值線,讓塔的形狀
  可以直接讀數字,不必只靠顏色;
- **黑線** = 等熵線 θ(每 4 K)—— 看暖心;
- **灰虛線** = 850 hPa 的 RMW。

**panel 標題用的是觀測強度分級**(Category 1 / Category 4,best-track 在該分析時刻的等級),
**不是** IC 自己的 peak $V_t$。兩者差很多:0.25° 網格上做方位平均會把眼牆抹平,所以
Category 4 的 0921 在 IC 裡只剩 33.9 m/s。這是刻意的 —— 標題標的是「這是哪個階段的颱風」,
渦旋自己的風速數字在下面表格裡。

**怎麼做的。** PV 用既有的 `diagnostics/_idealized.calculate_pv_spherical`(球面 Ertel PV,
PVU)算在完整三維場上,再用 `idealized_vortex.axisymmetric` 的 `radial_frame` / `radial_mean`
做方位平均 —— **這就是當初造出這些 IC 的同一個平均算子**,所以對這批場是精確的,而且與
`rz_sections` 回傳的 θ / $V_t$ 落在同一組半徑格點上,三個場共用一根 r 軸,不必內插。
沒有重寫任何物理。

### 怎麼讀

| | IC 的 peak $V_t$ / RMW | PV 最大值 | 850 hPa 核心($r<100$ km 平均) |
| --- | --- | --- | --- |
| **Category 1(0920 00Z)** | 24.7 m/s / 173 km | **4.45 PVU** @ 500 hPa($r\approx13$ km) | 2.20 PVU |
| **Category 4(0921 00Z)** | 33.9 m/s / 147 km | **7.25 PVU** @ 300 hPa($r\approx40$ km) | 3.25 PVU |

1. **PV 塔不只變強,峰值還往上抬。** 0920 的最大值在 **500 hPa**,0921 抬到 **300 hPa**,
   而且 300 hPa 的值從 3.58 → 7.25 PVU(翻一倍)。這不是「同一個塔按比例放大」,
   是塔的重心上移。
2. **低層同步收緊。** 850 hPa 核心 2.20 → 3.25 PVU,RMW 173 → 147 km,PV 高值區的
   徑向寬度明顯變窄 —— 對照報告 L1(a) 那條 $V_t(r)$ 曲線,這是它背後的垂直結構。
3. **環境完全相同。** 兩個 panel 的 f、SST、lat/lon 是同一份(0920 的),所以左右兩張圖的
   **所有差異都來自渦旋本身**。這正是階梯設計要的受控條件。
4. **PV 核心比 RMW 更靠內。** 走到 850 hPa RMW(灰虛線)那條半徑時,PV 已經只剩
   **1.0–1.6 PVU**(0920)/ **1.5–2.2 PVU**(0921,850/500/300 hPa 三層),遠低於核心的
   4.45 / 7.25 PVU —— **平均風速最大的半徑不是 PV 最集中的半徑**,PV 高值區明顯更靠內。
   等熵線同時在核心上凸(暖心),這批 IC 在 n = 0 是自洽的渦旋,不是被平均壞掉的東西。

> **一個順手看到、但還不下結論的觀察。** 把 `--inits` 換成四個成員跑一次,PV 塔最大值是
> **1.90 → 4.45 → 7.25 → 6.54 PVU**(0917 / 0920 / 0921 / 0922)—— **也在 0921 達峰、
> 0922 翻轉**,與報告裡 ΔPV 響應的 rollover 同向。這很有意思,但它是**絕對場**的性質,
> 與響應的 rollover 是兩件事,兩者是否同源需要另外的實驗才能說。這裡只登記現象,不解釋。
> (四成員版本不是本節的主圖,要看就跑下面第二條指令,會另存成 `mp1_axisym_pv_ladder4.png`。)

### 必讀的 caveat

1. **這是絕對 PV,不是 ΔPV。** 報告表格裡的 ±0.1–0.8 PVU 是「加熱打進去多出來的那一點」,
   這張圖的 2–7 PVU 是渦旋本體。**不要放在同一句話裡比大小。**
2. **0921 用的是階梯 IC**(`axisym_202518W_2025092100_env2025092000.npz`),它帶的是
   **0920 的科氏力**(f = 4.142e-05,16.5°N),不是 0921 自己的 18.25°N。核心 ζ ~ 7e-4
   遠大於 f ~ 4e-5,**PV 塔本身幾乎不受影響**;但 $r\gtrsim500$ km 的外圍 ζ → 0,那裡的
   PV 帶有 f 的 ~10% 偏差 —— **外圍數值不要拿去做定量宣稱**。
3. **天花板 200 hPa**(`_idealized.P_TOP_HPA`)。再往上模式層變稀、模式自身誤差主導,
   全 repo 的垂直剖面都停在這裡。
4. **軸對稱只在 n = 0 嚴格成立。** `sfc[9:]`(SST / f / 地形 / 海陸)保留真實的,下邊界
   不對稱,從第一步就把對稱重新烙回。這張圖畫的正好是 n = 0 —— **所以這是唯一一個
   「方位平均剖面 = 完整場」的時刻**,不是摘要。
5. **徑向 bin = 一個網格間距(26.7 km),最內圈中心在 r = 13.3 km。** 0920 的 PV 最大值
   就落在這第一圈上,格點數少;$r<30$ km 的細節不要過度解讀。0921 的最大值在 r ≈ 40 km,
   受這個限制較小。

### 重現

```bash
conda activate pangu_env
cd /wk2/pc/AI_models/LLAT_manifold_experiments/experiments/diabatic_heating/src/myplot

# 預設:0920 + 0921 → figs/myplot/mp1_axisym_pv.png
python fig_MP1_axisym_pv.py

# 整條階梯(0917/0920/0921/0922),自動排成四個 panel
python fig_MP1_axisym_pv.py --inits 2025091700 2025092000 2025092100 2025092200 \
    --out ../../figs/myplot/mp1_axisym_pv_ladder4.png

# 其他旗標:--rmax 800(半徑上限 km)、--style paper(去掉畫布上的圖說並多出一份 PDF)
```

腳本會把每個成員的 `peak V_t / RMW / max PV / 最大值所在壓力層` 印在 stdout,
上面表格裡的數字都可以這樣覆核。
