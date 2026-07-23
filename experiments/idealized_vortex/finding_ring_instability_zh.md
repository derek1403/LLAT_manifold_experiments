# 發現整理:LLAT 中渦度環的正壓不穩定 —— 橢圓變形有、rollup 沒有

> 非正式研究筆記(中文)。實驗設計與指令見 [`README.md`](README.md);
> 產圖程式:`src/llat_manifold/idealized_vortex/{azimuthal,ellipse}.py`。
> 日期:2026-07-17。背景:202518W RAGASA `2025091700` 分析場均勻化後的安靜環境
> (風=0、熱力=域平均剖面、uniform ocean、保留 f 的 β)。
> 時間軸慣例:iteration $n$ = 名義 3 h(**非嚴格物理時間**,同 ITCZ 專案鐵律);
> 「day」= $n\times0.125$ 僅供直覺。

## 一句話結論

**LLAT 掌握了環正壓不穩定的「入口」—— m=2 橢圓變形長出來,而且配上與
Kuo et al. (2016) 簡化模式一致的自洽次環流(長軸端輻合上升、短軸側切向風極大、
質量場同步橢圓化);但它不讓不穩定走完 —— 所有環在 day 2–3 內被收編成單極
(monopole)並當成颱風增強,沒有出現離散 mesovortex。**
「線性階段在、非線性結局被吸子接管」是目前對 manifold 最精確的描述;
成長機制是否真為正壓(而非濕過程搶跑)尚未閉環,判別實驗是 q-lock(見文末)。

## 實驗設計摘要

- **問題**:眼牆尺度的渦度環在 0.25° 下只有 2–3 格,不可解析;利用正壓動力
  scale-free 的性質把環放大(核心直徑 8–40 格),看 LLAT 是否重現
  Guinn & Schubert (1994) / Schubert et al. (1999) 一系的環破裂動力,
  波數選擇對照 Hendricks et al. (2009) 的 γ–δ 相圖。
- **方法**:snapshot semi-linear(凍結時間 re-centering,$u'_i=M(\bar u+u'_{i-1})-\bar u$,
  與 ITCZ 專案同構);環放在擾動 $u'$ 裡,背景 $\bar u$ 是安靜環境。
- **初始環**(`ring.py`+`balance.py`):三區平滑 ζ(r)(hollowness γ、thickness δ=r1/r2)
  → V(r) 積分 + 外圍 taper(環流關在域內)→ 逐層 gradient-wind $\Phi'$ +
  靜力一致暖心 $T'$ + msl′;垂直用類颱風衰減剖面 F(p)(850 以下=1,150 hPa→0)。
  **保 V 不保 ζ**:V_max=35 m/s 固定,環放大 → ζ_ring∝1/r 變小 → 成長變慢。
  `zeta_bands` 選項可排任意「強-弱-強」或正負變號的多帶剖面(Rayleigh 判準的變號)。
  平衡驗證:V_t 精確落在梯度風上(`ic_check/wind_balance_ideal.png`;
  對照組用強盛期 RAGASA `2025092000`,850/500 hPa 同樣 LLAT≈梯度風)。
- **邊界**:擾動外框 8 格每步釘回背景(`boundary_relax`);對照組確認訊號非邊界人造物。
- **run 長度**:首輪 80 iters(10 名義天);**融合在 day 3 前完成,故新 base 已改 40 iters(5 天),
  面板圖改密集畫 day 0–3**。

## 怎麼讀 `growth_Am.png`(方位角波數成長圖)

這張圖的量 $A_m$ 是這樣算的(`azimuthal.py`):

1. 每個 iteration 取 850 hPa 的擾動渦度 $\zeta'(x,y)$(由 $u',v'$ 直接算,渦度線性所以合法);
2. 對域中心(=環中心)內插到極座標 $\zeta'(r,\theta)$;
3. 沿 $\theta$ 做 FFT → 各方位角波數 m 的振幅 $A_m(r)$;
4. 在「環半徑帶」($0.5\text{–}1.5\times$ rmw)取最大 → 一條 $A_m(n)$ 曲線。

判讀規則:

- **m=0(黑虛線)= 環本身**(軸對稱部分)。它先掉(重力波調整+環被抹平)再回升(單極增強)。
- **m≥1 = 非對稱擾動**。半對數圖上的**直線段 = 指數成長**,斜率就是成長率
  $\sigma$(圖例的 σ/day 是對該段的擬合;擬合窗自動選在飽和前)。
  **正壓不穩定的訊號 = 某個 m≥2 出現乾淨的指數段、且其 σ 領先其他 m**。
- **m=1 要小心**:它主要是**渦心位移**(β-gyre / β-drift、中心 wobble)在波數空間的投影,
  不是環破裂;判讀不穩定看 m≥2。
- 曲線群一起走平 = 非線性飽和(base run 在 day ~4);之後的排序沒有動力意義。
- 起始值不是 0:初始就有 ~2–3% 的 m=2/m=4 網格+球面度規印記(0.25° 下環帶只有幾格,
  這是本質的),它們是不穩定的種子之一 —— noise0 對照組證明成長率與人工噪音無關。

以 base run(γ0.2, δ0.7, rmw2.5°)為例:day 0–1 全體衰減(調整期)→ day 1–3.5
指數段(m=1 最快、m=2 σ≈0.94/day、m=3、m=4 依序)→ day 4 飽和。
這是「低波數階梯」而非理論預期的 m≈4–5 峰值 —— 見發現 1。

## 發現 1:普遍的「單極漏斗」—— 七組 IC 殊途同歸

![base run ζ′ 演化](../../outputs/idealized_vortex/ring_snapshot/ring_g0p2_d0p7_rmw2p5_init2025091700/plots/zeta850_maps.png)

*(base run ζ′(850) 演化。所有 run 的同款圖在各自 `plots/zeta850_maps.png`。)*

![rmw4° run ζ′ 演化](../../outputs/idealized_vortex/ring_scale_sweep/ring_rmw4deg_init2025091700/plots/zeta850_maps.png)

*(rmw4° run ζ′(850) 演化。取出來看分布 。)*


| run | 設計 | 結局 | σ(m=2)/day |
|---|---|---|---|
| base(rmw2.5°, δ0.7) | 環帶 3 格 | day1–2 抹成單極→增強→β-drift | 0.94 |
| noise0 | 噪音種子關掉 | 與 base 幾乎相同 | 0.93(**成長內稟,與噪音無關**) |
| norelax | 邊界鬆弛關掉 | 更慢、更亂 | —(**訊號非邊界人造物**) |
| rmw4° | 環帶 5 格 | day1.6 **m=2 橢圓** → day2.4 塌縮單極 | 0.04 |
| big_thick(rmw5°, δ0.5) | 環帶 10 格 | day1.6 **最清楚的 m=2 橢圓** → 單極 | 0.14 |
| bands_double(強弱強雙環) | dζ/dr 變號×4 | 雙環先糊成盤 → 橢圓(兩端聚積)→ 單極 | 0.19 |
| bands_negative(正環+負裙帶) | ζ 真變號 | **環活最久、幾乎不長 m≥1、安靜填成單極** | ~0 |

- **沒有任何一組走完 rollup → 離散 mesovortex → 合併** 的教科書生命史;
  單極化(理論 mixing 的終態)不經過中間態直接發生。
- 解析度已排除:環帶 3 格→10 格結局相同。
- **成長率隨環放大而變慢的方向正確**(保 V ⇒ ζ_ring 減半,σ 應減半),
  但掉得比線性比例快(2.5°→5°:0.94→0.14,~6.7 倍)—— 塌縮動力搶走主導權,
  σ 不能直接拿去對 dry theory 的 $\sigma\propto\bar\zeta$ 檢驗。
- **bands_negative 是最反理論的一組**:乾正壓理論裡 ζ 變號越強越不穩定
  (Rayleigh 必要條件被加強;cf. Kossin & Schubert 2001 的帶負裙剖面),
  LLAT 卻是它最穩定、最軸對稱 —— 目前最乾淨的一條 manifold 落差證據。

## 發現 2:橢圓期的次環流動力自洽 —— 與 Kuo et al. (2016) 對得上

Kuo, Cheng et al. (2016, *JGR-Atmos.*, [doi:10.1002/2016JD025317](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JD025317)),
*Deep convection in elliptical and polygonal eyewalls of tropical cyclones*:
無輻散正壓模式(自由大氣)+ 非對稱 slab 邊界層,結論是**橢圓眼牆的 WN2 深對流
坐在長軸兩端**,機制是非圓渦漩的非對稱氣壓場在邊界層驅動較強的輻合徑向風。
我們用全物理 LLAT 跑出來的橢圓期(big_thick,iter 13 / day 1.6,長軸近東西向)
逐項對照:

![−ω 面板](../../outputs/idealized_vortex/ring_scale_sweep/ring_rmw5_thick_init2025091700/plots/w850_maps.png)

*(shading = $-\omega$(850),>0 = 上升;黑線 = z(850) 等值線;綠虛線 = ζ 橢圓。
DLAMPty 的 w 通道是 ω(Pa/s,ERA5 慣例)。外圍方框黑線是邊界鬆弛框的 z 等值線,cosmetic。)*

![軸剖面(Kuo 2016 Fig. 6 讀法)](../../outputs/idealized_vortex/ring_scale_sweep/ring_rmw5_thick_init2025091700/plots/longaxis_profile.png)

*(m=2 相位定出長軸;紅實線 = 長軸(對向兩端平均)、藍虛線 = 短軸;
$V_r$ 外流為正、$V_t$ 氣旋式為正。)*

| Kuo 2016 的預期 | LLAT(big_thick iter 13) | 符合? |
|---|---|---|
| 長軸端徑向輻合最強 | $V_r$ 長軸入流峰 **−9 m/s**(r≈4.6°),短軸只 −4 | ✓ |
| WN2 上升在長軸兩端 | $-\omega$ 長軸峰 **0.71 Pa/s**(r≈2.9°)> 短軸 0.5;−ω 面板 iter13 東西兩端紅 | ✓ |
| 切向風極大在短軸側 | $V_t$(850) 短軸 **31 m/s** vs 長軸 19;300 hPa 面板亮區沿短軸翼 | ✓ |
| 質量場橢圓 | z(850) 等值線與 ζ 橢圓同形同向 | ✓ |

rmw4° 與 bands_double 的同款圖(各自 `plots/{w850_maps,spd300_maps,longaxis_profile}.png`)
定性相同。**這證明 LLAT 長出的 m=2 不是雜訊,而是一個帶正確次環流的動力結構**
—— 連「正壓渦漩動力 → 邊界層輻合 → 對流位置」這條跨過程的鏈都是對的。

## 所以,LLAT 重現了正壓不穩定嗎?—— 分層回答

1. **線性/早期階段:是。** 厚環(δ=0.5)長出 m=2,落在 Schubert/Hendricks
   相圖「厚環 → 低 m」的正確區間;橢圓的次環流與 Kuo 2016 的正壓+邊界層
   理論逐項吻合;成長與人工噪音無關(noise0)、與邊界無關(norelax)、
   成長率隨 ζ_ring 減小而變慢(方向正確)。
2. **非線性生命史:否。** 沒有離散 mesovortex、沒有 vortex merger、
   沒有 polygonal 階段;m=2 一成形就被快速軸對稱化+增強接管,
   day 3 後只剩單極颱風 + β-drift。
3. **歸因:未閉環。** LLAT 是全物理濕模式:q 自由、加熱回饋開著。
   目前不能排除「成長段其實由濕過程主導、只是空間結構像正壓模態」。
   要說「重現了正壓不穩定」,還缺兩塊:
   - **q-lock 對照**(`lock_upper_vars: [q]`,共用 driver 已有機制):
     封鎖濕通道逼近乾動力 —— 若橢圓照長甚至走完 rollup,機制是正壓的、
     單極化是濕吸子搶跑;若 m=2 消失,連橢圓都是濕過程的產物。
   - **正壓能量轉換診斷**:eddy 動能收支的 $-\overline{u'v'}\,\partial\bar V/\partial r$
     項,直接量「非對稱從基流剪切抽能量」的正壓轉換率(可加進 `azimuthal.py`)。

## 限制與待驗證

- 以上全部基於單一背景(0917 均勻化)、單一 V_max(35 m/s)、snapshot 模式。
- σ 的定量還不能對 dry theory:擬合窗短、m=1 位移訊號污染、塌縮動力疊加。
- ω 慣例假設 DLAMPty w 通道 = ERA5 ω(Pa/s);量級(~1 Pa/s)與結構支持此讀法,未獨立驗證。
- 300 hPa 在飽和振幅後出現格狀 artifact(模式自身),讀上層圖時避開 day 6+。
- δ-sweep(0.55/0.85)、rmw 1–3° configs 就緒未跑;在 q-lock 出結果前預期只重複同一漏斗。

## 圖檔索引

- 每個 run:`plots/zeta850_maps.png`(ζ′ 演化,day0–3 密集)、`plots/growth_Am.png`、
  `plots/spectrum_final.png`;橢圓 run 另有 `plots/{w850_maps,spd300_maps,longaxis_profile}.png`。
- IC 驗證:`outputs/idealized_vortex/ic_check/`(r–z ζ/V_t/θ vs RAGASA 0917/0920、
  梯度風平衡)。
- 重跑診斷:`python -m llat_manifold.idealized_vortex.azimuthal <run_dir>`、
  `python -m llat_manifold.idealized_vortex.ellipse <run_dir> [--iter N]`。
