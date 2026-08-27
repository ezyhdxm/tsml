# 6. Hull--White 一因子：精确拟合今天的曲线

Hull--White 模型将均值回复 Gaussian short-rate model 扩展为 time-inhomogeneous 模型，从而精确拟合任意初始曲线 [@hullwhite1990]。

## 6.1 两种等价参数化

第一种写法是

$$
\boxed{
dr_t=[\theta(t)-ar_t]dt+\sigma dW_t^{\mathbb Q}}
$$

其中 $a>0$，$\sigma\ge0$。

更透明的写法是 deterministic shift：

$$
r_t=x_t+\phi(t),
$$

$$
\boxed{dx_t=-ax_tdt+\sigma dW_t^{\mathbb Q}},
\qquad x_0=0.
$$

两者关系为

$$
\theta(t)=\phi'(t)+a\phi(t).
$$

## 6.2 如何选择 $\phi(t)$ 拟合初始曲线

令市场初始瞬时远期曲线为

$$
f^M(0,t)=-\partial_t\log P^M(0,t).
$$

因为 $x$ 是零均值 Gaussian OU，

$$
\phi(t)
=
\boxed{
f^M(0,t)
+
\frac{\sigma^2}{2a^2}(1-e^{-at})^2
}.
$$

对应地，若使用 $dr=[\theta(t)-ar]dt+\sigma dW$ 的参数化，

$$
\boxed{
\theta(t)
=
\partial_t f^M(0,t)
+a f^M(0,t)
+
\frac{\sigma^2}{2a}(1-e^{-2at})
}.
$$

文献和软件中经常因为采用不同的 drift 参数化而出现一个 $a$ 或 $a^2$ 的差异。必须先确认 SDE 是 $dr=[\theta(t)-ar]dt$，还是 $dr=a[\theta(t)-r]dt$，再套公式。

## 6.3 债券价格

定义

$$
B_a(t,T)=\frac{1-e^{-a(T-t)}}{a}.
$$

则

$$
P(t,T)=A_{HW}(t,T)e^{-B_a(t,T)r_t},
$$

其中

$$
A_{HW}(t,T)
=
\frac{P^M(0,T)}{P^M(0,t)}
\exp\left[
B_a(t,T)f^M(0,t)
-
\frac{\sigma^2}{4a}(1-e^{-2at})B_a(t,T)^2
\right].
$$

这保证 $t=0$ 时模型完全复现 $P^M(0,T)$。

## 6.4 精确模拟与折现积分

对 shifted state $x_t$，

$$
x_{t+\Delta}
=e^{-a\Delta}x_t
+
\sigma\sqrt{\frac{1-e^{-2a\Delta}}{2a}}Z_1.
$$

区间积分

$$
J_{t,\Delta}=\int_t^{t+\Delta}x_sds
$$

满足

$$
\mathbb E_t[J_{t,\Delta}]=B_a(\Delta)x_t,
$$

$$
\operatorname{Var}_t(J_{t,\Delta})
=
\frac{\sigma^2}{a^2}
\left[
\Delta
-\frac{2(1-e^{-a\Delta})}{a}
+\frac{1-e^{-2a\Delta}}{2a}
\right],
$$

$$
\operatorname{Cov}_t(x_{t+\Delta},J_{t,\Delta})
=
\frac{\sigma^2}{2a^2}(1-e^{-a\Delta})^2.
$$

所以可以精确生成 $(x_{t+\Delta},J_{t,\Delta})$，然后

$$
\int_t^{t+\Delta}r_sds
=
J_{t,\Delta}+\int_t^{t+\Delta}\phi(s)ds.
$$

这比只精确模拟 $r_{t+\Delta}$、却用左端点近似 discount factor 更一致。

## 6.5 模型含义

$\phi(t)$ 只负责拟合今天的曲线，不增加随机因子。Hull--White 1F 仍然只有一个 Brownian shock，因此远端和近端收益率变化高度受限。它通常适合：

- 需要低维 Markov state 的 callable/Bermudan 定价；
- 树、PDE 或 Longstaff--Schwartz Monte Carlo；
- 用 cap/floor 或 swaption 的一个波动率方向校准。

它不擅长同时解释 level、slope、curvature 的独立变化，也不自然地产生 volatility smile。

# 7. G2++：两因子 Gaussian 精确拟合模型

G2++ 用两个相关 OU 因子增加曲线形变自由度：

$$
r_t=x_t+y_t+\phi(t),
$$

$$
dx_t=-ax_tdt+\sigma dW_t^{1,\mathbb Q},
$$

$$
dy_t=-by_tdt+\eta dW_t^{2,\mathbb Q},
$$

$$
d\langle W^1,W^2\rangle_t=\rho dt,
\qquad -1\le\rho\le1.
$$

参数要求 $a,b>0$、$\sigma,\eta\ge0$。

## 7.1 初始曲线 shift

定义

$$
q(t)
=
\frac{\sigma^2}{a^2}(1-e^{-at})^2
+
\frac{\eta^2}{b^2}(1-e^{-bt})^2
+
\frac{2\rho\sigma\eta}{ab}(1-e^{-at})(1-e^{-bt}).
$$

若 $x_0=y_0=0$，取

$$
\boxed{
\phi(t)=f^M(0,t)+\frac12q(t)
}
$$

即可精确拟合初始 discount curve。

## 7.2 债券价格

令

$$
B_a(\tau)=\frac{1-e^{-a\tau}}a,
\qquad
B_b(\tau)=\frac{1-e^{-b\tau}}b.
$$

条件积分方差为

$$
\begin{aligned}
\mathcal V(\tau)
={}&
\frac{\sigma^2}{a^2}
\left[
\tau-\frac{2(1-e^{-a\tau})}{a}
+\frac{1-e^{-2a\tau}}{2a}
\right]\\
&+
\frac{\eta^2}{b^2}
\left[
\tau-\frac{2(1-e^{-b\tau})}{b}
+\frac{1-e^{-2b\tau}}{2b}
\right]\\
&+
\frac{2\rho\sigma\eta}{ab}
\left[
\tau-\frac{1-e^{-a\tau}}a
-\frac{1-e^{-b\tau}}b
+\frac{1-e^{-(a+b)\tau}}{a+b}
\right].
\end{aligned}
$$

于是

$$
\boxed{
P(t,T)
=
\exp\left[
-\int_t^T\phi(s)ds
-B_a(\tau)x_t
-B_b(\tau)y_t
+\frac12\mathcal V(\tau)
\right]
}.
$$

## 7.3 精确 endpoint 模拟中的相关性细节

一步转移为

$$
x_{t+\Delta}=e^{-a\Delta}x_t+\varepsilon_x,
$$

$$
y_{t+\Delta}=e^{-b\Delta}y_t+\varepsilon_y,
$$

其中 $(\varepsilon_x,\varepsilon_y)$ 是均值为零的二维正态，协方差为

$$
\operatorname{Var}(\varepsilon_x)
=
\frac{\sigma^2(1-e^{-2a\Delta})}{2a},
$$

$$
\operatorname{Var}(\varepsilon_y)
=
\frac{\eta^2(1-e^{-2b\Delta})}{2b},
$$

$$
\operatorname{Cov}(\varepsilon_x,\varepsilon_y)
=
\frac{\rho\sigma\eta(1-e^{-(a+b)\Delta})}{a+b}.
$$

重要的是：endpoint innovations 的相关系数一般**不等于**瞬时 Brownian correlation $\rho$，除非特殊参数或极小步长。正确做法是对上面的协方差矩阵做 Cholesky，而不是直接用相关系数 $\rho$ 生成两个 endpoint shocks。

## 7.4 G2++ 相比 Hull--White 增加了什么

- 两个均值回复速度允许短端和长端以不同 persistence 变化。
- $\rho$ 控制两个因子的共同移动，能产生更丰富的 slope/curvature dynamics。
- 仍是 Gaussian，因此解析债券价格、Gaussian integration 和精确 endpoint simulation 都保留。
- 仍可能产生负短利率，仍缺少随机波动率与 smile。
