---
title: "Poisson Process：从零开始，到 Order Book Master Equation"
subtitle: "面向第一次系统学习 Poisson 过程的读者：先建立事件计数、等待时间和 infinitesimal probability 的直觉，再把它直接接到你书里 queue 的 master equation。"
date: "2026-08-23"
lang: zh-CN
---

> **Source of truth:** edit this Markdown file. The rendered artifact is `poisson_process_for_order_book.html`.

> **先记住一句话：** homogeneous Poisson process 是一个“随机事件到达模型”。参数 $\lambda$ 是**每单位时间的平均事件率**。在很短的 $\Delta t$ 内，发生一次事件的概率大约是 $\lambda\Delta t$；把这些小时间段拼起来，就得到 Poisson 计数分布、Exponential 等待时间，以及你书里的 master equation。

## 0. 先把四个对象分清楚

很多初学 Poisson process 的困惑，其实是把下面四样东西混在了一起。

| 对象 | 记号 | 它回答什么问题？ | 分布 |
|---|---:|---|---|
| 事件计数 | $N(t)$ | 从 $0$ 到 $t$ 一共发生了几次？ | $\operatorname{Poisson}(\lambda t)$ |
| 第 $n$ 个事件时间 | $S_n$ | 第 $n$ 次事件什么时候发生？ | Erlang/Gamma |
| 相邻事件间隔 | $X_n=S_n-S_{n-1}$ | 下一次还要等多久？ | $\operatorname{Exp}(\lambda)$ |
| 事件率 | $\lambda$ | 平均每单位时间发生多少次？ | 不是概率；单位是 $1/\text{time}$ |

最关键的量是 counting process

$$
N(t)=\#\{\text{events occurring in }[0,t]\}.
$$

例如 limit order 平均每秒来 4 次，可以写 $\lambda=4/\text{s}$。这**不表示**每 $0.25$ 秒准时来一个订单，只表示平均间隔是 $1/\lambda=0.25$ 秒。

## 1. 最直观的定义：看一个极短时间 $dt$

把时间缩得非常短。Poisson process 的核心局部规则是

$$
\Pr(\Delta N=1)=\lambda\,dt+o(dt),
$$

$$
\Pr(\Delta N=0)=1-\lambda\,dt+o(dt),
$$

$$
\Pr(\Delta N\ge2)=o(dt).
$$

这里 $o(dt)$ 的意思是：当 $dt\to0$ 时，它比 $dt$ 更快地趋近于 0，所以除以 $dt$ 后会消失。

### 一个数值例子

若 $\lambda=4/\text{s}$，取 $dt=0.01\text{s}$，那么

$$
\Pr(\text{one event in }dt)\approx 4\times0.01=0.04.
$$

精确的“至少一个事件”概率其实是

$$
1-e^{-\lambda dt}=1-e^{-0.04}\approx0.03921,
$$

和 $0.04$ 很接近。短时间近似 $\lambda dt$ 正是 master equation 推导的入口。

**重要：** $\lambda$ 本身不是概率。只有乘上一个很短的时间长度 $dt$ 后，$\lambda dt$ 才近似是“这段时间内发生一次事件”的概率。

## 2. 为什么 $N(t)$ 是 Poisson 分布？直接从 $dt$ 推出来

令

$$
p_n(t)=\Pr(N(t)=n).
$$

考虑从 $t$ 到 $t+dt$。若最终一共有 $n$ 次事件，主要有两种一阶概率路径：

1. 在 $t$ 时已经有 $n$ 次，接下来 $dt$ 内没有新事件；
2. 在 $t$ 时有 $n-1$ 次，接下来 $dt$ 内恰好来一次。

因此

$$
p_n(t+dt)
=p_n(t)(1-\lambda dt)+p_{n-1}(t)\lambda dt+o(dt).
$$

移项、除以 $dt$、令 $dt\to0$：

$$
\frac{dp_n(t)}{dt}
=-\lambda p_n(t)+\lambda p_{n-1}(t),\qquad n\ge1.
$$

对于 $n=0$：

$$
\frac{dp_0(t)}{dt}=-\lambda p_0(t),\qquad p_0(0)=1,
$$

所以

$$
p_0(t)=e^{-\lambda t}.
$$

再递推求解即可得到

$$
\boxed{\Pr(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}}.
$$

这就是 Poisson distribution。于是

$$
\mathbb E[N(t)]=\lambda t,
\qquad
\operatorname{Var}(N(t))=\lambda t.
$$

**这一步与你书里的 master equation 几乎是同一种推理：**“当前状态的概率变化 = 从别的状态流进来的概率 − 从当前状态流出去的概率”。

## 3. 为什么等待时间是 Exponential？

设 $X_1$ 是等到第一次事件的时间。事件“$X_1>t$”等价于“$[0,t]$ 内一个事件都没发生”，也就是 $N(t)=0$。

因此

$$
\Pr(X_1>t)=\Pr(N(t)=0)=e^{-\lambda t}.
$$

所以 CDF 是

$$
F_{X_1}(t)=1-e^{-\lambda t},
$$

密度为

$$
f_{X_1}(t)=\lambda e^{-\lambda t},\qquad t\ge0,
$$

即

$$
\boxed{X_1\sim\operatorname{Exp}(\lambda)}.
$$

而且

$$
\mathbb E[X_1]=\frac1\lambda.
$$

所以“计数是 Poisson”和“间隔是 Exponential”其实是**同一个过程的两种视角**。

## 4. Memoryless 到底是什么意思？

Exponential distribution 有

$$
\Pr(X>s+t\mid X>s)=\Pr(X>t)=e^{-\lambda t}.
$$

意思是：已经等了多久，不改变接下来还要等多久的分布。

例如，一个 Poisson order flow 已经 2 秒没有新订单，并不意味着“现在更该来一个了”。在模型里，未来的 hazard 仍然是同一个 $\lambda$。

对应的 hazard rate 是常数：

$$
h(t)=\frac{f(t)}{\Pr(X>t)}=\lambda.
$$

这也是为什么“constant arrival rate”与 Exponential waiting time 是同一件事的两个表述。

## 5. Independent increments 与 stationary increments

Homogeneous Poisson process 通常可以由三条性质描述：

1. $N(0)=0$；
2. **independent increments**：不重叠时间区间里的事件数彼此独立；
3. **stationary increments**：一个长度为 $h$ 的区间内的事件数分布只依赖 $h$，不依赖区间放在几点钟。

因此

$$
N(t+h)-N(t)\sim\operatorname{Poisson}(\lambda h).
$$

注意 stationary increments 并不等于“市场本身 stationary”。现实订单流常有开盘、午间、收盘等明显 intraday seasonality，所以真实市场常需要 time-varying intensity $\lambda(t)$。

## 6. Superposition：为什么多个 Poisson clock 的 rate 可以相加？

如果

$$
N_1(t)\sim PP(\lambda_1),\qquad
N_2(t)\sim PP(\lambda_2)
$$

且独立，那么合并事件流

$$
N(t)=N_1(t)+N_2(t)
$$

仍然是 Poisson process，而且

$$
\boxed{N(t)\sim PP(\lambda_1+\lambda_2)}.
$$

短时间里也很好理解：

$$
\Pr(\text{任意一个 clock 响})
\approx \lambda_1dt+\lambda_2dt
=(\lambda_1+\lambda_2)dt.
$$

这正是你书里为什么 execution rate $\mu$ 与 cancellation rate $\nu$ 可以合并成 departure rate $\mu+\nu$。

## 7. Thinning：反过来拆一个 Poisson process

假设 rate 为 $\lambda$ 的每个 arrival 独立地以概率 $p$ 被标记为 A，否则标记为 B。那么 A/B 两条流分别是

$$
PP(p\lambda),
\qquad
PP((1-p)\lambda),
$$

并且彼此独立。

这在市场微观结构里很有用：例如把总体 message flow 按 buy/sell、venue、order type 等随机标签拆分，Poisson 模型下对应 intensity 会按概率比例缩放。

## 8. Competing exponentials：你那一页最需要的结论

现在假设有三个独立 clock：

- 新 limit order：$T_\lambda\sim\operatorname{Exp}(\lambda)$；
- execution：$T_\mu\sim\operatorname{Exp}(\mu)$；
- cancellation：$T_\nu\sim\operatorname{Exp}(\nu)$。

真正发生的下一件事由

$$
T=\min(T_\lambda,T_\mu,T_\nu)
$$

决定。

最小等待时间仍是 exponential：

$$
\boxed{T\sim\operatorname{Exp}(\lambda+\mu+\nu)}.
$$

而下一件事是哪一种的概率为

$$
\Pr(\text{arrival first})
=\frac{\lambda}{\lambda+\mu+\nu},
$$

$$
\Pr(\text{execution first})
=\frac{\mu}{\lambda+\mu+\nu},
$$

$$
\Pr(\text{cancellation first})
=\frac{\nu}{\lambda+\mu+\nu}.
$$

### 一个数字例子

若

$$
\lambda=5,\qquad \mu=3,\qquad \nu=2
$$

（单位都为 events/s），那么总 event rate 是 $10$/s，所以下一个事件平均只要等

$$
\frac1{10}=0.1\text{s}.
$$

下一事件分别为 arrival / execution / cancellation 的概率就是 $0.5/0.3/0.2$。

## 9. 现在回到 order book：式 (5.2) 就很自然了

令 $V$ 是 queue volume。书中的最简单模型有三类 Poisson event：

$$
V-1 \xrightarrow{\lambda} V \xleftarrow{\mu+\nu} V+1
$$

在极短的 $dt$ 内，考虑 $t+dt$ 时 queue 恰好等于 $V$。

### 路径 A：原来就在 $V$，而且什么都没发生

三个 clock 的总 rate 是 $\lambda+\mu+\nu$，因此

$$
P(V,t)\left[1-(\lambda+\mu+\nu)dt\right].
$$

### 路径 B：原来在 $V-1$，来了一个新 limit order

贡献为

$$
\lambda dt\,P(V-1,t).
$$

### 路径 C：原来在 $V+1$，发生 execution 或 cancellation

贡献为

$$
(\mu+\nu)dt\,P(V+1,t).
$$

### 路径 D：queue depletion 后重新注入到 $V$

若 depletion 的总概率流为 $J(t)$，新 queue 的初始 volume 分布为 $\rho(V)$，贡献为

$$
J(t)\rho(V)dt.
$$

把四项加起来：

$$
\begin{aligned}
P(V,t+dt)
={}&P(V,t)\left[1-(\lambda+\mu+\nu)dt\right]\\
&+\lambda dt\,P(V-1,t)\\
&+(\mu+\nu)dt\,P(V+1,t)\\
&+J(t)\rho(V)dt+o(dt).
\end{aligned}
$$

减去 $P(V,t)$，除以 $dt$，再令 $dt\to0$：

$$
\boxed{
\frac{\partial P(V,t)}{\partial t}
=-(\lambda+\mu+\nu)P(V,t)
+\lambda P(V-1,t)
+(\mu+\nu)P(V+1,t)
+J(t)\rho(V)
}.
$$

这就是你照片里的 master equation。

**所以 master equation 并不是“凭空写出来的微分方程”。**它只是把一个非常短的 $dt$ 内所有一阶概率路径列出来，然后做概率守恒。

## 10. 为什么同时发生两个 event 可以忽略？

这是 Poisson infinitesimal argument 里很重要的一点。

一个 clock 在 $dt$ 内发生一次的概率是 $O(dt)$。两个独立 clock 都在同一个 $dt$ 内响，概率量级是

$$
O(dt)\times O(dt)=O(dt^2).
$$

在 master equation 里最后会除以 $dt$，于是

$$
\frac{O(dt^2)}{dt}=O(dt)\to0.
$$

因此只保留“零个事件”和“一个事件”这两类一阶项即可。

## 11. Poisson distribution 与 Poisson process 的常见混淆

| 常见误解 | 正确理解 |
|---|---|
| “$\lambda=5$ 表示事件概率是 5” | 错。$\lambda$ 是 rate，单位是 events/time；$\lambda dt$ 才近似是短时间概率。 |
| “Poisson process 的等待时间也是 Poisson” | 错。计数 $N(t)$ 是 Poisson；等待时间 $X$ 是 Exponential。 |
| “平均间隔 $1/\lambda$，所以事件每 $1/\lambda$ 秒来一次” | 错。间隔随机；$1/\lambda$ 只是均值。 |
| “没来很久了，下一刻应该更容易来” | homogeneous Poisson 下错。Exponential 是 memoryless。 |
| “多个 event rate 相加只是经验规则” | 不是。独立 Poisson process superposition 后严格仍是 Poisson，rate 相加。 |
| “master equation 是连续状态 PDE” | 在你这页里 $V$ 是离散 queue state；时间连续，所以本质是连续时间 Markov chain 的 forward equation。 |

## 12. 用 NumPy 模拟一个 Poisson process

最简单的方法不是每个很小 $dt$ 扔硬币，而是直接模拟 Exponential interarrival times：

```python
import numpy as np

rng = np.random.default_rng(0)
lambda_ = 4.0          # 4 events / second
T = 5.0                # simulate 5 seconds

gaps = []
t = 0.0
while t < T:
    gap = rng.exponential(scale=1 / lambda_)
    t += gap
    if t <= T:
        gaps.append(t)

arrival_times = np.array(gaps)
print(arrival_times)
print("N(T) =", len(arrival_times))
```

如果只想模拟 $[0,T]$ 内的总事件数，则直接：

```python
N_T = rng.poisson(lambda_ * T)
```

二者对应同一个 Poisson process 的两个表述。

## 13. 哪些地方在真实 order book 中会失效？

Poisson model 很重要，因为它给出最干净的 baseline；但真实市场往往违反它的强假设：

- arrival intensity 会随 time-of-day 变化：$\lambda\to\lambda(t)$；
- intensity 会依赖 queue size、spread、imbalance 等 state：$\lambda\to\lambda(V,\text{spread},\ldots)$；
- 事件会 clustering，自激效应可能使 independent increments 不成立；
- 不同 event type 之间可能互相影响；
- cancellation 很可能不是 constant rate。

进一步的模型通常包括 non-homogeneous Poisson process、state-dependent continuous-time Markov chain、Cox process 和 Hawkes process。

但学习顺序最好仍然是：

$$
\boxed{\text{Poisson} \rightarrow \text{Exponential clocks} \rightarrow \text{Master equation} \rightarrow \text{state-dependent intensities / Hawkes}}.
$$

## 14. 一页 Cheat Sheet

#### 计数
$$N(t)\sim\operatorname{Poisson}(\lambda t)$$
$$\Pr(N(t)=n)=e^{-\lambda t}\frac{(\lambda t)^n}{n!}$$

#### 等待时间
$$X_i\overset{iid}{\sim}\operatorname{Exp}(\lambda)$$
$$\Pr(X_i>t)=e^{-\lambda t},\qquad \mathbb E[X_i]=1/\lambda$$

#### 短时间概率
$$\Pr(1\text{ event in }dt)=\lambda dt+o(dt)$$
$$\Pr(\ge2\text{ events})=o(dt)$$

#### 多个独立 clocks
$$\min_i T_i\sim\operatorname{Exp}\!\left(\sum_i\lambda_i\right)$$
$$\Pr(i\text{ wins})=\frac{\lambda_i}{\sum_j\lambda_j}$$


## 15. 推荐参考资料

我建议按下面顺序读。第一项最适合作为你的主教材；第二项补 merging/splitting；第三项更系统、更偏 stochastic processes。

1. **MIT OpenCourseWare 6.041SC — Lecture 14: Poisson Process I, John Tsitsiklis.** 从 Bernoulli process 过渡到 Poisson process，覆盖 definition、count PMF、interarrival time 和基本性质。  
   <https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-iii/lecture-14/>

2. **MIT OpenCourseWare 6.041SC — Lecture 15: Poisson Process II.** 重点是 merging/splitting 和例子。  
   <https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-iii/lecture-15/>

3. **Robert Gallager, MIT 6.262 Discrete Stochastic Processes — Chapter 2: Poisson Processes.** 系统覆盖 arrival processes、memorylessness、Poisson PMF、combining/splitting、non-homogeneous process、conditional arrival times/order statistics。  
   <https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/resources/mit6_262s11_chap02/>


本报告把 Poisson process 的基础概念有意组织成与 queue/master-equation 推导直接衔接的顺序。公式采用浏览器原生 MathML，不依赖外部 MathJax CDN。
