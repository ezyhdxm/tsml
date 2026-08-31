---
title: "Order Book Queue：Master Equation 与稳态分布推导"
subtitle: "从 infinitesimal probability balance 出发，逐项解释式 (5.2)，再推到 characteristic equation、稳定条件和固定 $V_0$ reinjection 下的闭式稳态分布。"
date: "2026-08-23"
lang: zh-CN
---

> **Source of truth:** edit this Markdown file. The rendered artifact is `order_book_master_equation_derivation.html`.

> 这一页的核心是一个 **birth-death dynamics + boundary reset**。在 $V\ge 2$ 的内部状态，queue volume 只做 nearest-neighbor jump：arrival 令 $V\to V+1$，execution/cancellation 令 $V\to V-1$。但在边界 $V=1$，一次 downward event 会先使 queue depletion 到 0，然后模型立刻按 $\rho(V)$ 把 queue reset 到新的正 volume。因此，**内部 dynamics 是 nearest-neighbor；完整过程在边界允许一次 nonlocal reset**。Master equation 只是把这些概率流逐项记账。

## 0. 模型先翻译成人话

记

- $V\in\{1,2,\ldots\}$：当前 queue volume；
- $\lambda$：新 limit order arrival rate，因此 $V\to V+1$；
- $\mu$：execution rate；
- $\nu$：cancellation rate；
- $c=\mu+\nu$：总 departure rate，因此 $V\to V-1$；
- $P(V,t)$：时刻 $t$ 看到 queue volume 为 $V$ 的概率；
- $J(t)$：queue depletion 到 0 的 probability flux；
- $\rho(V)$：depletion 后新 queue 的初始 volume 分布。

$$
V-1 \xrightarrow{\lambda} V \xleftarrow{c=\mu+\nu} V+1
$$

由于 queue 一到 0 就被重新生成，模型在观测层面不让系统停留在 0，因此

$$
P(0,t)=0.
$$

注意这不等于“从来没有 hitting 0”；恰恰相反，hitting 0 发生时会产生 depletion flux $J(t)$，然后瞬间 reinject。

## 1. 式 (5.2)：先把 \(V=1\) 和 \(V\ge2\) 分开，\(J(t)\) 就不会突然出现

你提出的疑问是对的：如果只看三个 Poisson clocks，那么在一个足够短的 \(dt\) 内，一阶上至多发生一个 clock event；而在普通内部状态，这个 event 只会让 queue volume 改变一个单位。

因此，**最自然的推导方式确实是先把边界 \(V=1\) 和内部 \(V\ge2\) 分开。**

### 1.1 先看内部状态：\(V\ge2\)

令

$$
c:=\mu+\nu.
$$

当当前状态 \(V\ge2\) 时：

- arrival：\(V\to V+1\)，rate 为 \(\lambda\)；
- execution/cancellation：\(V\to V-1\)，rate 为 \(c\)。

所以如果暂时不考虑从边界 reinject 进来的概率，在 \(dt\) 内确实只有 nearest-neighbor movement。

要在 \(t+dt\) 时处于 \(V\ge2\)，局部的三条路是

$$
P(V,t)\bigl[1-(\lambda+c)dt\bigr],
$$

$$
\lambda dt\,P(V-1,t),
$$

和

$$
c\,dt\,P(V+1,t).
$$

到这里完全没有 \(J(t)\)。

### 1.2 真正特殊的是 \(V=1\)

当系统已经在 \(V=1\) 时，一个 downward event 不再只是普通的

$$
1\to0.
$$

书里额外规定：

> queue 一旦到 0，就**立即**生成一个新的 queue volume \(U\)，其中
> \[
> \Pr(U=V)=\rho(V).
> \]

所以 boundary event 实际上是

$$
1
\;\xrightarrow{\text{departure, rate }c}\;
0
\;\xrightarrow{\text{instant reset}}\;
U\sim\rho.
$$

这里最重要的一点是：

> **这不是在 \(dt\) 内发生了两个独立 Poisson events。**

只有第一个

$$
1\to0
$$

是 Poisson clock 触发的随机 event。后面的

$$
0\to U
$$

是模型规定的**瞬时 boundary reset rule**，没有额外等待时间，也不是第四个 Poisson clock。

所以，“一个很短的 \(dt\) 内至多发生一个 Poisson event”仍然成立；但是由于这个 event 一旦击中边界，会触发一个 nonlocal reset，**观测到的状态可以从 1 直接变成 \(U>2\)**。

这正是看起来和“每次只移动一个单位”冲突的地方。

### 1.3 \(J(t)\) 其实不是新参数；它就是从 \(V=1\) 流到 0 的 boundary flux

在当前这个 single-unit downward-jump 模型里，只有处在 \(V=1\) 时，一个 departure 才能把 queue depletion 到 0。

因此在时刻 \(t\)：

- 系统处于 \(V=1\) 的概率是 \(P(1,t)\)；
- conditional on \(V=1\)，downward hazard 是 \(c=\mu+\nu\)。

所以 depletion probability per unit time 是

$$
\boxed{
J(t)=cP(1,t)=(\mu+\nu)P(1,t).
}
$$

也就是说，\(J(t)\) **不是突然多出来的第四个 intensity**；它是由 \(V=1\) 的概率质量决定的 boundary probability flux。

书里后面在 stationary case 写的

$$
J_0=(\mu+\nu)P_{\mathrm{st}}(1)
$$

就是这条关系的稳态版本。

### 1.4 现在分别写 \(V=1\) 和 \(V\ge2\) 的 \(dt\) balance

先写 \(V=1\)。

要在 \(t+dt\) 时看到 \(V=1\)，一阶上有三种可能：

1. 原来就在 1，而且没有任何 Poisson event；
2. 原来在 2，发生一个 downward event；
3. 原来在 1，发生 depletion，然后 reset 又恰好抽到 \(U=1\)。

所以

$$
\begin{aligned}
P(1,t+dt)
={}&P(1,t)\bigl[1-(\lambda+c)dt\bigr]\\
&+c\,dt\,P(2,t)\\
&+c\,dt\,P(1,t)\rho(1)
+o(dt).
\end{aligned}
$$

注意这里**没有**

$$
\lambda dt\,P(0,t),
$$

因为 0 不是一个会停留的状态；模型规定到 0 后立刻 reset。

再看任意 \(V\ge2\)。

除了普通的 local inflow，还可能有一条来自边界的路径：

- 原来在 \(V=1\)；
- 发生一个 downward event；
- queue depletion；
- reset volume 恰好抽到当前这个 \(V\)。

因此

$$
\begin{aligned}
P(V,t+dt)
={}&P(V,t)\bigl[1-(\lambda+c)dt\bigr]\\
&+\lambda dt\,P(V-1,t)\\
&+c\,dt\,P(V+1,t)\\
&+c\,dt\,P(1,t)\rho(V)
+o(dt),
\qquad V\ge2.
\end{aligned}
$$

这时 \(J(t)\rho(V)\) 的来源已经完全清楚了，因为

$$
cP(1,t)\rho(V)
=
J(t)\rho(V).
$$

### 1.5 最后才把两种情况压成书里的统一写法

定义

$$
P(0,t)=0
$$

以及

$$
J(t):=cP(1,t).
$$

那么对所有 \(V\ge1\)，上面的两组方程可以统一写成

$$
\begin{aligned}
P(V,t+dt)
={}&P(V,t)\left[1-(\lambda+c)dt\right]\\
&+\lambda dt\,P(V-1,t)\\
&+c\,dt\,P(V+1,t)\\
&+J(t)\rho(V)dt
+o(dt).
\end{aligned}
$$

减去 \(P(V,t)\)，除以 \(dt\)，再令 \(dt\to0\)：

$$
\boxed{
\frac{\partial P(V,t)}{\partial t}
=
-(\lambda+c)P(V,t)
+\lambda P(V-1,t)
+cP(V+1,t)
+J(t)\rho(V)
}.
$$

代回 \(c=\mu+\nu\)：

$$
\boxed{
\frac{\partial P(V,t)}{\partial t}
=
-(\lambda+\mu+\nu)P(V,t)
+\lambda P(V-1,t)
+(\mu+\nu)P(V+1,t)
+J(t)\rho(V)
}.
$$

所以书里的写法不是错，而是**已经把 boundary case \(V=1\) 压缩进 \(J(t)\rho(V)\) 这一项里了**。

从教学角度看，先把 \(V=1\) 分开推，再合并成统一公式，确实更容易理解。

### 1.6 固定 reinjection 到 \(V_0\) 时，这一点尤其清楚

若

$$
\rho(V)=\mathbf 1\{V=V_0\},
$$

那么

$$
J(t)\rho(V)
=
J(t)\mathbf 1\{V=V_0\}.
$$

也就是说：

- \(V=1\) 是 **depletion boundary**；
- \(V=V_0\) 是 **reinjection source**；
- 其它状态只满足普通 nearest-neighbor birth-death equation。

如果 \(V_0>1\)，动态方程可以明确写成

$$
\frac{\partial P(1,t)}{\partial t}
=
-(\lambda+c)P(1,t)+cP(2,t),
$$

$$
\frac{\partial P(V,t)}{\partial t}
=
-(\lambda+c)P(V,t)
+\lambda P(V-1,t)
+cP(V+1,t),
\qquad
V\ge2,\;V\neq V_0,
$$

以及 source state

$$
\frac{\partial P(V_0,t)}{\partial t}
=
-(\lambda+c)P(V_0,t)
+\lambda P(V_0-1,t)
+cP(V_0+1,t)
+J(t).
$$

其中始终有

$$
\boxed{J(t)=cP(1,t).}
$$

这就是后面 stationary solution 为什么要特别处理 \(V=1\) 和 \(V=V_0\) 的原因。

### 1.7 如果坚持“任何一次状态变化都只能 \(\pm1\)”，那就必须换模型

书里的 instantaneous reinjection 本身允许 boundary 上的 nonlocal jump：

$$
1\to0\to U.
$$

如果你希望整个过程在任何时候都严格 nearest-neighbor，那么就不能使用这种 arbitrary-\(\rho\) 的瞬时 reset。可以改成例如：

- 把 \(V=0\) 保留成真实状态；
- refill 也通过显式的 \(0\to1\to2\to\cdots\) 事件逐步发生；

或者至少给 \(0\to U\) 一个有限等待时间，而不是 instantaneous reset。

书里选择 instantaneous reinjection，主要是为了让 queue depletion 后立刻开始一个新的 regenerative cycle，并避免 probability mass 长时间停在 0。

**因此你指出的区分是正确的：原始 Poisson dynamics 在内部确实每次只走 \(\pm1\)；\(J(t)\rho(V)\) 描述的是 boundary reset，而不是另一个普通的 local jump。**

## 2. 为什么同时发生两个 event 不写进去？

两个独立 Poisson event 都落在同一个长度 $dt$ 的区间，其概率是 $O(dt^2)$。Master equation 最后会除以 $dt$，因此

$$
\frac{O(dt^2)}{dt}=O(dt)\to0.
$$

所以只保留一阶的 $O(dt)$ 项。

## 3. 为什么 stationary solution 要令时间导数为 0？

stationary distribution 的定义就是：经过足够长时间后，每个 queue size 的概率不再随时间变化。

写作 $P_s(V)$，即

$$
\frac{\partial P(V,t)}{\partial t}=0.
$$

于是 master equation 变成一个关于离散状态 $V$ 的差分方程：

$$
0=-(\lambda+c)P_s(V)+\lambda P_s(V-1)+cP_s(V+1)+J_0\rho(V).
$$

其中 stationary state 下 $J(t)$ 也必须变成常数 $J_0$。

## 4. 先看没有 reinjection source 的区间

对 $\rho(V)=0$ 的位置，方程是齐次的：

$$
cP_{V+1}-(\lambda+c)P_V+\lambda P_{V-1}=0.
$$

试一个几何形式

$$
P_V=a^V.
$$

代入：

$$
ca^{V+1}-(\lambda+c)a^V+\lambda a^{V-1}=0.
$$

除以 $a^{V-1}$：

$$
\boxed{ca^2-(\lambda+c)a+\lambda=0}.
$$

这就是书中的式 (5.3)，若把 $c$ 展开即

$$
(\mu+\nu)a^2-(\lambda+\mu+\nu)a+\lambda=0.
$$

它可以直接因式分解：

$$
ca^2-(\lambda+c)a+\lambda
=(a-1)(ca-\lambda).
$$

所以两个 characteristic roots 是

$$
\boxed{a_1=1,\qquad a_2=\frac{\lambda}{c}=\frac{\lambda}{\mu+\nu}}.
$$

因此齐次解的结构是

$$
\boxed{P_V=A+B\left(\frac{\lambda}{c}\right)^V}.
$$

书上写的“$k+a^V$”就是在提示“常数解 + 几何解”这件事。

## 5. 为什么稳定要求 $\lambda<\mu+\nu$？

定义

$$
r=\frac{\lambda}{c}=\frac{\lambda}{\mu+\nu}.
$$

如果 $\lambda>c$，向上的 birth rate 大于向下的 departure rate，queue 有正 drift：

$$
\mathbb E[dV]/dt\approx \lambda-c>0.
$$

它会不断向大 $V$ 漂移，无法得到可归一化的 stationary distribution。

若 $\lambda=c$，则 $r=1$，几何尾部不衰减；在无上界 queue 上同样不能得到通常意义下的可归一化稳态。

所以这里真正需要的是

$$
\boxed{\lambda<c=\mu+\nu}.
$$

也就是

$$
\boxed{r<1}.
$$

这和普通 $M/M/1$ queue 的 stability condition 完全同型：平均流入必须小于平均流出能力。

## 6. 固定 reinjection 到 $V_0$：$\rho(V)=\mathbf 1\{V=V_0\}$

现在假设每次 queue depletion 后都立刻重生到同一个 volume $V_0$：

$$
\rho(V)=\mathbf 1\{V=V_0\}.
$$

这时只有 $V=V_0$ 处有 source $J_0$。

一个很干净的做法是看相邻状态之间的 probability current。定义“向上”的净流

$$
j_V=\lambda P_V-cP_{V+1}.
$$

对没有 source 的状态，stationary master equation 可以写成

$$
0=j_{V-1}-j_V,
$$

所以同一段区间里的 current 是常数。

### 6.1 $V>V_0$：无穷远不能持续漏概率

在 $V_0$ 上方没有 source，而且不能有持续流向 $+\infty$ 的概率流，因此

$$
j_V=0.
$$

于是

$$
\lambda P_V=cP_{V+1},
$$

即

$$
P_{V+1}=rP_V.
$$

因此

$$
\boxed{P_V=P_{V_0}r^{V-V_0},\qquad V\ge V_0.}
$$

### 6.2 $1\le V\le V_0$：每层都承载着朝 0 的 depletion current

每次 reinjection 在 $V_0$ 注入 probability mass，最终都要从 1 向 0 流出。因此在 $V_0$ 下方，向下的 current 大小为 $J_0$：

$$
cP_{V+1}-\lambda P_V=J_0.
$$

也就是

$$
P_{V+1}=rP_V+\frac{J_0}{c}.
$$

从边界 $P_0=0$ 开始递推：

$$
P_1=\frac{J_0}{c},
$$

$$
P_2=\frac{J_0}{c}(1+r),
$$

一般地

$$
P_V=\frac{J_0}{c}\sum_{k=0}^{V-1}r^k.
$$

用几何级数求和：

$$
\boxed{
P_V=\frac{J_0}{c-\lambda}(1-r^V),\qquad 1\le V\le V_0.
}
$$

因此

$$
P_{V_0}=\frac{J_0}{c-\lambda}(1-r^{V_0}).
$$

### 6.3 用归一化求 $J_0$

要求

$$
\sum_{V=1}^{\infty}P_V=1.
$$

把 $V\le V_0$ 与 $V>V_0$ 两段相加，可以化简为

$$
1=\frac{J_0V_0}{c-\lambda}.
$$

因此

$$
\boxed{J_0=\frac{c-\lambda}{V_0}
=\frac{\mu+\nu-\lambda}{V_0}.}
$$

代回前面的分段解：

$$
\boxed{
P_V=
\begin{cases}
\displaystyle \frac{1}{V_0}(1-r^V), & 1\le V\le V_0,\\[8pt]
\displaystyle \frac{1}{V_0}(1-r^{V_0})r^{V-V_0}, & V\ge V_0,
\end{cases}
\qquad r=\frac{\lambda}{\mu+\nu}<1.
}
$$

在 $V=V_0$ 两段一致，所以这确实是连续拼接的离散分布。

## 7. $J_0$ 的另一种直觉：regenerative cycle

每次 depletion 后都从 $V_0$ 重生，因此过程可以分成一个个独立风格的 cycle：

$$
V_0\longrightarrow \cdots \longrightarrow 1\longrightarrow0\longrightarrow V_0.
$$

对于 birth rate $\lambda$、death rate $c>\lambda$ 的随机游走，从 $V_0$ 到 0 的平均 hitting time 是

$$
\mathbb E[\tau_0\mid V(0)=V_0]
=\frac{V_0}{c-\lambda}.
$$

stationary depletion rate 就是一单位时间平均完成多少个 cycle：

$$
J_0=\frac1{\mathbb E[\tau_0]}
=\frac{c-\lambda}{V_0}.
$$

这与上面的归一化结果完全一致。

## 8. 从 continuous-time Markov chain 的角度看

如果暂时忽略 boundary reinjection，这个模型在内部状态 $V\ge2$ 的 generator 只有三条非零元素：

$$
Q_{V,V+1}=\lambda,
$$

$$
Q_{V,V-1}=c,
$$

$$
Q_{V,V}=-(\lambda+c).
$$

所以 master equation 本质上就是 continuous-time Markov chain 的 Kolmogorov forward equation。你不需要先学 generator 才能理解它；$dt$ probability balance 已经包含全部直觉。

## 9. 你这页最容易卡住的三个点

| 卡点 | 解释 |
|---|---|
| 为什么 outflow 是 $\lambda+\mu+\nu$？ | 因为从状态 $V$ 出发，任一 arrival/execution/cancellation 都会让系统离开 $V$；独立 Poisson hazards 相加。 |
| 为什么 $P(0,t)=0$ 但还有 depletion？ | 0 是瞬时触发 reset 的边界，不是一个会停留的观测状态；流到 0 的 probability flux 就是 $J(t)$。 |
| 为什么试 $a^V$？ | stationary equation 是常系数二阶线性差分方程；几何序列之于差分方程，就像指数函数之于常系数微分方程。 |

## 10. 最重要的直觉总结

1. **Poisson clock** 给出短时间事件概率 $\text{rate}\times dt$；
2. **Master equation** 是把每个状态的 inflow 与 outflow 列出来；
3. **Stationary distribution** 就是让每个状态 net probability flow 为 0；
4. **Characteristic root** $r=\lambda/(\mu+\nu)$ 控制尾部衰减速度；
5. **Stability** 需要 $r<1$；
6. **Reinjection** 把每次 depletion 重新送回 $V_0$，形成 regenerative cycles；
7. 固定 $V_0$ 时，depletion rate 为

$$
\boxed{J_0=\frac{\mu+\nu-\lambda}{V_0}}.
$$

本报告不嵌入原书页面图片；只整理数学推导。公式采用浏览器原生 MathML，因此不依赖外部 MathJax CDN。
