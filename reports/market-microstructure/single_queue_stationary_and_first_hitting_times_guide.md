---
title: "Single-Queue Dynamics：从稳态分布到 First-Hitting Time"
subtitle: "pp.82–86 的逐步解释：stationary distribution、exit flux、first-step decomposition、Laplace transform、mean/variance 与 critical case。"
date: "2026-08-23"
lang: zh-CN
---

> **Source of truth:** edit this Markdown file. The rendered artifact is `single_queue_stationary_and_first_hitting_times_guide.html`.

# 从稳态分布到 First-Hitting Time

这几页其实在研究同一个 continuous-time birth–death random walk 的两个问题：

1. **长期来看 queue volume 通常是多少？**——stationary distribution。
2. **从当前 volume \(V\) 开始，多久第一次掉到 0？**——first-hitting time。

把 execution 和 cancellation 合并成

$$
c := \mu+\nu,
$$

那么模型只有两个方向：

- upward jump \(V\to V+1\)，rate 为 \(\lambda\)；
- downward jump \(V\to V-1\)，rate 为 \(c=\mu+\nu\)。

真正控制整个模型的参数是净向下 drift

$$
\delta := c-\lambda = \mu+\nu-\lambda.
$$

如果第一遍只记四个结果，记下面这些：

$$
\lambda<c,
$$

$$
a=\frac{\lambda}{c}<1,
$$

$$
\mathbb E[T_1\mid V]=\frac{V}{c-\lambda},
$$

以及当 \(\lambda\uparrow c\) 时，queue size 和 hitting time 都会变得非常大；在临界点 \(\lambda=c\)，最终仍会 hit 0，但 mean hitting time 变成无穷大。

---

## 1. 先把 continuous-time dynamics 变成“等待 + 上下走一步”

三个 Poisson clocks 是：

- arrival：rate \(\lambda\)；
- execution：rate \(\mu\)；
- cancellation：rate \(\nu\)。

execution 和 cancellation 都让 \(V\) 减 1，因此可以合并：

$$
c=\mu+\nu.
$$

任意一种 queue-changing event 的总 rate 是

$$
R=\lambda+c.
$$

因此，从任意内部状态 \(V\ge 1\) 出发，下一次 jump 的等待时间满足

$$
\tau_1\sim \operatorname{Exp}(R),
\qquad
\mathbb E[\tau_1]=\frac{1}{R}.
$$

等到 jump 发生之后：

$$
p:=\Pr(\text{up})=\frac{\lambda}{R},
\qquad
q:=\Pr(\text{down})=\frac{c}{R}.
$$

所以整个过程可以理解成：

> 先等待一个 \(\operatorname{Exp}(\lambda+c)\) 的时间，然后以概率 \(\lambda/(\lambda+c)\) 往上一步，以概率 \(c/(\lambda+c)\) 往下一步。

后面的 stationary distribution、first-step recursion 和 hitting-time distribution 都从这个结构出来。

---

## 2. 为什么 stationary solution 是“常数 + 几何项”？

在没有 reinjection source 的位置，stationary master equation 是

$$
cP_{V+1}-(\lambda+c)P_V+\lambda P_{V-1}=0.
$$

这是一个常系数二阶差分方程。和常系数 ODE 试 \(e^{rx}\) 类似，对差分方程试

$$
P_V=a^V.
$$

代入：

$$
ca^{V+1}-(\lambda+c)a^V+\lambda a^{V-1}=0.
$$

除以 \(a^{V-1}\)：

$$
ca^2-(\lambda+c)a+\lambda=0.
$$

因式分解：

$$
(a-1)(ca-\lambda)=0.
$$

所以两根是

$$
a_+=1,
\qquad
a_-=\frac{\lambda}{c}.
$$

因此 homogeneous solution 是

$$
P_V=A+B\left(\frac{\lambda}{c}\right)^V.
$$

令

$$
a:=\frac{\lambda}{c}.
$$

如果希望大 \(V\) 时概率衰减到 0，就必须有

$$
a<1
\iff
\lambda<c.
$$

这就是 stability condition。

---

## 3. 为什么在 \(V_0\) 两边要写成不同的形式？

模型规定：queue 一旦 depletion 到 0，就立刻 reinject 到 \(V_0\)。

因此 \(V_0\) 是一个 probability source。对 \(V\neq V_0\)，stationary equation 是 homogeneous；在 \(V=V_0\) 处，方程多一个 source \(J_0\)。

所以书里写

$$
P_{\mathrm{st}}(V)=
\begin{cases}
A+Ba^V, & V\le V_0,\\[4pt]
Ca^V, & V>V_0.
\end{cases}
$$

上半段允许 constant root；但当 \(V\to\infty\) 时，常数项不能存在，因此 \(V>V_0\) 的区域只保留 decaying root \(a^V\)。

---

## 4. p.82 的四个条件分别在做什么？

### 4.1 \(V=1\) 的 boundary condition：得到 \(A=-B\)

由于系统到 0 后立即 reset，所以它不会在状态 0 停留：

$$
P_{\mathrm{st}}(0)=0.
$$

在 \(V=1\)，没有来自 \(V=0\) 的普通 arrival inflow，因此 stationary equation 是

$$
-(\lambda+c)P_1+cP_2=0.
$$

将

$$
P_V=A+Ba^V
$$

代入，并利用 \(a=\lambda/c\)，可以化简为

$$
A=-B.
$$

因此 \(1\le V\le V_0\) 时

$$
P_{\mathrm{st}}(V)
=
A(1-a^V).
$$

### 4.2 \(V=V_0\)：source 把左右两边拼起来

\(V_0\) 处有 reinjection source \(J_0\)，因此 stationary equation 在这里不是 homogeneous equation。

这条方程的作用是把：

- \(V<V_0\) 的解；
- \(V>V_0\) 的解；
- reinjection flux \(J_0\)

联系起来。

它最终给出

$$
C=A(a^{-V_0}-1).
$$

### 4.3 exit flux：为什么 \(J_0=cP_{\mathrm{st}}(1)\)？

要第一次掉到 0，最后一步一定是

$$
1\to0.
$$

只有当前 queue 在 \(V=1\) 时，一个 downward jump 才会造成 depletion。

因此：

- 处在 \(V=1\) 的概率是 \(P_{\mathrm{st}}(1)\)；
- downward hazard 是 \(c\)。

所以每单位时间的 depletion probability flux 是

$$
J_0=cP_{\mathrm{st}}(1).
$$

由于

$$
P_{\mathrm{st}}(1)=A(1-a),
$$

而 \(a=\lambda/c\)，所以

$$
J_0
=
cA(1-a)
=
A(c-\lambda).
$$

### 4.4 normalization：为什么最后变成 \(AV_0=1\)？

把分段解代入

$$
\sum_{V=1}^\infty P_{\mathrm{st}}(V)=1.
$$

前半部分是

$$
\sum_{V=1}^{V_0}A(1-a^V)
=
A\left[
V_0-\frac{a(1-a^{V_0})}{1-a}
\right].
$$

后半部分是

$$
\sum_{V=V_0+1}^{\infty}
A(a^{-V_0}-1)a^V
=
A\frac{a(1-a^{V_0})}{1-a}.
$$

两个 geometric-series correction 正好 cancel，因此只剩

$$
AV_0=1.
$$

所以

$$
A=\frac{1}{V_0}.
$$

最终 stationary distribution 是

$$
P_{\mathrm{st}}(V)=
\begin{cases}
\displaystyle
\frac{1}{V_0}(1-a^V),
&
1\le V\le V_0,
\\[10pt]
\displaystyle
\frac{1}{V_0}(a^{-V_0}-1)a^V,
&
V>V_0,
\end{cases}
$$

其中

$$
a=\frac{\lambda}{\mu+\nu}<1.
$$

---

## 5. 为什么平均 queue size 在 critical point 附近发散？

由 stationary distribution 可以求平均 queue volume。exact expression 可以整理成

$$
\bar V
=
\frac{V_0}{2}
+
\frac{1+a}{2(1-a)}.
$$

当 \(a\uparrow1\) 时，发散的主导部分是

$$
\bar V
\sim
\frac{1}{1-a}.
$$

因为

$$
1-a
=
1-\frac{\lambda}{c}
=
\frac{c-\lambda}{c},
$$

所以

$$
\bar V
\sim
\frac{c}{c-\lambda}
=
\frac{\mu+\nu}{\mu+\nu-\lambda}.
$$

因此，当 arrival rate 几乎抵消 departure rate 时，queue 会变得非常大。

---

## 6. 随机 reinjection distribution \(\rho(V_0)\) 在说什么？

固定 \(V_0\) 的解可以当作 building block。

如果每次 queue depletion 后，不是固定回到某个 \(V_0\)，而是按照分布

$$
\rho(V_0)
$$

随机生成新的 queue volume，那么由于 master equation 对概率是线性的，可以把不同 \(V_0\) 对应的 solution 做 linear superposition。

归一化条件相应变成

$$
A\sum_{V_0}V_0\rho(V_0)=1.
$$

也就是

$$
A=\frac{1}{\mathbb E[V_0]}.
$$

---

## 7. First-Hitting Time 是什么？

定义

$$
T_1(V)
:=
\inf\{t\ge0:V(t)=0\mid V(0)=V\}.
$$

也就是：

> 从当前 queue volume \(V\) 开始，到 queue 第一次 depletion 所需要的时间。

这是一个随机变量。

---

## 8. 为什么 \(J_0 = 1/\mathbb E[T_1\mid V_0]\)？

因为每次 hit 0 后系统都立即 reset 到 \(V_0\)，所以路径自然分成一个个 regenerative cycles：

$$
V_0
\longrightarrow
\cdots
\longrightarrow
0
\longrightarrow
V_0
\longrightarrow
\cdots.
$$

每个 cycle：

- 从 \(V_0\) 开始；
- 在第一次 hit 0 时结束；
- 恰好贡献一次 depletion。

平均一个 cycle 的长度是

$$
\mathbb E[T_1\mid V_0].
$$

因此长期每单位时间完成的 cycle 数是

$$
\frac{1}{\mathbb E[T_1\mid V_0]}.
$$

而长期每完成一个 cycle 就有一次 exit，所以

$$
\boxed{
J_0
=
\frac{1}{\mathbb E[T_1\mid V_0]}
}.
$$

前面已经得到

$$
J_0=\frac{c-\lambda}{V_0},
$$

所以

$$
\mathbb E[T_1\mid V_0]
=
\frac{V_0}{c-\lambda}.
$$

---

## 9. p.84 的式 (5.10)–(5.11)：first-step decomposition

令

$$
\Phi(\tau,V)
$$

表示从 \(V\) 开始的 first-hitting-time density。

设第一次 jump 发生在 \(\tau_1\)。因为总 event rate 是

$$
R=\lambda+c,
$$

所以 no-event survival factor 是

$$
e^{-R\tau_1}.
$$

第一次 jump 是 upward event 的 density 是

$$
\lambda e^{-R\tau_1},
$$

第一次 jump 是 downward event 的 density 是

$$
c e^{-R\tau_1}.
$$

利用 Markov property：

$$
\Phi(\tau,V)
=
\int_0^\tau
\lambda e^{-R\tau_1}
\Phi(\tau-\tau_1,V+1)
\,d\tau_1
+
\int_0^\tau
c e^{-R\tau_1}
\Phi(\tau-\tau_1,V-1)
\,d\tau_1.
$$

这就是书里 (5.10)–(5.11) 的来源。

真正应该记住的是：

> **first-step analysis = 先等第一次 jump，再根据 jump 到哪里递归。**

---

## 10. 为什么突然引入 Laplace transform？

上面的 integral equation 有 convolution：

$$
\int_0^\tau
f(\tau_1)g(\tau-\tau_1)\,d\tau_1.
$$

Laplace transform 可以把 convolution 变成 multiplication。

定义

$$
\widehat\Phi(z,V)
=
\int_0^\infty
e^{-z\tau}\Phi(\tau,V)\,d\tau.
$$

则

$$
\widehat\Phi(z,V)
=
\frac{\lambda}{\lambda+c+z}
\widehat\Phi(z,V+1)
+
\frac{c}{\lambda+c+z}
\widehat\Phi(z,V-1).
$$

boundary condition 是

$$
\widehat\Phi(z,0)=1,
$$

因为如果一开始就在 0，hitting time 就是 0。

---

## 11. 求 mean hitting time 不需要先用 Laplace transform

定义

$$
m_V:=\mathbb E[T_1\mid V].
$$

从状态 \(V\) 出发：

1. 平均先等 \(1/R\)；
2. 以概率 \(\lambda/R\) 到 \(V+1\)；
3. 以概率 \(c/R\) 到 \(V-1\)。

所以

$$
m_V
=
\frac{1}{R}
+
\frac{\lambda}{R}m_{V+1}
+
\frac{c}{R}m_{V-1}.
$$

由于 drift 不依赖 \(V\)，试线性解

$$
m_V=A'V.
$$

代入后 \(V\) 项消掉：

$$
A'(\lambda-c)+1=0.
$$

所以

$$
A'=\frac{1}{c-\lambda},
$$

最终

$$
\boxed{
\mathbb E[T_1\mid V]
=
\frac{V}{c-\lambda}
=
\frac{V}{\mu+\nu-\lambda}
}.
$$

---

## 12. small-\(z\) expansion 为什么能给 moments？

因为

$$
\widehat\Phi(z,V)
=
\mathbb E[e^{-zT_1}\mid V].
$$

而

$$
e^{-zT}
=
1-zT+\frac{z^2T^2}{2}+O(z^3).
$$

所以

$$
\widehat\Phi(z,V)
=
1
-z\,\mathbb E[T_1\mid V]
+
\frac{z^2}{2}\mathbb E[T_1^2\mid V]
+
O(z^3).
$$

因此：

- 一阶项给 mean；
- 二阶项给 second moment；
- 更高阶项给更高 moments。

---

## 13. 完整 hitting-time distribution：为什么又出现 characteristic roots？

Laplace-space recurrence 是

$$
(\lambda+c+z)\widehat\Phi(z,V)
=
\lambda\widehat\Phi(z,V+1)
+
c\widehat\Phi(z,V-1).
$$

再试

$$
\widehat\Phi(z,V)=a(z)^V.
$$

得到

$$
\lambda a^2
-
(\lambda+c+z)a
+
c
=
0.
$$

两根为

$$
a_\pm(z)
=
\frac{
\lambda+c+z
\pm
\sqrt{(\lambda+c+z)^2-4\lambda c}
}{
2\lambda
}.
$$

在 \(\lambda<c\) 且 \(z=0\) 时：

$$
a_+(0)=\frac{c}{\lambda}>1,
\qquad
a_-(0)=1.
$$

由于 Laplace transform 不能随着 \(V\to\infty\) 指数爆炸，增长根必须排除，所以

$$
\boxed{
\widehat\Phi(z,V)=a_-(z)^V
}.
$$

从而

$$
\ln\widehat\Phi(z,V)
=
V\ln a_-(z).
$$

---

## 14. 为什么 log Laplace transform 与 \(V\) 成正比很重要？

log Laplace transform 生成 cumulants。

由于

$$
\ln\widehat\Phi(z,V)
=
V\ln a_-(z),
$$

hitting-time distribution 的所有 cumulants 都与 \(V\) 成正比。

pathwise 上也能理解：这个 random walk 是 skip-free 的。从 \(V\) 到 0 必须依次跨过

$$
V\to V-1\to\cdots\to1\to0.
$$

利用 spatial homogeneity 和 strong Markov property，可以把 hitting time 看成

$$
T_1(V)
\overset{d}{=}
X_1+\cdots+X_V,
$$

其中 \(X_i\) 是 iid 的“下降一个 level 所需时间”。

因此 mean、variance 和更高 cumulants 都与 \(V\) 线性增长。

---

## 15. variance 怎么得到？

令

$$
d:=c-\lambda>0.
$$

对小根做 small-\(z\) expansion：

$$
\ln a_-(z)
=
-\frac{z}{d}
+
\frac{c+\lambda}{2d^3}z^2
+
O(z^3).
$$

于是

$$
\ln\widehat\Phi(z,V)
=
-\frac{V}{d}z
+
\frac{V(c+\lambda)}{2d^3}z^2
+
O(z^3).
$$

另一方面，

$$
\ln\mathbb E[e^{-zT}]
=
-\kappa_1z
+
\frac{\kappa_2}{2}z^2
+\cdots.
$$

比较 coefficient：

$$
\mathbb E[T_1\mid V]
=
\frac{V}{c-\lambda},
$$

$$
\boxed{
\operatorname{Var}(T_1\mid V)
=
\frac{V(c+\lambda)}{(c-\lambda)^3}
}.
$$

即

$$
\boxed{
\operatorname{Var}(T_1\mid V)
=
\frac{
V(\mu+\nu+\lambda)
}{
(\mu+\nu-\lambda)^3
}
}.
$$

接近 critical point 时：

$$
\mathbb E[T_1]\sim(c-\lambda)^{-1},
$$

但

$$
\operatorname{Var}(T_1)\sim(c-\lambda)^{-3}.
$$

所以不仅平均 lifetime 很长，uncertainty 也会更快爆炸。

---

## 16. critical case \(\lambda=c\)：为什么最终 hit 0，但 mean time 是无穷？

当

$$
\lambda=c,
$$

upward 和 downward rates 完全平衡，mean drift 为 0。

这时：

1. 从有限 \(V\) 开始，最终 hit 0 的概率仍然是 1；
2. 但 \(\mathbb E[T_1\mid V]=\infty\)。

原因是 hitting-time distribution 有 heavy tail。

critical case 下，小 \(z\) behavior 变成

$$
\ln\widehat\Phi(z,V)
\sim
-
V\sqrt{\frac{z}{\lambda}},
\qquad
z\downarrow0.
$$

对应长时间 first-hitting density

$$
\Phi(t,V)
\sim
\frac{V}{2\sqrt{\pi\lambda}}
t^{-3/2}.
$$

这是 power-law decay，而不是 exponential decay。

因此会出现：

> 几乎必然最终结束，但极少数非常长的 excursion 足以让 mean lifetime 发散。

---

## 17. 三个 regime 放在一起

定义

$$
\delta:=\mu+\nu-\lambda.
$$

| Regime | Drift | Stationary distribution | Hitting 0 |
|---|---:|---|---|
| \(\delta>0\) | 向下 | 存在 | a.s. hit 0，mean finite |
| \(\delta=0\) | 0 | 不存在普通可归一化稳态 | a.s. hit 0，但 mean infinite |
| \(\delta<0\) | 向上 | 不存在 | 有正概率永远不 hit 0 |

---

## 18. 推荐重新读书的顺序

1. 先读 p.85 的 (5.16)–(5.18)，理解
   $$
   \mathbb E[T_1\mid V]=\frac{V}{\mu+\nu-\lambda}.
   $$
2. 再回 p.84 看 (5.10)–(5.11)：只问“第一次 jump 何时发生？往哪里走？”
3. 再看 Laplace transform：把它理解成把 convolution 变成 algebra。
4. 再看 p.85 characteristic equation：它和 stationary difference equation 是同一个套路。
5. 最后看 p.86 critical case：把它理解成 unbiased random walk 的 heavy-tail first-passage problem。
6. p.82 的 \(A,B,C\) bookkeeping 可以最后补。

---

## 19. Monte Carlo 验证 mean / variance

```python
import numpy as np


def simulate_hitting_time(
    V0: int,
    lam: float,
    mu: float,
    nu: float,
    n_paths: int = 50_000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    c = mu + nu
    R = lam + c

    out = np.empty(n_paths)

    for i in range(n_paths):
        V = V0
        t = 0.0

        while V > 0:
            t += rng.exponential(1.0 / R)

            if rng.random() < lam / R:
                V += 1
            else:
                V -= 1

        out[i] = t

    return out


V0 = 10
lam, mu, nu = 4.0, 3.0, 2.0

T = simulate_hitting_time(V0, lam, mu, nu)

theory_mean = V0 / (mu + nu - lam)
theory_var = (
    V0 * (mu + nu + lam)
    / (mu + nu - lam) ** 3
)

print("MC mean:", T.mean())
print("Theory mean:", theory_mean)

print("MC variance:", T.var())
print("Theory variance:", theory_var)
```

当 \(\lambda\) 非常接近 \(\mu+\nu\) 时，simulation 会明显变慢，而且 sample mean 会变得不稳定。这正是模型本身的 critical behavior。

---

## 20. 最后的结构总结

stationary distribution 和 first-hitting time 是同一个 stochastic process 的两个视角：

- **stationary view**：随机挑一个很晚的时刻，queue 在哪里？
- **pathwise view**：从某个 \(V\) 开始，多久第一次到 0？

reinjection 把两者连接起来：

$$
\boxed{
J_0
=
\frac{1}{\mathbb E[T_1\mid V_0]}
}.
$$

控制整个模型的核心量是

$$
\boxed{
\mu+\nu-\lambda
}.
$$

当它远离 0 时，queue 有明显的 drift toward depletion；当它趋近于 0 时，queue volume、lifetime 和 uncertainty 一起变大；在 0 处进入 critical random-walk regime。
