## BS7 · Stochastic volatility：独立波动率风险如何进入定价？ {#bs-stochastic-vol}

### BS7.1 用 Heston 把随机方差单独写成状态

以方差 $v_t$ 而不是 volatility 作为状态，瞬时股价 volatility 是 $\sqrt{v_t}$。先明确指定一套风险中性动态：

$$
\begin{aligned}
dS_t&=(r-q)S_tdt+\sqrt{v_t}S_t\,dW_t^{S,\mathbb Q},\\
dv_t&=\kappa_v(\bar v-v_t)dt+\xi\sqrt{v_t}\,dW_t^{v,\mathbb Q},\\
d\langle W^S,W^v\rangle_t&=\rho\,dt.
\end{aligned}
$$

$\kappa_v>0$ 是方差回复速度，$\bar v>0$ 是长期方差，$\xi$ 是方差过程的扩散参数，$\rho\in[-1,1]$ 控制股票与方差冲击的相关性。这里 $\kappa_v$ 不是 BS4 的 half-spread $\kappa$。全部方差漂移参数已按 $\mathbb Q$ 定义，尚不能从历史股票回报直接读出。

条件 $2\kappa_v\bar v\ge\xi^2$ 在初始 $v_0>0$ 时保证 CIR 方差的零边界不可达；违反该条件并不意味着真实 CIR 解会变负，而是可能触及 0。普通 Euler 离散却可能产生负数，因此数值实现需要合适的非负方案或边界处理，并检查离散偏差。$v_t$ 在真实市场也未必直接可观测，过滤和校准是额外任务。

[Heston (1993)，原论文](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf) 第 329 页明确讨论了 volatility risk price 和对应 PDE。下文用上述现代参数记号重推，而不是把“假定股票风险中性”误当成已经完成全部定价。

### BS7.2 为什么股票的 drift 变成 $r$ 还不够？

把相关 Brownian 写成两个独立驱动：

$$
dW^v=\rho\,dW^S+\sqrt{1-\rho^2}\,dW^\perp.
$$

在 $\mathbb P$ 下，设股票总回报漂移为 $\mu$，方差漂移为 $a_{\mathbb P}(t,S,v)$。在 $v>0$ 的区域，用两个风险价格改变测度：

$$
dW^{S,\mathbb Q}=dW^{S,\mathbb P}+\lambda_Sdt,\quad
dW^{\perp,\mathbb Q}=dW^{\perp,\mathbb P}+\lambda_\perp dt.
$$

股票总回报漂移为 $r$ 要求

$$
\lambda_S=\frac{\mu-r}{\sqrt v}.
$$

但它没有确定 $\lambda_\perp$。方差的风险中性漂移因此是

$$
\boxed{a_{\mathbb Q}=a_{\mathbb P}-\xi\sqrt v\left[\rho\lambda_S+\sqrt{1-\rho^2}\lambda_\perp\right].}
$$

这就是额外波动率风险溢价进入的位置。只交易股票和现金、且 $|\rho|<1$ 时，股票无套利约束一般不能唯一决定 $a_{\mathbb Q}$。Heston 的风险中性参数需要额外市场信息或定价准则，例如用股票期权价格校准。指定了一个合适的 $\mathbb Q$ 后可以计算价格，但它不是仅凭股票和现金就推出的唯一复制价格。

同时注意：等价测度变换改变漂移，**不改变同一连续过程的二次变差系数**。“历史 volatility”和“implied volatility”不相同，不应该被草率解释成同一时刻的扩散系数因换测度而随意变大；这里差异可来自风险中性方差动态、未来方差分布、跳跃、模型及报价口径。

### BS7.3 两个状态下的 Itô 展开和定价 PDE

令某个存续欧式合约价值为 $V(t,S,v)$。Itô 的漂移部分为

$$
V_t+(r-q)SV_S+\kappa_v(\bar v-v)V_v
+\tfrac12vS^2V_{SS}+\rho\xi vS V_{Sv}+\tfrac12\xi^2vV_{vv}.
$$

中间的交叉项来自

$$
d\langle S,v\rangle=\rho\xi vS\,dt.
$$

在已选定 $\mathbb Q$ 下，要求折现价格为 martingale，得到

$$
\boxed{V_t+(r-q)SV_S+\kappa_v(\bar v-v)V_v
+\frac12vS^2V_{SS}+\rho\xi vS V_{Sv}
+\frac12\xi^2vV_{vv}-rV=0.}
$$

欧式 call 的终值为 $(S-K)^+$，与 $v$ 无关；这不意味着到期以前的 $V_v$ 为 0。$V_v$ 是对**方差状态**的敏感度，而常见 vega 对 volatility $\sqrt v$ 求导，两者满足 $V_{\sqrt v}=2\sqrt v\,V_v$，不能在公式中不改单位直接互换。

### BS7.4 股票 delta hedge 后还剩哪一项？

持有合约并做空 $\theta$ 股。其随机扩散损益可以写成

$$
\sqrt v\left[S(V_S-\theta)+\rho\xi V_v\right]dW^S
+\xi\sqrt{v(1-\rho^2)}V_v\,dW^\perp.
$$

使用通常的 partial delta $\theta=V_S$，仍有 $\xi\sqrt v V_v\,dW^v$。即使每一瞬间都能调整股票，也没有交易工具用来直接消掉这项独立方差冲击。

进一步，若只最小化**瞬时扩散损益的条件方差**，最佳股票股数为

$$
\boxed{\theta^{\mathrm{minvar}}=V_S+\frac{\rho\xi}{S}V_v.}
$$

推导并不需要抽象投影：直接最小化

$$
v\left\{[S(V_S-\theta)+\rho\xi V_v]^2+(1-\rho^2)\xi^2V_v^2\right\}dt
$$

关于 $\theta$ 的二次函数即可。最小残余方差为

$$
\boxed{\xi^2v(1-\rho^2)V_v^2dt.}
$$

这区分了三件事：partial delta hedge；利用 spot–vol 相关性的局部最小方差 hedge；逐路径完整复制。第二个不等于第三个，也不自动等于有交易费、终端效用、动态约束或违约跳跃下的全局最优策略。

如果再有一个可连续无摩擦交易的期权 $O(t,S,v)$，且 $O_v\ne0$，理论上可令

$$
\nu=\frac{V_v}{O_v},\qquad \theta=V_S-\nu O_S
$$

用 $\nu$ 份期权和 $\theta$ 股匹配两条扩散风险。这依赖于工具确实覆盖同一个方差因子、敏感度矩阵非退化以及可执行的交易假设。一个指数期权并不自动完美对冲单个发行人的 volatility；再有 default jump 时，两个扩散 hedge 也未必覆盖额外信用跳跃。

### BS7.5 Heston 为什么能半解析计算？从 PDE 推到两个 ODE

令 $x=\log S$，定义条件特征函数

$$
\psi(u,\tau)=\mathbb E^{\mathbb Q}[e^{iu\log S_T}\mid S_t=S,v_t=v],\qquad \tau=T-t.
$$

由于生成的漂移和协方差系数对 $v$ 是仿射的，尝试

$$
\psi(u,\tau)=\exp\big(iux+A(u,\tau)+B(u,\tau)v\big).
$$

将它代入条件期望的 backward equation，分别匹配常数项和 $v$ 项，得到

$$
\frac{\partial B}{\partial\tau}=\tfrac12\xi^2B^2+(\rho\xi iu-\kappa_v)B-\tfrac12(u^2+iu),\quad B(u,0)=0,
$$

$$
\frac{\partial A}{\partial\tau}=iu(r-q)+\kappa_v\bar v B,\quad A(u,0)=0.
$$

第一个是 Riccati 方程，第二个是积分。求出特征函数后，用 Fourier inversion 计算 call。例如在适当积分和真 martingale 条件下，定义

$$
P_2=\frac12+\frac1\pi\int_0^\infty\operatorname{Re}\left[\frac{e^{-iu\log K}\psi(u,\tau)}{iu}\right]du,
$$

$$
P_1=\frac12+\frac1\pi\int_0^\infty\operatorname{Re}\left[\frac{e^{-iu\log K}\psi(u-i,\tau)}{iu\psi(-i,\tau)}\right]du,
$$

就有 $V=S e^{-q\tau}P_1-Ke^{-r\tau}P_2$，其中 $\psi(-i,\tau)=S e^{(r-q)\tau}$。$P_1$ 和 $P_2$ 仍对应不同的加权概率，与 BS2 的两个截断项平行。

“闭式解”在这里通常仍需要数值积分，不是用一个瞬时 $\sqrt v$ 塞进普通 BS 公式。实现还要处理 $u=0$ 极限、积分截断、复对数分支及退化参数。也可以直接积分上述 ODE 以做公式核验。本报告推导了这些方程，但没有声称运行了 Heston 的 Fourier 引擎或其完整校准；后面的随机 volatility 实验使用一个可以精确核算的揭示模型。

### BS7.6 Local–stochastic volatility 及“只知道方差范围”

可以进一步设

$$
dS=(r-q)Sdt+L(t,S)\sqrt v S\,dW^S,
$$

并保留随机方差动态。在无股票跳跃、适当正则性和同一初始分布下，要使 vanilla 边际分布与某条 local-vol 曲面一致，典型条件是

$$
L^2(t,s)\mathbb E^{\mathbb Q}[v_t\mid S_t=s]=\sigma_{\mathrm{loc}}^2(t,s).
$$

可把它理解成：在给定股价状态内，匹配条件瞬时方差。左边的条件分布本身也依赖 $L$，因此不是用一个历史平均 variance 直接相除就完成校准。得到同样的 vanilla marginals，也不等于得到相同的跨期联合动态或可转债路径价值。这里是模型扩展方向，未做数值校准。

还有一种不同的问题：不是相信某个随机 variance 的概率模型，而是只知道 $\sigma\in[\underline\sigma,\overline\sigma]$。在连续无摩擦的稳健超复制框架里，对应的卖方 PDE 会含有

$$
\sup_{a\in[\underline\sigma,\overline\sigma]}\frac12a^2S^2V_{SS}.
$$

凸区间选高方差，凹区间选低方差；这是一种模型不确定性边界，不是 Heston 的风险中性平均，也没有消除 bid–ask 交易成本。尤其不能把“保守取高 vol”同时当成统计预测、成本补偿和全风险保证。

## BS8 · 把这些修正放回可转债，而不是另起一个无关模型 {#bs-convertible-extension}

### BS8.1 信用 × 随机方差 × 最优行权

用 $\lambda_d(t,S,v)$ 表示风险中性违约强度，即原报告使用的 $h$；这里换记号，避免与 BS3–BS4 的调仓间隔 $h$ 混淆。假设违约使股票瞬时损失比例 $\eta$，可转债退出存续状态并立即支付回收 $G(t,S,v)$，利率保持确定。

为了补偿股票的违约跳跃损失，违约前股票漂移是 $(r-q+\eta\lambda_d)S$，不是 $r-q$。结合 BS7，continuation 区域的存续价值满足

$$
\begin{aligned}
0={}&V_t+\tfrac12vS^2V_{SS}+\rho\xi vS V_{Sv}+\tfrac12\xi^2vV_{vv}\\
&+(r-q+\eta\lambda_d)SV_S+\kappa_v(\bar v-v)V_v\\
&-(r+\lambda_d)V+\lambda_dG+c(t).
\end{aligned}
$$

$c(t)$ 只用于**连续票息率的教学版本**。真实离散 coupon 应通过支付日前后的价值跳跃条件处理，不要又在 PDE 加连续 coupon、又逐日额外加实际 coupon。转股、holder put、issuer call 的障碍/事件条件仍按原报告第 5 节处理，只是状态从 $(t,S)$ 变成 $(t,S,v)$；通知期和历史触发还会增加状态。

这是在已指定联合 $\mathbb Q$ 模型下的无摩擦 continuation PDE。它同时要求股票期权的方差/尾部校准以及信用工具对 $\lambda_d,G$ 的约束。PDE 出现更多项不等于所有风险都能实际交易掉。[Ayache–Forsyth–Vetzal，公开论文](https://cs.uwaterloo.ca/~paforsyt/convert.pdf) 讨论股票违约、回收与可转债一致定价；这里在其违约收益思想上加入前述方差状态，属于本报告明确假设下的组合推导，而非声称该论文逐字给出此式。

### BS8.2 更复杂的动态并不会自动修复离散交易和 spread

上式改进的是理想化的 joint pricing dynamics。把它算出的 $V_S,V_v$ 交给一个每天调仓、按 bid/ask 成交的交易台，实际 hedge 仍需用 BS4 的账本。若要优化带费用的报价，状态通常还要包括现有股票/期权库存、现金及交易约束；它不再只是一个关于 $(t,S,v)$ 的线性价格 PDE。

对 long convertible、short $\theta$ 股，违约时的瞬时跳跃损益仍是

$$
\boxed{G-V+\theta\eta S.}
$$

对冲连续 spot diffusion、方差 diffusion、违约跳跃，是三个不同目标。前文 $\theta^{\mathrm{minvar}}$ 只优化 Heston 的连续扩散部分；加入违约后，完整的局部或终端风险最优股票头寸也可能变化。

实际流程应分别记录：理想化 mid/model value；指定策略的离散 hedge 风险；股票及期权执行成本；借券和融资；剩余 volatility、信用及跳跃风险。这样才能分清“提高 vol 拟合价格”到底是在改变风险中性动态，还是把几个不同的 residual 全藏进一个参数。

### BS8.3 每天更新 IV，为什么不是已经使用 stochastic vol 模型？

若每日标价为 $C_{\mathrm{BS}}(t,S,I_t)$，$I_t$ 是更新的 implied volatility，则它的 Itô 展开会多出

$$
C_I\,dI+\tfrac12C_{II}\,d\langle I\rangle+C_{SI}\,d\langle S,I\rangle.
$$

“每天换一个 IV”只是一条标价/重校准规则，除非同时定义 $I$ 与股票的联合动态，否则并没有给出这些项的分布，也不能推出一致的未来 hedge。更何况一整张 smile 曲面不一定能由单个 $I_t$ 代表。

固定模型参数的 partial delta、沿某条 smile 变动约定的 total delta、利用相关性的最小方差 hedge，应在风险报告中标明口径。对可转债，信用曲线随股票变动或重校准时也有同样的区别。

### BS8.4 四类模型放在一起比较

| 模型 | 未来瞬时 volatility | 无摩擦股票 + 现金能否完整复制？ | 主要限制 |
|:--|:--|:--|:--|
| 常数 BS | 固定 $\sigma$ | 标准一因子条件下可以 | 不能产生同期限 strike smile |
| 确定时间变化 | 已知 $\sigma(t)$ | 标准一因子条件下可以 | 只改变累计方差和期限结构 |
| Local vol | $\sigma_{\mathrm{loc}}(t,S_t)$ | 一因子、非退化及适当条件下可以 | 拟合 vanilla 不保证正确的跨期 smile 动态 |
| Heston / 一般 SV | 另有 variance 状态与独立风险 | 通常不可以，仅有股票无法覆盖独立因子 | 需要风险价格、额外 hedge 工具及联合校准 |

表内的“可以”都没有包含交易费、离散执行和违约跳跃。任何一行加入这些因素后，都要重新审计复制论证，而不是沿用表格里的结论。
