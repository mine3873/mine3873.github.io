---
title: "Score-based Generatvie Modeling Through Stochastic Differential Equations"
date: 2026-06-01 15:27:00 +0900
categories: [DL, Generative-model]
tags: [dl, generative-model, score-based,] 
math: true
mermaid: true
---

# Score-based Generatvie Modeling Through Stochastic Differential Equations

> [Score-based Generatvie Modeling Through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456)
{: .prompt-info}

## Indroduction

[DDPM](https://mine3873.github.io/posts/difussion/) 과 [NCSN](https://mine3873.github.io/posts/scorebased/) 두 논문을 이전에 살펴봤는데, 두 방식 모두 기존 데이터에 노이즈를 점차 섞어주는데, 이 과정을 그대로 거슬러서 노이즈를 제거해가며 기존 데이터에 접근하듯이 학습한다.  

여기서 DDPM 에서는 타임스텝 $t$, NCSN 에서는 노이즈 레벨 $l$ 과 같이 이산적인 상태 공간이 아닌, 연속형의 상태 공간에서, DDPM 의 목적함수가 각 스텝의 노이즈에서의 Score을 계산하는 것과 같으며, 따라서, NCSN 과 DDPM 두 방식을 **Score-based generative models** 로 일반화하여 나타낸다.  

논문에서는 이러한 Score-based generative models에 대해 기존보다 더 향상시키기 위해 Stochastic Differential Equations, SDEs 라는 것을 소개하는데, 한번 알아보자. 

![아자자잣](https://i.imgur.com/dRaOEXq.gif)

## 짧게 설명

> 이전 NCSN과 DDPM을 읽었다는 전제하에 짧게 설명한다... 절대 귀찮아서 또 설명하지 않는게 아니다..
{: .prompt-warning}

### Denoising Score Matching with Langevin Dynamics, SMLD 

> 자세하게는 [Generative Modeling by Estimating Gradients of the Data Distribution](https://mine3873.github.io/posts/scorebased/) 를 확인하자..
{: .prompt-info}

NCSN은 SMLD 프레임워크를 적용한 신경망으로 생각할 수 있는데, 일단 짧게 복습하면, 이 논문에선 약간의 표기의 차이가 있지만 노이즈를 $p\_{\sigma}(\tilde{x}\|x) = \mathcal{N}(\tilde{x}; x, \sigma^{2}\mathbf{I})$ 로, 노이즈가 섞인 데이터의 분포를 $p\_{\sigma}(\tilde{x}) = \int p\_{\text{data}}(x)p\_{\sigma}(\tilde{x}\|x)\,dz$ 로 나타낸다. 

각 노이즈 레벨 $\set{\sigma\_{i}}\_{i=1}^{N} \quad (\sigma\_{1} < \sigma\_{2} < \cdots < \sigma_{N})$ 에 대해서, 목적함수는 다음과 같이 기존 논문과 일치한다.

$$
\begin{align}
    \ell(\theta; \sigma_{i}) &= \mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\sigma_{i}}(\tilde{x}|x)} \left[ \| s_{\theta}(\tilde{x}, \sigma_{i}) - \nabla_{\tilde{x}} \log p_{\sigma_{i}}(\tilde{x} | x) \|_{2}^{2} \right] \\
    \theta^{\star} &= \underset{\theta}{\arg\min} \sum_{i=1}^{N} \lambda(\sigma_{i}) \ell(\theta; \sigma_{i}) = \underset{\theta}{\arg\min} \sum_{i=1}^{N} \sigma^{2} \ell(\theta; \sigma_{i})
\end{align}
$$

또한 $p\_{\sigma\_{i}}(x)$ 에서 $M$ 단계의 샘플링 과정은 다음과 같다. 

$$
x_{i}^{m} = x_{i}^{m-1} + \frac{\epsilon_{i}}{2} s_{\theta^{\star}}(x_{i}^{m-1}, \sigma_{i}) + \sqrt{\epsilon_{i}} z_{i}^{m} \quad (m = 1,2,...,M)
$$

이때 $\epsilon\_{i} > 0$ 는 학습률, $z\_{i}^{m} \sim \mathcal{N}(0, \mathbf{I})$ 은 표준 정규 분포이다.

초기 위치 $x\_{N}^{0} \sim \mathcal{N}(x \| 0, \sigma_{N}^{2}\mathbf{I})$ 부터 시작하여, $i$ 노이즈 레벨에서의 최종 위치가 $i-1$ 노이즈 레벨에서의 초기 위치가 되는 식으로 반복된다.  

각 노이즈 레벨에서의 위치 갱신의 수가 무한해지고 $M \to \infty$, 학습률이 매우 작아 정교해지면 $\epsilon\_{i} \to 0$, 위 식의 최종 위치를 반환하는 분포는 $p\_{\sigma\_{1}}(x) \approx p\_{\text{data}}(x)$ 와 같이 실제 데이터 분포와 같아진다. 


### Denoising Diffusion Probabilistic Models, DDPM

> 자세하게는 [Denoising Diffusion Probabilistic Models](https://mine3873.github.io/posts/difussion/) 를 확인하자..
{: .prompt-info}

DDPM 도 마찬가지로 조금 표기의 차이가 있지만, SMLD와 비슷하게 노이즈가 섞인 데이터의 분포를 $p\_{\alpha\_{i}}(\tilde{x}) = \int p\_{\text{data}}(x)p\_{\alpha\_{i}}(\tilde{x}\|x)\,dz$ 로 나타낸다. 여기서 DDPM 의 Markov-chain의 각 과정이 $p(x\_{i}\|x\_{i-1}) = \mathcal{N}(x\_{i}; \sqrt{1 - \beta_{i}}x\_{i-1}, \beta\_{i}\mathbf{I})$ 로 정의되고, $x\_{0} \sim p\_{\text{data}}(x)$ 에서 Closed-form으로 $x\_{t} \sim p\_{\alpha\_{i}}(x\_{i}\|x\_{0}) = \mathcal{N}(x\_{i}; \sqrt{\alpha\_{i}}x\_{0}, (1-\alpha\_{i})\mathbf{I})$ 으로 정의된다. 이때 $\alpha\_{i} = \prod\_{j=1}^{i} (1-\beta\_{j})$ 이다. 

이제 다음과 같이 서커스를 해보자.

$$
\begin{align}
    p_{\alpha_{i}}(x_{i}|x_{0}) &= \mathcal{N}(x_{i}; \sqrt{\alpha_{i}}x_{0}, (1-\alpha_{i})\mathbf{I}) \\
    \log p_{\alpha_{i}}(x_{i}|x_{0}) &= - \frac{(x_{i} - \sqrt{\alpha_{i}}x_{0})^{2}}{2(1-\alpha_{i})} \\
    \nabla_{x_{i}} \log p_{\alpha_{i}}(x_{i}|x_{0}) &= - \frac{x_{i} - \sqrt{\alpha_{i}}x_{0}}{1-\alpha_{i}} \quad (x_{i} = \sqrt{\alpha_{i}}x_{0} + \sqrt{1-\alpha_{i}}\epsilon)\\
    \therefore \nabla_{x_{i}} \log p_{\alpha_{i}}(x_{i}|x_{0})&= - \frac{\sqrt{1-\alpha_{i}}\epsilon}{1-\alpha_{i}} = - \frac{\epsilon}{\sqrt{1-\alpha_{i}}} \\
    \therefore s_{\theta}(x_{i}, i) &= - \frac{\epsilon_{\theta}(x_{i}, i)}{\sqrt{1-\alpha_{i}}} \\
\end{align}
$$

SMLD 와 표기를 통일하기 위해 $x_{i} \to \tilde{x}, x_{0} \to x$ 로 표기하고, DDPM의 Simplified 목적함수를 가져와서 서커스를 해보자.

$$
\begin{align}
    \ell(\theta; i) &=  \mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\alpha_{i}}(x_{i}|x)}[\|\epsilon - \epsilon_{\theta}(\tilde{x}, i)\|_{2}^{2}] \quad (\tilde{x} = x_{i} = \sqrt{\alpha_{i}}x_{0} + \sqrt{1-\alpha_{i}}\epsilon)\\
    (1-\alpha_{i})\ell(\theta; i) &= (1-\alpha_{i})\mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\alpha_{i}}(x_{i}|x)}\left[\left|\left|\frac{\epsilon}{\sqrt{1-\alpha_{i}}} - \frac{\epsilon_{\theta}(\tilde{x}, i)}{\sqrt{1-\alpha_{i}}}\right|\right|_{2}^{2}\right] \\
    &= (1-\alpha_{i})\mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\alpha_{i}}(x_{i}|x)}[\| s_{\theta}(\tilde{x}, i) - \nabla_{x_{i}} \log p_{\alpha_{i}}(\tilde{x}|x)\ \|_{2}^{2}] \\
    \theta^{\star} &= \underset{\theta}{\arg\min} \sum_{i=1}^{N} (1-\alpha_{i})\ell(\theta; i)
\end{align}
$$

초기 데이터 $x\_{N} \sim \mathcal{N}(0, \mathbf{I})$ 부터 시작하여 샘플링의 각 Markov chain 과정은 다음과 같다. 

$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_{i}}}(x_{i} + \beta_{i}s_{\theta^{\star}}(x_{i}, i)) + \sqrt{\beta_{i}}z_{i} \quad (i = M,N-1,...,1,  z_{i} \sim (0, \mathbf{I}))
$$

이전에 작성한 [NCSN](https://mine3873.github.io/posts/scorebased/) 글과 위 식의 전개를 통해서, 다음과 같음을 알 수 있다.
- $\mathbb{E}[\|\|\nabla\_{x} \log p\_{\sigma\_{i}} (\tilde{x}\|x)\|\|\_{2}^{2}] \propto \frac{1}{\sigma\_{i}^{2}}$
- $\mathbb{E}[\|\|\nabla\_{x} \log p\_{\alpha\_{i}} (\tilde{x}\|x)\|\|\_{2}^{2}] \propto \frac{1}{1-\alpha\_{i}}$  

이를 통해 노이즈 레벨 $i$ 에 상관없이 각 손실함수 $\ell(\theta; \sigma\_{i}), \ell(\theta; i)$ 는 모두 같은 비중으로 모델이 다룰 것이다.

## Score-based Generative Modeling With SDEs

| ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 1_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 2_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 3_ |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 4_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 5_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 6_ |

앞선 노이즈를 추가해가는 과정을 일반화해서, $t \in [0,T]$ 의 연속형의 $t$ 에 따라 연속적으로 증가해가는 과정으로 일반화할 수 있다.

이러한 연속적인 시간(단계)의 흐름에서 데이터 $x$ 의 변화량을 **Stochastic Differential Fquations, SDEs**라는 것으로 정의하는데. SDEs 란 단순히 기존 미분방정식에서 하나 이상의 확률적 샘플링이 요구되는 항이 포함된 것인데, 아무튼 알아보자. 

### Perturbing Data With SDEs

일단 들어가기 전에 $x(t)$ 를 $t$ 에서의 데이터, $p\_{t}(x)$ 를 $x(t)$ 의 확률 밀도, $p\_{st}(x(t)\|x(s))$ 를 $x(s)$ 에서 $x(t)$ 로의 전이 커널로 나타낸다.  

데이터 분포의 샘플 $x(0) \sim p_{0}$ 과, 완전히 Noisy 해지는 사전 분포의 샘플 $x(T) \sim p_{T}$ 에 대해서, 기존 Diffusion Process $\set{x(t)}\_{t=0}^{T}$ 를 연속형 $t \in [0,T]$ 에서 다시 정의해야 하는데 이를 다음과 같이 이토(Itô) SDE 로 나타낼 수 있다. 

$$
dx = f(x,t)dt + g(t)dw
$$

여기서 $w$ 는 Standard Wiener Process(표준 브라운 운동), $f(x,t): \mathbb{R}^{d} \to \mathbb{R}^{d}$ 는 $x(t)$ 의 드리프트(Drift) 계수, $g(t): \mathbb{R} \to \mathbb{R}$ 는 확산(Diffusion) 계수이다.

뭐 카트라이더마냥 드리프트 어쩌구 뭐 솰라솰라 브라운 박사 저쩌구 거시기 뭔 소린지 도통 모르겠는데, SDE란 뭔지, 자세한건 [여기를 확인하자..](#general-stochastic-differential-equations)

아무튼 그래서 연속형 $t$ 에 대해서 Diffusion Process를 위와 같은 형태로 정의했다는 것이다. 각 계수들을 어떻게 정의할지는 다양한 방법이 있는데, 논문에서 어떻게 정의했는지 예시들을 후에 알려준다고 하니 손가락 빨면서 일단 넘어가자.

### Generating Samples by Reversing The SDE

이제 Diffusion Process의 반대인 $x(T) \to x(0)$ 의 샘플링 과정도 연속형 $t$ 에 대해서 정의해야 하는데, Anderson 형님의 Reverse-time diffusion equation models 논문에서, Diffusion Process의 반대 과정도 Diffusion Process이며, 기존 Diffusion Process와 마찬가지로 반대로 거슬러가는 $t$ 에 대해 다음과 같이 SDE로 나타낼 수 있다는 것을 증명하셨다.

$$
dx = [f(x,t) - g(t)^{2}\nabla_{x}\log p_{t}(x)]dt + g(t)d\bar{w}
$$

여기서 $\bar{w}$ 는 마찬가지로 Standard Wiener process($T \to 0$ 에서의 표준 브라운 운동)이고, $dt$ 는 아~주 작은 음의 시간 변화량이다.  

기존 Diffusion Process에서 $f(x,t)$ 와 $g(t)$ 는 이미 정의됐고.. 결국 $\nabla\_{x}\log p\_{t}(x)$ 만 알고 있으면, 위 식을 통한 Reverse Diffusion Process로 샘플링을 할 수 있다는 것이다. 

### Estimating Scores For The SDE

그럼 샘플링을 위해서 $\nabla\_{x}\log p\_{t}(x)$ 을 추정할 모델 $s_{\theta}(x,t)$ 을 학습해야 하는데, 앞서 살펴본 SMLD 와 DDPM 의 각 목적함수를 최적화하는 최적의 파라미터 $\theta^{\star}$ 를 다음과 같이 일반화하여 나타낸다.

$$
\theta^{\star} = \underset{\theta}{\arg\min}\, \mathbb{E}_{t}\left[\lambda(t)\mathbb{E}_{x(0)}\mathbb{E}_{x(t)|x(0)}\left[ \|s_{\theta}(x,t) - \nabla_{x(t)} \log p_{0t}(x(t)|x(0)) \|_{2}^{2} \right]\right]
$$

이때 $\lambda: [0,T] \to \mathbb{R}^{+}$ 인 가중치 함수이고, $t \sim \mathcal{U}(0,T)$ 로 정의된다. Denoising Score Matching 의 형태로 정의된 것을 볼 수 있는데, 다른 Score Matching 방식도 상관없다고 한다.

위 식에 대한 최적해 $\theta^{\star}$ 에 대해서, 충분한 데이터와 시간만 있으면,  모든 $t$ 와 $x$ 에 대해 모델 $s_{\theta^{\star}}(x,t) = \nabla\_{x}\log p\_{t}(x)$ 를 만족하며, SMLD와 DDPM 에서 살펴봤듯이, $\mathbb{E}[\|\|\nabla\_{x(t)} \log p\_{0t}(x(t)\|x(0)) \|\|_{2}^{2}] \propto \lambda(t)^{-1}$ 이 되도록 $\lambda(t)$ 를 정의하여 연속적인 $t$ 에 따라 일관된 비중으로 모델이 학습하도록 한다. 

아무튼 위 식을 통해 모델을 학습하기 위해선, 전이 커널 $p\_{0t}(x(t)\|x(0))$ 의 값을 알아야 할텐데, Diffusion Process 에서의 $f(x,t)$ 가 아핀(Affine) 함수인 경우, 모든 전이 커널은 가우시안 분포가 된다. 따라서 DDPM의 논문에서 봤던 것처럼, $x(t)$ 를 샘플링하는데 Closed-Form 으로 할 수 있다는 것이다.  

$f(x,t)$ 가 아핀 함수가 아닌, 일반적인 경우엔 뭐.. Kolmogorov's forward equation 이란걸 사용한다는데... 논문에선 그냥 앞서 본 Reverse Diffusion Process 대로 $p\_{0t}(x(t)\|x(0))$ 를 샘플링하고, 위 식을 Sliced Score Matching 의 형태로 바꿔서 학습을 진행한다고 한다. 

### VE, VP, sub-VP SEDs

#### VE SDE

앞선 SDEs 에 기반해서, SMLD와 DDPM는 이산화된 SDE로 생각될 수 있다. 

SMLD 부터 살펴보면, 총 $N$ 개의 노이즈 레벨에 대해서 각 커널 $p\_{\sigma\_{i}}(x \| x\_{0})$ 은 다음 Markov Chain 의 $x_{i}$ 의 분포와 일치한다.

$$
\begin{align}
    x_{i} &\sim p_{\sigma_{i}}(x | x_{0}) = \mathcal{N}(x_{i}; x_{0}, \sigma_{i}^{2}\mathbf{I}) \\
    x_{i} - x_{i-1} &\sim \mathcal{N}(x_{i} - x_{i-1}; x_{0} - x_{0}, (\sigma_{i}^{2} - \sigma_{i-1}^{2})\mathbf{I})\\
    &= \mathcal{N}(x_{i} - x_{i-1}; 0, (\sigma_{i}^{2} - \sigma_{i-1}^{2})\mathbf{I}) \\
    x_{i} - x_{i-1} &= \sqrt{\sigma_{i}^{2} - \sigma_{i-1}^{2}}z_{i-1} \\
    \therefore x_{i} &= x_{i-1} + \sqrt{\sigma_{i}^{2} - \sigma_{i-1}^{2}}z_{i-1} \quad i = 1,...,N, \quad z_{i-1} \sim \mathcal{N}(0, \mathbf{I})
\end{align}
$$

이제 이를 우리가 앞서 살펴본 SDE의 형태로 나타내기 위해, 이산형 노이즈 레벨 $i$ 가 아닌, 연속형 레벨 $t \in [0,1]$ 로 바꿔서 나타내보자. 그럼 다음과 같다. 

$$
\begin{align}
    x(t + \nabla t) - x(t) &= \sqrt{\sigma^{2}(t + \nabla t) - \sigma^{2}(t)} z(t) \\
    \frac{dx}{dt} \Delta t &= \sqrt{\frac{d[\sigma^{2}(t)]}{dt}\Delta t} \frac{z(t)}{\sqrt{\Delta t}} \\
    \therefore dx &= \sqrt{\frac{d[\sigma^{2}(t)]}{dt}} dw \quad (z(t) \approx \frac{\Delta w}{\sqrt{\Delta t}},\Delta w \sim \mathcal{N}(0, \Delta t \mathbf{I}))
\end{align}
$$

$f(x,t) = 0$, $g(t) = \sqrt{\frac{d[\sigma^{2}(t)]}{dt}}$ 로 SMLD 에서 정의된다.  

따라서 $t \to \infty$ 일 때 $\sigma{t}$ 의 값도 덩달아 커지므로, 이를 **Variance Exploding (VE) SDE** 라고 한다. 

#### VP SDE

DDMP의 각 커널 $\set{p\_{\alpha\_{i}}(x \| x\_{0})}\_{i=1}^{N}$ 에 대해서도 살펴보자. 

$$
\begin{align}
    x_{i} &= \sqrt{1-\beta_{i}}x_{i-1} + \sqrt{\beta_{i}}z_{i-1} \quad i=1,...,N 
\end{align}
$$

마찬가지로 연속형 레벨 $t \in [0,1]$ 로 바꿔서 나타내면 다음과 같다. 

$$
\begin{align}
    x_{i} &= \sqrt{1-\beta_{i}}x_{i-1} + \sqrt{\beta_{i}}z_{i-1} \\
    x_{i} &= \left(1 - \frac{\beta_{i}}{2}\right)x_{i-1} + \sqrt{\beta_{i}}z_{i-1} \quad \left(\sqrt{1 - x} \approx 1 - \frac{x}{2}\right) \\
    dx &= - \frac{\beta(t)\Delta t}{2}x(t) + \sqrt{\beta(t)\Delta t}z(t) \\
    dx &= - \frac{1}{2}\beta(t)x(t)\Delta t + \sqrt{\beta(t)} dw \quad (dw \approx \sqrt{\Delta t}z(t)) \\
    \therefore dx &= - \frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)} dw 
\end{align}
$$

$f(x,t) = - \frac{1}{2}\beta(t)x$, $g(t) = \sqrt{\beta(t)}$ 로 DDPM 에서 저렇게 정의된다.  

이전 블로그 글에서도 언급했지만, 이전 $\text{Var}(x_{i-1}) = 1$ 인 경우에, $(\sqrt{1-\beta_{i}})^{2}\text{Var}(x_{i-1}) + (\sqrt{\beta_{i}})^{2}\text{Var}(z_{i-1}) = 1 \quad (\text{Var}(x_{0}) = 1)$ 이므로, 위 식은 **Variance Preserving (VP) SDE** 라고 한다. 

#### sub-VP SDE

논문에서는 이러한 VP SDE 를 조금 변형하여, 마찬가지로 분산의 크기를 제한한 새로운 형태의 SDE 를 소개하는데, 다음 식을 **sub-VP SDE** 라고 한다. 

$$
dx = - \frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)(1 - \exp(-2 \int_{0}^{t} \beta(s)ds))}dw
$$

아무튼 소개한 VE, VP, sub-VP SDE 모두 드리프트 계수는 $f(x,t)$ 가 아핀 함수로 정의되는데, 따라서 모든 전이 커널 $p\_{0t}(x(t)\|x(0))$ 은 모두 가우시안 분포로 정의되어, Closed-forms 으로 계산할 수 있다. 

## Sampling

그래서 앞에서 

### General-purpose Numerical SDE Solvers

앞서서 연속형의 $t$ 에 대해서 $dx = \cdots$ 형식으로 SDE 들을 뭐.. 정의내렸는데, 아무튼 이를 컴퓨터에서 샘플링을 진행하기 위해서는 일단 SDE 들을 다시 이산형으로 쪼개야한다... 이렇게 스텝들로 쪼개서 근사치를 구하는 걸 **Numerical Solver** 라고 한다.  

위에서 본 DDPM 에서의 샘플링 과정 식 그거... 그거를 Ancestral Sampling 이라고 하는데, 아무튼 그거도 Numerical Solver 의 한 종류이다. 이거 말고도 뭐.. Euler-Maruyama, Stochastic Runge-Kutta 등등이 있다고 한다. 

뭐가 많아도 드럽게 많다. 아무튼 뭐가 많은데.. 결국 중요한 점은 모든 SDE에 통용되는 것이 아니라, 각 SDE 식 별로 달라서 형태에 맞게 유도해야 한다는 것이다..

그래서 논문에서는 간단하게, 원래 방향의 Diffusion Process 에서 쪼갠 방식 그대로 Reverse Diffusion Process 에서도 동일하게 쪼개는 방식을 소개하는데, 이 방식을 **Reverse Diffusion Samplers** 라고 한다. 

### Predictor-Corrector (PC) Samplers

요거는 앞서 설명한 Numerical Solvers 들을 통해 다음 스텝을 계산하고, 이 계산된 값을, $s\_{\theta^{\star}}(x,t) \approx \nabla\_{x} \log p\_{t}(x)$ 를 이용하여 Score0-based MCMC 를 통해 교정하는 과정을 반복하는 과정인데, 이를 **Predictor-Corrector (PC) Sampler** 라고 한다.  

앞서 살펴본 SMLD는 Corrector (Langevin Dynamics)만 사용한 경우이고, DDPM은 Predictor (Ancestral Sampling)만 사용한 경우의 PC Sampler 의 한 경우로 볼 수 있다. 

### Probability ODE

지금까지 알아본 SDE 는... 결국 $dw$ 항의 랜덤 샘플링을 통해 이동 궤적이 음... 무작위의 지그재그? 형태에 가까웠었다. 근데!!! 논문의 저자가 보니까 무작위 샘플링이 없는, 즉 $dw$ 항이 없는 Ordinary Differential Equations, ODEs 가 사실 같은 확률 분포 궤적을 그린다는 사실을 알아냈다. 이 식은 다음과 같다. 

$$
dx = \left[f(x,t) - \frac{1}{2}g(t)\nabla_{x}\log p_{t}(x)\right]dt
$$

식을 보면 확률적 샘플링 과정이 아에 사라지고, 결정론적으로 샘플링을 할 수 있게 되는데, 이를 **Probability Flow ODE** 라고 한다. 

이를 통해서, DDPM 에서의 ELBO를 통한 기존 우도함수의 근사치를 손실함수로 쓰는게 아닌, 정확한 손실함수를 계산할 수 있게 되는 것이다.  

또한 ODE는 결정론적이므로, 입력 데이터 $x(0)$ 을 $x(T)$ 로 인코딩하고, 다시 $x(0)$ 으로 디코딩하면 정확히 입력 데이터와 같은 형태가 나온다. 




## 찌라시시

![살려줘](https://i.imgur.com/izzuaSg.gif){: width="400"}

무량공처로 지금 뇌가 엉엉방방하다. 그냥 글 흐름이 개념 개념 개념개념개념개념개념 연속인데, 이게 무량공처다.  
일단 초안으로 놔두고.. 본문 계속 다듬고 추가설명 부분도 아직 비워뒀는데, 내일 끝내겠다. 
...나중에 다시 살펴볼 때 논문 안보고 이것만 보고 다 알 수 있게 작성하려다 보니.. 뭔가 포스터마다 겁나게 내용이 긴거 같은데, 미안하다. 글도 쓰다보면 언젠간 나아지겠지.

## 추가 설명 

### General Stochastic Differential Equations






