---
title: "Score-based Generative Modeling Through Stochastic Differential Equations"
date: 2026-06-01 15:27:00 +0900
categories: [DL, Generative-model]
tags: [dl, generative-model, score-based, diffusion] 
math: true
mermaid: true
---

# Score-based Generative Modeling Through Stochastic Differential Equations

> [Score-based Generative Modeling Through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456)
{: .prompt-info}

## Introduction

[DDPM](https://mine3873.github.io/posts/difussion/) 과 [NCSN](https://mine3873.github.io/posts/scorebased/) 두 논문을 이전에 살펴봤는데, 두 방식 모두 기존 데이터에 노이즈를 점차 섞어가며, 이 과정을 그대로 역재생하듯 노이즈를 제거해가며 데이터 분포를 학습한다. 

여기서 DDPM 에서는 타임스텝, NCSN 에서는 노이즈 레벨과 같이 이산적인 상태 공간이 아닌, 연속형의 상태 공간에서, DDPM 의 목적함수가 각 스텝의 노이즈에서의 Score을 계산하는 것과 비슷한데, NCSN 과 DDPM 이 두 방식을 **Score-based generative models** 로 일반화하여 하나의 형태로 나타낼 수 있다.  

논문에서는 이러한 형태를 **Stochastic Differential Equations, SDEs** 라는 것으로 나타내는데, 한번 알아보자. 

![아자자잣](https://i.imgur.com/dRaOEXq.gif)

## 짧게 설명

> 이전 NCSN과 DDPM을 읽었다는 전제하에 짧게 설명한다... 절대 귀찮아서 또 설명하지 않는게 아니다..
{: .prompt-warning}

### Denoising Score Matching with Langevin Dynamics, SMLD 

> 자세하게는 [Generative Modeling by Estimating Gradients of the Data Distribution](https://mine3873.github.io/posts/scorebased/) 를 확인하자..
{: .prompt-info}

우리가 전에 살펴본 Noise Conditional Score Networks, NCSN은 Denoising Score Matching with Langevin Dynamics, SMLD 프레임워크를 적용한 신경망으로, 여기선 SMLD를 기준으로 설명한다. 

일단 짧게 복습하면, 이 논문에선 약간의 표기의 차이가 있지만 노이즈를 $p\_{\sigma}(\tilde{x}\|x) = \mathcal{N}(\tilde{x}; x, \sigma^{2}\mathbf{I})$ 로, 노이즈가 섞인 데이터의 분포를 $p\_{\sigma}(\tilde{x}) = \int p\_{\text{data}}(x)p\_{\sigma}(\tilde{x}\|x)\,dx$ 로 나타낸다. 

각 노이즈 레벨 $\set{\sigma\_{i}}\_{i=1}^{N} \quad (\sigma\_{1} < \sigma\_{2} < \cdots < \sigma_{N})$ 에 대해서, 모델 $s\_{\theta}$ 의 최적 파라미터 $\theta^{\star}$ 는 다음과 같다.

$$
\begin{align}
    \ell(\theta; \sigma_{i}) &= \mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\sigma_{i}}(\tilde{x}|x)} \left[ \| s_{\theta}(\tilde{x}, \sigma_{i}) - \nabla_{\tilde{x}} \log p_{\sigma_{i}}(\tilde{x} | x) \|_{2}^{2} \right] \\
    \theta^{\star} &= \underset{\theta}{\arg\min} \sum_{i=1}^{N} \lambda(\sigma_{i}) \ell(\theta; \sigma_{i}) = \underset{\theta}{\arg\min} \sum_{i=1}^{N} \sigma^{2} \ell(\theta; \sigma_{i})
\end{align}
$$

일단 Denoising Score Matching 의 형태로 나타냈다...  

또한 $p\_{\sigma\_{i}}(x)$ 에서 $M$ 단계의 샘플링 과정은 다음과 같다. 

$$
x_{i}^{m} = x_{i}^{m-1} + \frac{\epsilon_{i}}{2} s_{\theta^{\star}}(x_{i}^{m-1}, \sigma_{i}) + \sqrt{\epsilon_{i}} z_{i}^{m} \quad (m = 1,2,...,M)
$$

이때 $\epsilon\_{i} > 0$ 는 학습률, $z\_{i}^{m} \sim \mathcal{N}(0, \mathbf{I})$ 이다.

초기 위치 $x\_{N}^{0} \sim \mathcal{N}(x \| 0, \sigma_{N}^{2}\mathbf{I})$ 부터 시작하여, $x_{i+1}^{0} = x_{i}^{M}$, 즉 현재 노이즈 레벨에서 최종 위치가 다음 레벨의 초기 위치가 되는 식으로 반복된다.  
~~자세한건 말했듯이 앞서 링크 걸은 블로그 글을 다시 보자...글이 너무 길어져서 좀 압축하려고 그런거다...~~

### Denoising Diffusion Probabilistic Models, DDPM

> 자세하게는 [Denoising Diffusion Probabilistic Models](https://mine3873.github.io/posts/difussion/) 를 확인하자..
{: .prompt-info}

DDPM 도 마찬가지로 조금 표기의 차이가 있지만, SMLD와 비슷하게 노이즈가 섞인 데이터의 분포를 $p\_{\alpha\_{i}}(\tilde{x}) = \int p\_{\text{data}}(x)p\_{\alpha\_{i}}(\tilde{x}\|x)\,dx$ 로 나타낸다. 여기서 DDPM 의 Markov-chain의 각 과정이 $p(x\_{i}\|x\_{i-1}) = \mathcal{N}(x\_{i}; \sqrt{1 - \beta_{i}}x\_{i-1}, \beta\_{i}\mathbf{I})$ 로 정의되고, $x\_{0} \sim p\_{\text{data}}(x)$ 에서 Closed-form으로 $x\_{t} \sim p\_{\alpha\_{i}}(x\_{i}\|x\_{0}) = \mathcal{N}(x\_{i}; \sqrt{\alpha\_{i}}x\_{0}, (1-\alpha\_{i})\mathbf{I})$ 으로 정의된다. 이때 $\alpha\_{i} = \prod\_{j=1}^{i} (1-\beta\_{j})$ 이다. 

이제 다음과 같이 서커스를 해보자.

$$
\begin{align}
    p_{\alpha_{i}}(x_{i}|x_{0}) &= \mathcal{N}(x_{i}; \sqrt{\alpha_{i}}x_{0}, (1-\alpha_{i})\mathbf{I}) \\
    \log p_{\alpha_{i}}(x_{i}|x_{0}) &= - \frac{(x_{i} - \sqrt{\alpha_{i}}x_{0})^{2}}{2(1-\alpha_{i})} \\
    \nabla_{x_{i}} \log p_{\alpha_{i}}(x_{i}|x_{0}) &= - \frac{x_{i} - \sqrt{\alpha_{i}}x_{0}}{1-\alpha_{i}} \quad (x_{i} - \sqrt{\alpha_{i}}x_{0} = \sqrt{1-\alpha_{i}}\epsilon)\\
    \therefore \nabla_{x_{i}} \log p_{\alpha_{i}}(x_{i}|x_{0})&= - \frac{\sqrt{1-\alpha_{i}}\epsilon}{1-\alpha_{i}} = - \frac{\epsilon}{\sqrt{1-\alpha_{i}}} \\
    \therefore s_{\theta}(x_{i}, i) &= - \frac{\epsilon_{\theta}(x_{i}, i)}{\sqrt{1-\alpha_{i}}} \\
\end{align}
$$

SMLD의 목적함수와 표기를 통일하기 위해 $x_{i} \to \tilde{x}, x_{0} \to x$ 로 나타내고, DDPM의 목적함수를 가져와서 다음과 같이 서커스를 해보자.

$$
\begin{align}
    \ell(\theta; i) &=  \mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\alpha_{i}}(x_{i}|x)}[\|\epsilon - \epsilon_{\theta}(\tilde{x}, i)\|_{2}^{2}] \quad (\tilde{x} = x_{i} = \sqrt{\alpha_{i}}x_{0} + \sqrt{1-\alpha_{i}}\epsilon)\\
    (1-\alpha_{i})\ell(\theta; i) &= (1-\alpha_{i})\mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\alpha_{i}}(x_{i}|x)}\left[\left|\left|\frac{\epsilon}{\sqrt{1-\alpha_{i}}} - \frac{\epsilon_{\theta}(\tilde{x}, i)}{\sqrt{1-\alpha_{i}}}\right|\right|_{2}^{2}\right] \\
    &= (1-\alpha_{i})\mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{p_{\alpha_{i}}(x_{i}|x)}[\| s_{\theta}(\tilde{x}, i) - \nabla_{x_{i}} \log p_{\alpha_{i}}(\tilde{x}|x)\ \|_{2}^{2}] \\
    \theta^{\star} &= \underset{\theta}{\arg\min} \sum_{i=1}^{N} (1-\alpha_{i})\ell(\theta; i)
\end{align}
$$

짜잔... 앞선 SMLD의 것과 모양이 비슷해졌다! DDPM 논문에서 왜 최종 Simplified 목적함수가 Denoising Score matching과 같다고 언급했던건지 지금 알 수 있게 된거다!  
~~그런 말을 했었다..~~

초기 데이터 $x\_{N} \sim \mathcal{N}(0, \mathbf{I})$ 부터 시작하여 샘플링의 각 Markov chain 과정은 다음과 같다. 

$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_{i}}}(x_{i} + \beta_{i}s_{\theta^{\star}}(x_{i}, i)) + \sqrt{\beta_{i}}z_{i} \quad (i = M,N-1,...,1,  z_{i} \sim (0, \mathbf{I}))
$$

SMLD, DDPM 모두 개별 목적함수들의 가중치합으로 전체 목적함수가 정의되는데, 여기서 각 가중치들은 다음을 만족한다.
- $\mathbb{E}[\|\|\nabla\_{x} \log p\_{\sigma\_{i}} (\tilde{x}\|x)\|\|\_{2}^{2}] \propto \frac{1}{\sigma\_{i}^{2}}$
- $\mathbb{E}[\|\|\nabla\_{x} \log p\_{\alpha\_{i}} (\tilde{x}\|x)\|\|\_{2}^{2}] \propto \frac{1}{1-\alpha\_{i}}$  

이를 통해 노이즈 레벨 $i$ 에 상관없이 각 손실함수 $\ell(\theta; \sigma\_{i}), \ell(\theta; i)$ 는 모두 동일한 비중으로 모델이 다루게 된다. 다시말해서, 특정 노이즈 레벨에만 집중하지 않는다는 것이다.

## Score-based Generative Modeling With SDEs

| ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 1_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 2_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 3_ |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 4_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 5_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 6_ |

앞서 살펴본 것들을 일반화해서, $i = 1,2,...,N$ 에서 $t \in [0,T]$ 의 연속형 노이즈 레벨로 일반화할 수 있다.  

이러한 연속적인 시간(단계)의 흐름에서 데이터 $x$ 의 변화량을 나타내야하는데, 이는 **확률 미분 방정식 (Stochastic Differential Equations, SDEs)** 로 정의할 수 있다..  

...SDEs 란 단순히 하나 이상의 확률적 샘플링이 요구되는 항을 포함하는 미분 방정식이란다... 나도 여기서 처음 본 개념인데, 개념은 간단하다...

### Perturbing Data With SDEs

일단 들어가기 전에 $x(t)$ 를 $t$ 에서의 데이터, $p\_{t}(x)$ 를 $x(t)$ 의 확률 밀도, $p\_{st}(x(t)\|x(s))$ 를 $x(s)$ 에서 $x(t)$ 로의 전이 커널로 나타낸다. 그냥 앞서 본 노이즈 섞인 데이터, 노이즈 다 똑같은데, 앞으로 연속형 $t$ 에 대해 다뤄야 하는거라 표기만 이렇게 나타낸거다.  

데이터 분포의 샘플 $x(0) \sim p_{0}$ 과, 완전히 Noisy 해지는 사전 분포의 샘플 $x(T) \sim p_{T}$ 에 대해서, 기존 Diffusion Process $\set{x(t)}\_{t=0}^{T}$ 를 연속형 $t \in [0,T]$ 에서 다시 정의해야 하는데, 이를는 다음과 같이 일반적인 SDE의 형태로 나타낸다.

$$
dx = f(x,t)dt + g(t)dw
$$

여기서 $w$ 는 Standard Wiener Process(표준 브라운 운동) 으로 $dw \sim \mathcal{N}(0, \sqrt{t}\mathbf{I})$ 이고, $f(x,t): \mathbb{R}^{d} \to \mathbb{R}^{d}$ 는 $x(t)$ 의 드리프트(Drift) 계수, $g(t): \mathbb{R} \to \mathbb{R}$ 는 확산(Diffusion) 계수이다. 여기서 확률 샘플링 과정이 포함되는 $dw \sim \mathcal{N}(0, \sqrt{t}\mathbf{I})$ 이 항으로 인해, 이식 Stochastic Differential Equation, SDE 라고 하는 것이다.  

뭐 카트라이더마냥 드리프트 어쩌구 뭐 솰라솰라 브라운 박사 저쩌구 거시기 뭔 소린지 도통 모르겠는데, 일단 연속형 $t$ 에 대해서 Diffusion Process의 기본 형태를 위와 같이 정의한거고, 결정론적 $f(x,t), g(t)$ 를 어떻게 정의하는지에 따라, 우리가 앞서 살펴본 SMLD와 DDPM도 이 형태로 나타낼 수 있다.  

기존 SMLD, DDPM 에서 샘플링의 Markov-chain 식은 다음과 같았다.

$$
\begin{align}
    x_{i}^{m} &= x_{i}^{m-1} + \frac{\epsilon_{i}}{2} s_{\theta^{\star}}(x_{i}^{m-1}, \sigma_{i}) + \sqrt{\epsilon_{i}} z_{i}^{m} \\
    x_{t-1} &= \frac{1}{\sqrt{1-\beta_{i}}}(x_{i} + \beta_{i}s_{\theta^{\star}}(x_{i}, i)) + \sqrt{\beta_{i}}z_{i}
\end{align}
$$

$\text{결정론적 항} + \text{무작위 항}$ 의 형태로, 위의 Diffusion Process SDE 식이 뭔 말인지 알 것이라고 생각한다... 그러니까 $f(x,t)$ 는 특정 방향을 의미하고, $g(t)$ 는 무작위 노이즈의 형태로, 이 샘플링 과정은 특정 방향을 향해 무작위로 지그재그 해가며 이동하는 형태를 떠올릴 수 있다. 

#### 더 일반화

일단 위 식에서 $g(t): \mathbb{R} \to \mathbb{R}$ 는 스칼라를 취하는 함수로 나타냈는데, 일단 더 일반화하면 $f(x,t)$ 와 마찬가지로 $x(t)$ 와 종속인 $G(x,t): \mathbb{R}^{d} \to \mathbb{R}^{d \times d}$ 형태이다. 즉, 다음 형태가 일반적인 Diffusion Process의 SDE 식이다.

$$
dx = f(x,t)dt + G(x,t)dw
$$


### Generating Samples by Reversing The SDE

앞서 Diffusion Process를 SDE로 정의했으니, 정반대 과정인 Reverse Process $x(T) \to x(0)$ 도 연속형 $t$ 에 대해서 정의해야한다.  
다행히도 Anderson 형님의 Reverse-time diffusion equation models 논문에서, Diffusion Process의 반대 과정도 Diffusion Process이며, 기존 Diffusion Process와 마찬가지로 반대로 거슬러가는 $t$ 에 대해 다음과 같이 SDE로 나타낼 수 있다는 것을 증명하셨다.

$$
dx = [f(x,t) - g(t)^{2}\nabla_{x}\log p_{t}(x)]dt + g(t)d\bar{w}
$$

여기서 $\bar{w}$ 는 마찬가지로 Standard Wiener process($T \to 0$ 에서의 표준 브라운 운동)이고, $dt$ 는 아~주 작은 음의 시간 변화량이다.  

기존 Diffusion Process에서 $f(x,t)$ 와 $g(t)$ 는 이미 정의됐고.. 결국 **$\nabla\_{x}\log p\_{t}(x)$ 만 알고 있으면, 위 식의 형태로 Reverse Diffusion Process 를 SDE로 나타낼 수 있게 된다.**

#### 일반화 버전

위 식은 스칼라 값을 취하는 $g(t)$의 경우고, $G(x,t): \mathbb{R}^{d} \to \mathbb{R}^{d \times d}$ 의 버전에선, 다음과 같이 정의된다. 

$$
\begin{align}
    dx &= \set{f(x,t) - \nabla \cdot [G(x,t)G(x,t)^{T}] - G(x,t)G(x,t)^{T}\nabla_{x} \log p_{t}(x)}dt + G(x,t)d\bar{w} \\
    &\cdots \quad G(.) = (g^{1}(.), ... , g^{d}(.))^{T} \to \nabla G(.) = (\nabla g^{1}(.), ... ,\nabla g^{d}(.))^{T}
\end{align}

$$

### Estimating Scores For The SDE

![Refresh time](https://i.imgur.com/FfgzwjU.jpeg)
_글만 있으면 딱딱해서 넣어봤다._

그럼 Diffusion Process, Reverse Diffusion Proecess 모두 SDE의 형태로 어떻게 나타내는지 알아보았고, 그럼 샘플링을 위해서 Reverse Diffusion Process 에 사용할 $\nabla\_{x}\log p\_{t}(x)$ 을 추정할 모델 $s_{\theta}(x,t)$ 을 학습해야 한다.  

일단 그러기 위해서 앞서 살펴본 SMLD 와 DDPM 의 각 최적의 파라미터 $\theta^{\star}$ 식을 일반화하여 다음과 같이 나타내자. 

$$
\theta^{\star} = \underset{\theta}{\arg\min}\, \mathbb{E}_{t}\left[\lambda(t)\mathbb{E}_{x(0)}\mathbb{E}_{x(t)|x(0)}\left[ \|s_{\theta}(x,t) - \nabla_{x(t)} \log p_{0t}(x(t)|x(0)) \|_{2}^{2} \right]\right]
$$

이때 $\lambda: [0,T] \to \mathbb{R}^{+}$ 인 가중치 함수이고, $t \sim \mathcal{U}(0,T)$ 로 정의된다. 

위 식에 대한 최적해 $\theta^{\star}$ 에 대해서, 충분한 데이터와 시간만 있으면,  모든 $t$ 와 $x$ 에 대해 모델 $s_{\theta^{\star}}(x,t) = \nabla\_{x}\log p\_{t}(x)$ 를 만족하며, SMLD와 DDPM 에서 살펴봤듯이, $\mathbb{E}[\|\|\nabla\_{x(t)} \log p\_{0t}(x(t)\|x(0)) \|\|_{2}^{2}] \propto \lambda(t)^{-1}$ 이 되도록 $\lambda(t)$ 를 정의하여 연속적인 $t$ 에 따라 일관된 비중으로 모델이 학습하도록 한다. 

아무튼 위 식을 통해 모델을 학습하기 위해선, 전이 커널 $p\_{0t}(x(t)\|x(0))$ 의 값을 알아야 한다.  
Diffusion Process 에서의 $f(x,t)$ 가 아핀(Affine) 함수인 경우, 모든 전이 커널은 가우시안 분포가 된다. 따라서 DDPM의 논문에서 봤던 것처럼, $x(t)$ 를 샘플링하는데 Closed-Form 으로 할 수 있다는 것이다.  

$f(x,t)$ 가 아핀 함수가 아닌, 일반적인 경우엔 뭐.. $p\_{0t}(x(t)\|x(0))$ 를 Closed-form으로 계산하기 어렵다. 따라서 앞서 Denoising Score Matching 형태의 목적함수를 통해 모델을 훈련하기 좀 거시기해진다.  
이런 경우에  Kolmogorov's forward equation 이란걸 사용하거나, 논문에선 그냥 앞서 본 Reverse Diffusion Process 대로 not Closed-form으로 그냥 값을 샘플링하고, 목적함수를 $p\_{0t}(x(t)\|x(0))$의 계산이 필요없는, Sliced Score Matching과 같은 다른 형태로 바꿔서 학습을 진행한다고 한다. 

## VE, VP, sub-VP SDEs

앞선 DIffusion Process SDE가 SMLD, DDPM에서 어떻게 정의되는지 알아보자..

### VE SDE

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

일반적인 Diffusion Process SDE 식에서 $f(x,t) = 0$, $g(t) = \sqrt{\frac{d[\sigma^{2}(t)]}{dt}}$ 로 정의하면 SMLD를 나타내는 것임을 알 수 있다.

위 식을 보면, $t \to \infty$ 일 때 $\sigma(t)$ 의 값도 덩달아 커지므로, 이를 **Variance Exploding (VE) SDE** 라고 한다. 

### VP SDE

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
    x_{i} &\approx \left(1 - \frac{\beta_{i}}{2}\right)x_{i-1} + \sqrt{\beta_{i}}z_{i-1} \quad \left(\sqrt{1 - x} \approx 1 - \frac{x}{2}\right) \\
    x_{i} - x_{i-1} &\approx  \frac{\beta_{i}}{2}x_{i-1} + \sqrt{\beta_{i}}z_{i-1}\\
    \Delta x &\approx - \frac{\beta(t)\Delta t}{2}x(t) + \sqrt{\beta(t)\Delta t}z(t) \quad (\beta_{i} = \beta(t)\Delta t)\\
    \Delta x &= - \frac{1}{2}\beta(t)x(t)\Delta t + \sqrt{\beta(t)} \sqrt{\Delta t}z(t) \quad (dw \approx \sqrt{\Delta t}z(t)) \\
    \therefore dx &= - \frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)} dw 
\end{align}
$$

마찬가지로 $f(x,t) = - \frac{1}{2}\beta(t)x$, $g(t) = \sqrt{\beta(t)}$ 로 정의하여 DDPM의 Diffusion Process가 정의된다.

이는 분산이 $t$ 가 증가함에 따라 폭발하지 않고 제한되는데, 때문에 이를 **Variance Preserving (VP) SDE** 라고 한다. 


왜 분산이 폭발하지 않는지 알아보자.  
VP SDE가 아핀 함수의 드리프트 계수 $f(.)$ 와 DIffusion 계수 $g(.)$ 를 가지기 때문에, 다음과 같이 분산 변화량에 대한 미분방정식 ODE를 나타낼 수 있다. 

$$
\frac{\Sigma_{\text{VP}}(t)}{dt} = \beta(t)(\mathbf{I} - \Sigma_{\text{VP}}(t))
$$

이때 $\Sigma_{\text{VP}}(t) = \text{Cov}(x(t)) \quad (t \in [0, 1])$ 으로, $\set{x(t)}_{t=0}^{1}$ 의 공분산, 즉, 흩어짐 정도를 나타낸다.. 위 식을 풀어보면, 다음과 같다. 자세한 풀이과정은 [여기를 확인하자..](#vp-공분산-부분-계산-과정)

$$
\Sigma_{\text{VP}}(t) = \mathbf{I} + \exp\left(\int_{0}^{t} - \beta(s)\,ds \right)(\Sigma_{\text{VP}}(0) - \mathbf{I})
$$

어....일단 하나씩 뜯어보자. 
- $\mathbf{I}$: 그냥 단위행렬
- $\exp\left(\int_{0}^{t} - \beta(s)\,ds \right)$: 이거는... 보면.. 일단 $\beta(s)$ 가 노이즈의 양을 의미하는 스케줄러였는데, 중요한건 모두 양수였다는 점이다. 그러니까 즉, 이 항은 $(0,1]$ 범위의 값으로, $t$ 가 증가할 수록 값이 $0$ 으로 작아지는 형태가 될 것이다. 

그러니까... 위 식을 다음과 같이 나타내자.

$$
\Sigma_{\text{VP}}(t) - \mathbf{I} = \exp\left(\int_{0}^{t} - \beta(s)\,ds \right)(\Sigma_{\text{VP}}(0) - \mathbf{I})
$$

여기서 $0 < \exp\left(\int_{0}^{t} - \beta(s)\,ds \right) \leq 1$ 이었으니, 즉!!!! $t$ 에서의 분산이 $\mathbf{I}$ 와 멀어진 정도는 맨처음 분산이 $\mathbf{I}$ 와 멀어진 정도보다 크지 않다는 것이다.  

### sub-VP SDE

논문에서는 이러한 VP SDE 를 조금 변형하여, 마찬가지로 분산의 크기를 제한한 새로운 형태의 SDE로, 다음과 같이 **sub-VP SDE** 를 소개한다. 

$$
dx = - \frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)(1 - \exp(-2 \int_{0}^{t} \beta(s)ds))}dw
$$

마찬가지로 공분산을 계산해보면 다음과 같다. 

$$
\Sigma_{\text{sub-VP}}(t) = \mathbf{I} + \exp\left(-2 \int_{0}^{t} \beta(s)\,ds\right)\mathbf{I} + \exp\left(-\int_{0}^{t} \beta(s)ds\right)(\Sigma_{\text{sub-VP}}(0) - 2\mathbf{I})
$$

아무튼 살펴보면... 같은 $\beta(s)$ 에 대해서, $\Sigma_{\text{sub-VP}}(t) \leq \Sigma_{\text{VP}}(t) \quad (t \geq 0)$ 을 만족하고, 초기 공분산의 값은 같다. 

아무튼 소개한 VE, VP, sub-VP SDE 모두 드리프트 계수는 $f(x,t)$ 가 아핀 함수로 정의되는데, 따라서 모든 전이 커널 $p\_{0t}(x(t)\|x(0))$ 은 모두 가우시안 분포로 정의되어, Closed-forms 으로 계산할 수 있다. 

$$
p_{0t}(x(t)|x(0)) = \begin{cases}
    \mathcal{N}(x(t); x(0), [\sigma^{2}(t) - \sigma^{2}(0)]\mathbf{I}) &\quad (\text{VE}) \\
    \mathcal{N}(x(t); x(0)\exp\left(-2 \int_{0}^{t} \beta(s)\,ds\right), \mathbf{I} - \mathbf{I}\exp\left(-\int_{0}^{t} \beta(s)ds\right)) &\quad (\text{VP}) \\
    \mathcal{N}(x(t); x(0)\exp\left(-2 \int_{0}^{t} \beta(s)\,ds\right), \left[1-\exp\left(-\int_{0}^{t} \beta(s)ds\right)\right]^{2}\mathbf{I}) &\quad (\text{sub-VP}) 
\end{cases}
$$

짜잔... 이제 우리는 바로 모델 훈련하러 가면 된다.

## Sampling

![살려줘](https://i.imgur.com/izzuaSg.gif){: width="400"}
_무량공처 아직 안끝났다._

......이제 샘플링 알아봐야제..

### General-purpose Numerical SDE Solvers

앞서서 뭐 연속형으로 뭘 나타내든 뭐든 간에 컴퓨터에서 계산하기 위해서는 일단 다시 이산형으로 쪼개야한다...  

이렇게 스텝들로 쪼개서 SDE의 근사치를 구하는 걸 **Numerical Solver** 라고 한다.  

위에서 본 DDPM 에서의 샘플링 과정 식 그거... 그거를 Ancestral Sampling 이라고 하는데, 아무튼 그거도 Numerical Solver 의 한 종류이다. 
이거 말고도 뭐.. Euler-Maruyama, Stochastic Runge-Kutta 등등이 있다고 한다. 

뭐가 많아도 드럽게 많다. 아무튼 뭐가 많은데.. 결국 중요한 점은 모든 SDE에 통용되는 것이 아니라, 각 SDE 식 별로 달라서 형태에 맞게 유도해야 한다는 것이다..

그래서 논문에서는 간단하게, 원래 방향의 Diffusion Process 에서 쪼갠 방식 그대로 Reverse Diffusion Process 에서도 동일하게 쪼개는 방식을 소개하는데, 이 방식을 **Reverse Diffusion Samplers** 라고 한다. 

음... 이를 설명하자면.. 일단 먼저 Diffusion SDE가 다음과 같이 주어진다 하자.

$$
dx = f(x,t)dt G(t)dw
$$

그리고 이 식이 다음과 같이 이산형으로 쪼개진다고 가정하자. 

$$
x_{i+1} = x_{i} + f_{i}(x_{i}) + G_{i}z_{i} \quad i = 0, 1, 2, ..., N-1
$$

그르면? Reverse Diffusion SDE는 다음과 같다.

$$
dx = [f(x,t) - G(t)G(t)^{T}\nabla_{x}\log p_{t}(x)]dt + G(t)d\bar{w}
$$

그럼 이산형으로 쪼개면 다음과 같다.

$$
x_{i} = x_{i+1} - f_{i+1}(x_{i+1}) + G_{i+1}G_{i+1}^{T}s_{\theta^{\star}}(x_{i+1}, i+1) + G_{i+1}z_{i+1} \quad i = 0, 1, 2, ..., N-1
$$

아무튼 이를 우리가 살펴본 VE, VP SDEs 에 적용하면 각각의 Reverse VP, VE SDEs 에 대해 컴퓨터로 계산할 수 있는 것이다.  
Ancestral Sampling 그거... 그거도 이를 통해 표현될 수 있는데.. DDPM 샘플링 과정 그거도 결국엔 VP SDE를 푸는 여러 방법 중 하나였다는 것이다...

### Predictor-Corrector (PC) Samplers
 
요거는 앞서 설명한 Numerical Solvers 들을 통해 다음 위치를 계산하고, 이 계산된 위치에서, Score-based MCMC 를 통해 교정하는 과정을 반복하는 과정으로 이 교정하는 과정은 우리가 SMLD에서의 특정 노이즈 레벨에서의 Langevin Dynamics 과정으로 생각할 수 있다. 이를 **Predictor-Corrector (PC) Sampler** 라고 한다.  

앞서 살펴본 SMLD는 Corrector (Langevin Dynamics)만 사용한 경우이고, DDPM은 Predictor (Ancestral Sampling)만 사용한 경우의 PC Sampler 의 한 경우로 볼 수 있다.

![PC for VE,VP](https://i.imgur.com/H0sSvwM.png)

위 사진은 VE, VP SDEs 에 PC Samplers 를 적용한 알고리즘을 나타낸다....

### Probability ODE

지금까지 알아본 SDE 는... 결국 $dw$ 항의 랜덤 샘플링을 통해 이동 궤적이 음... 무작위의 지그재그? 형태에 가까웠었다. 근데!!! 논문의 저자가 보니까 무작위 샘플링이 없는, 즉 $dw$ 항이 없는 Ordinary Differential Equations, ODEs 가 사실 같은 확률 분포 궤적을 그린다는 사실을 알아냈다. 

일반적인 Diffusion SDE 식을 다시 상기해보자.

$$
dx = f(x,t)dt + G(x,t)dw
$$

이제 여기서 $t$ 에 따른 확률 밀도 $p_{t}(x(t))$ 의 변화량은 Kolmogorov's Forward Equation 이란걸로 나타낼 수 있는데, 다음과 같다. 

$$
\frac{d}{dt}p(x,t) = - \frac{d}{dx}[\mu(x,t)p(x,t)] + \frac{1}{2} \frac{d^{2}}{dx^{2}} [\sigma^{2}(x,t)p(x,t)]
$$

이 형태대로, 우리의 Diffusion SDE 식을 적용해보자.

$$
\begin{align}
    \frac{dp_{t}(x)}{dt} &= - \sum_{i=1}^{d} \frac{d}{dx_{i}}[f_{i}(x,t)p_{t}(x)] + \underbrace{\frac{1}{2} \sum_{i=1}^{d}\sum_{j=1}^{d} \frac{d^{2}}{dx_{i}dx_{j}} \left[\sum_{k=1}^{d} G_{ik}(x,t)G_{jk}(x,t) p_{t}(x)\right]}_{(1)} \\
    \\
    (1) &= \frac{1}{2} \sum_{i=1}^{d} \frac{d}{dx_{i}} \left[\underbrace{\sum_{j=1}^{d} \frac{d}{dx_{j}} \left[\sum_{k=1}^{d} G_{ik}(x,t)G_{jk}(x,t) p_{t}(x)\right]}_{(2)}\right] \\ 
    \\
    (2) &= \sum_{j=1}^{d} \frac{d}{dx_{j}} \left[\sum_{k=1}^{d} G_{ik}(x,t)G_{jk}(x,t)\right]p_{t}(x) + \sum_{j=1}^{d} \sum_{k=1}^{d}G_{ik}(x,t)G_{jk}(x,t)p_{t}(x) \frac{d}{dx_{j}}\log p_{t}(x)\\
    &= p_{t}(x) \nabla [G(x,t)G(x,t)^{T}] + p_{t}(x)G(x,t)G(x,t)^{T}\nabla_{x} \log p_{t}(x) \\
    \\
    \frac{dp_{t}(x)}{dt} &= - \sum_{i=1}^{d} \frac{d}{dx_{i}}[f_{i}(x,t)p_{t}(x)] + \frac{1}{2} \sum_{i=1}^{d} \left[ p_{t}(x) \nabla [G(x,t)G(x,t)^{T}] + p_{t}(x)G(x,t)G(x,t)^{T}\nabla_{x} \log p_{t}(x)\right] \\
    &=  - \sum_{i=1}^{d} \frac{d}{dx_{i}}\left[f_{i}(x,t)p_{t}(x) - \frac{1}{2}\left[  \nabla \cdot [G(x,t)G(x,t)^{T}] + G(x,t)G(x,t)^{T}\nabla_{x} \log p_{t}(x)\right]p_{t}(x)  \right] \\\\
    \frac{dp_{t}(x)}{dt} &= - \sum_{i=1}^{d} \frac{d}{dx_{i}}[\tilde{f}_{i}(x,t)p_{t}(x)]\\ \\
    \therefore \tilde{f}(x,t) &= f(x,t) - \frac{1}{2} \nabla \cdot [G(x,t)G(x,t)^{T}] - \frac{1}{2} G(x,t)G(x,t)^{T}\nabla_{x} \log p_{t}(x) \\
    \therefore dx &= \tilde{f}(x,t)d_{t}
\end{align}
$$

~~논문 저자는 뭐하는 사람일까~~

...아무튼 식을 최종 식을 보면... 맨처음 식은.. 1차미분의 드리프트 부분과 2차미분의 Diffusion 부분으로 나뉘어져 있었다. 근데 마지막 최종 식을 보니? $\tilde{f}$ 의 1차미분으로만 나타나졌다..! 즉, 기존 Diffusion 항 부분, 확률 과정을 필요로 하는 항이 사라진 것이다. 즉, $\tilde{G}(x,t) = 0$ 인 다음 식의 Kolmogorov's Forward Equation과 같아진 것이다..

$$
dx = \tilde{f}(x,t)dt + \tilde{G}(x,t)dw = \tilde{f}(x,t)dt
$$

즉...! 더이상 SDE가 아닌 ODE가 된 것이다...!!!!!!

![참 잘했어요](https://i.imgur.com/vQFG04q.gif)

다시 말하자면, 우리가 지금껏 했던 샘플링 과정을, 지그재고 형태로 가는 SDE나, 차분하게 똑바로 가는 ODE나, 이들이 이루는 전체적인 확률 밀도는 같다라는 것이다. 

그래서 정리하자면, 

$$
\begin{align}
    dx &= f(x,t)dt + g(t)dw \\
    &\to \\
    dx &= \underbrace{\left[f(x,t) - \frac{1}{2}g(t)^{2}s_{\theta}(x,t)\right]}_{\tilde{f}_{\theta}(x,t)}dt
\end{align}
$$

$$
\begin{align}
    dx &= f(x,t)dt + G(x,t)dw \\
    &\to \\
    dx &= \underbrace{\left[f(x,t) - \frac{1}{2} \nabla \cdot [G(x,t)G(x,t)^{T}] - \frac{1}{2} G(x,t)G(x,t)^{T}s_{\theta}(x,t)\right]}_{\tilde{f}_{\theta}(x,t)}dt
\end{align}
$$

식을 보면 확률적 샘플링 과정이 아에 사라지고, 결정론적으로 샘플링을 할 수 있게 되는데, 아무튼 이를 **Probability Flow ODE** 라고 한다. 

---

이 개념을 통해서, DDPM 에서의 ELBO를 통한 기존 우도함수의 근사치를 손실함수로 쓰는게 아닌, 정확한 손실함수를 계산할 수 있게 되는 것이다.  

$$
\log p_{0}(x(0)) = \log p_{T}(x(T)) + \int_{0}^{T} \nabla \cdot \tilde{f}_{\theta}(x(t),t) \, dt
$$

여기서 $x(t)$ 는 방금 우리가 실컷 본 Probability Flow ODE 을 통해 얻을 수 있다. 

여기서 $\nabla \cdot \tilde{f}(x(t),t)$ 을 그대로 계산하는건 조금 부담이 크기 때문에, SKilling-Hutchinson trace estimator 라는 걸로 다음과 같이 계산한다. 

$$
\nabla \cdot \tilde{f}_{\theta}(x(t),t) = \mathbb{E}_{p(\epsilon)}[\epsilon^{T}\nabla\tilde{f}_{\theta}(x(t),t)\epsilon]
$$

이때 $\nabla\tilde{f}\_{\theta}$ 는 $\tilde{f}\_{\theta}$ 의 Jacobian 이고, $\mathbb{E}\_{p(\epsilon)}[\epsilon] = 0$, $\text{Cov}\_{p(\epsilon)}(\epsilon)=\mathbf{I}$ 를 만족한다. 

그래서 Sliced Score Matching 과 비슷하게, $\epsilon^{T}\nabla\tilde{f}\_{\theta}(x(t),t)$ 를 Reverse-mode Automatic Differentiation 을 통해 계산하여 얻은 $\epsilon^{T}\nabla\tilde{f}\_{\theta}(x(t),t)\epsilon$ 을 평균냄으로써, 전체 로그 우도함수를 정확화게 계산할 수 있다.

---

그러면? 샘플링 과정도 ODE 로 나타낼 수 있는데, 다시 다음 Diffusion SDE를 가정해보자...

$$
\begin{align}
    dx &= f(x,t)dt + G(t)dw \\
    &\to \\
    x_{i+1} &= x_{i} + f_{i}(x_{i}) + G_{i}z_{i} \quad i = 0, 1, ..., N-1
\end{align}
$$

이 방식을 참고해서.. Probability Flow ODE 를 Reverse Process에 대해 이 방식대로 이산형으로 나타내면 다음과 같다.

$$
\begin{align}
    dx &= \left[f(x,t) - \frac{1}{2}G(t)G(t)^{T}\nabla_{x} \log p_{t}(x)\right]dt \\
    &\to \\
    x_{i} &= x_{i+1} - f_{i+1}(x_{i+1}) + \frac{1}{2}G_{i+1}G_{i+1}^{T}s_{\theta^{\star}}(x_{i+1}, i+1) \quad i = 0, 1, ..., N-1
\end{align}
$$

이렇게 샘플링 과정에도 확률 과정이 사라져서... 만약 입력 데이터 $x_{0} \sim p_{\text{data}}(x)$ 를 $p_{T}(x)$ 로 인코딩하고, 다시 위와 같이 샘플링 과정을 거치면 같은 데이터 $x_{0}$ 이 나올 것이다.. Diffusion, Reverse-diffusion Processes 모두 확률 과정이 사라졌으니까...  



## 찌라시시

![살려줘](https://i.imgur.com/izzuaSg.gif){: width="400"}

따로 실험해보는건... 일단 먼 훗날로 미루겠다.  
지금 글이 개념개념개념개념개념개념 0.2초 무량공처인데, 일단 계속해서 조금이라도 가독성 좋아지게끔 고쳐나가겠다. (나름대로 노력해서)  
...수학 잘하고 싶다.  


## 추가 설명 

### VP 공분산 부분 계산 과정

$$
\begin{align}
    \frac{d\Sigma(t)}{dt} &= \beta(t)(\mathbf{I} - \Sigma(t)) \\
    \frac{d (\mathbf{I} - A(t))}{dt} &=  \beta(t)A(t) \quad (A(t) = \mathbf{I} - \Sigma(t)) \\
    \frac{d\mathbf{I}}{dt} - \frac{dA(t)}{dt} &= \beta(t)A(t) \\
    - \frac{1}{dt}dA(t) &= \beta(t)A(t) \\
    \int_{A(0)}^{A(t)} \frac{1}{A}\,dA &= -\int_{0}^{t} \beta(s)\,ds \\
    [\ln A]_{A(0)}^{A(t)} &= \int_{0}^{t} -\beta(s)\,ds \\
    \ln A(t) - \ln A(0) &= \int_{0}^{t} -\beta(s)\,ds \\
    \exp\left(\ln \frac{A(t)}{A(0)}\right) &= \exp\left(\int_{0}^{t} -\beta(s)\,ds\right) \\
    A(t) &= \exp\left(\int_{0}^{t} -\beta(s)\,ds\right)A(0) \\
    \mathbf{I} - \Sigma(t) &= \exp\left(\int_{0}^{t} -\beta(s)\,ds\right)(\mathbf{I} - \Sigma(t)) \quad (A(t) = \mathbf{I} - \Sigma(t))\\
    \therefore \Sigma(t) &= \mathbf{I} + \exp\left(\int_{0}^{t} -\beta(s)\,ds\right)(\Sigma(t) - \mathbf{I})
\end{align}
$$

