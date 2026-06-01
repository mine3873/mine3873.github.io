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

이를 통해 노이즈 레벨 $i$ 에 상관없이 각 손실함수 $\ell(\theta; \sigma\_{i}), \ell(\theta; i)$ 는 모두 일정한 크기를 가진다. 

### Score-based Generative Modeling With SDEs

| ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 1_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 2_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 3_ |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 4_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 5_ | ![힐링타임](https://i.imgur.com/vQFG04q.gif)_잠시 힐링 좀 하겠다 6_ |




