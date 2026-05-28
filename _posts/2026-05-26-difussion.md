---
title: "Denoising Diffusion Probabilistic Models"
date: 2026-05-27 19:27:00 +0900
categories: [DL]
tags: [dl, ddpm] 
math: true
mermaid: true
---

# Denoising Diffusion Probabilistic Models

> [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

## Introduction

좁은 엘레베이터 안, 누군가 방귀를 뀌었다. 방귀 분자는 공기 중으로 무질서하게 퍼지기 시작하고, 이내 엘레베이터 내부 전체를 가득 채운다. ~~똥! ㅋㅋ~~

아무튼 물리학에서는 이러한 현상을 확산(Diffusion)이라고 한다. 질서 정연하게 뭉쳐있던 분자들이 시간이 흐름에 따라 자연스럽게 무질서한 상태 (엔트로피가 증가하는 방향으로) 로 퍼져나가는 현상을 말한다. 결국 시간이 흐르면, 엘레베이터 안은 방귀 분자가 완전히 포화되어, 누가 뀌었는지 알지도 못하는 완전한 무작위 상태가 된다.  

그렇다면, 이 분자가 퍼지는 과정을 완벽하게 역추적한다면? 도대체 어떤 경우없는 사람이 방귀를 뀌었는지 알아낼 수 있는 초기 상태를 확인할 수 있다.  

[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) 여기서 이러한 개념을 생성형 모델에 처음 적용했는데, 지금 살펴볼 **DDPM(Denoising Diffusion Probabilistic Models)** 은 여기서 더 발전된 것으로 볼 수 있다.  



![forward process exp](https://i.imgur.com/sM6GzLs.gif)

분자들이 공기 중으로 퍼져 결국엔 완전히 포화되는 것처럼, 기존 이미지에 소량의 노이즈들이 더해져가며 결국 완전히 노이즈해지는 과정을 **Forward Process, $q$**라고 하고, 우리가 학습할 목표인 이 과정을 역으로 뒤집어 완전한 노이즈로부터 초기 상태로 되돌리는 과정을 **Reverse Process (Diffusion Process), $p_{\theta}$** 라고 한다.  

![DGM_both_process](https://i.imgur.com/6HtTRJd.png)

위 사진과 같이 전체 Diffusion의 구조는 Forward Process, Reverse Process로 구성되는데, 두 Process는 모두 Markov chain으로 정의되며, 각 연쇄 단계에서 각각 소량의 노이즈를 추가/제거가는 과정이다. 

## Background

### Forward Process

노이즈를 더해가는 과정인 **Forward Process, $q$** 는 다음과 같이 정의된다.  

$$
\begin{align}
    q(x_{1:T}|x_{0}) &= \prod_{t=1}^{T} q(x_{t}\|x_{t-1}) \\
    q(x_{t}|x_{t-1}) &= \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}\mathbf{I})
\end{align}
$$

여기서 $\beta_{t} \in (0, 1)$ 는 노이즈 스케줄이며, 각 단계에서의 상태 변환이 가우시안 분포로 이루어지는 것을 볼 수 있다. 

여기서 $x_{t} = \sqrt{1-\beta_{t}}x_{t-1} + \sqrt{\beta_{t}}\epsilon \quad ( \epsilon \sim \mathcal{N}(0, \mathbf{I}))$ 으로 나타낼 수 있는데, $t$ 시점에서의 $x_{t}$ 는 이전 상태 $x_{t-1}$ 기 $\sqrt{1 - \beta_{t}} \quad (1 - \beta_{t} < 1)$ 만큼 감소된 후, $\sqrt{\beta_{t}}\epsilon \quad ( \epsilon \sim \mathcal{N}(0, \mathbf{I}))$ 만큼의 노이즈가 추가된 상태라고 볼 수 있다. 즉 앞선 엘레베이터 예시로 다시 설명하자면, 신선한 공기(이전 상태)가 조금씩 밖으로 빠져나가고 방귀 냄새(노이즈)가 그 빈자리를 채워나가는 것이다.  

여기서 모든 $t$에 대해 $x_{t}$ 의 분산은 $1$ 로 유지되는데, 이를 통해서 고정된 분산으로 학습이 용이하며, 최종적으로 $T$ 단계에서 완전히 노이즈해지는 $x_{T}$ 가 $\mathcal{N}(0, \mathbf{I})$ 와 같이 표준 정규 분포가 될 수 있게끔 한다. (왜 분산이 1로 고정되는지는 [여기](#forward-process-의-분산이-1로-고정되는-이유))

이러한 $\beta_{t}$ 값은 하이퍼파라미터로 상수로 고정되거나, 학습시킬 수 있는데, 이 논문에선 $10^{-4} \to 2 \times 10^{-2}$ 로 선형적으로 증가하게끔 상수로 고정하였다.  

Forward Process의 가장 큰 특징은 각 과정이 모두 가우시안으로 정의되기 때문에, 특정 스텝의 $x_{t}$ 를 **단힌 형태 (Closed Form)** 로 계산할 수 있다는 것이다. 다시 말해서, $x_{t}$ 를 계산하기 위해서 $x_{0}$ 부터 시작해서 $x_{1} \to x_{2} \to \cdots \to x_{t}$ 이런 식으로 순차적으로 계산해 나가는 것이 아니라, $x_{0}$ 이 주어지고, $\beta_{t}$ 의 값만 알면, $x_{0} \to x_{t}$ 로 바로 계산할 수 있다는 것이다! 

$$
\begin{align}
    q(x_{t}|x_{0}) &= \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}}x_{0}, (1-\bar{\alpha})\mathbf{I}) \\
    \alpha_{t} &= 1 - \beta_{t} \\
    \bar{\alpha}_{t} &= \prod_{s=1}^{t} \alpha_{s}
\end{align}
$$

즉, 다음과 같다.  

$$
x_{t} = \sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1-\bar{\alpha}_{t}} \epsilon \quad ( \epsilon \sim \mathcal{N}(0, \mathbf{I}))
$$

왜 이따구가 되는지는 [여기를 확인하자](#closed-form-of-forward-process)

$x_{t}$ 를 바로 계산할 수 있다는 것은 꽤나 중요한데, 뒤에서 설명하겠지만 DDPM의 학습은 매 학습스텝마다 $t \sim \mathcal{U}(1, T)$ 값을 무작위로 가져와 $x_{t}$ 를 모델의 입력으로 사용하는데, 매 학습스텝에서 일일히 $x_{1} \to x_{2} \to \cdots \to x_{t}$ 이딴식으로 계산이나 하고 있으면 학습 시간이 드럽게 오래걸릴 것이다. (거기다 가뜩이나 논문에선 $T$ 의 값을 $T=1000$ 과 같이 크게 설정한다..) 

### Reverse Process

노이즈를 제거해가는 과정인 **Reverse Process, $p_{\theta}$** 는 다음과 같이 정의된다. 

$$
\begin{align}
    p_{\theta}(x_{0:T}) &= p(x_{T})\prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_{t}) \\
    p_{\theta}(x_{t-1}|x_{t}) &= \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t},t), \Sigma_{\theta}(x_{t},t))
\end{align}
$$

Forward Process와 마찬가지로 가우시안으로 정의된다. 앞서서 $\beta_{t}$ 의 값을 논문에선 $10^{-4} \to 2 \times 10^{-2}$ 와 같이 매우 작은 값으로 설정하였는데, $\beta_{t}$ 의 값은 결국 매 단계에서 추가되는 노이즈의 양으로 생각될 수 있다.  이러한 추가되는 노이즈의 양이 매우 적을 때, 그리고 추가적으로 Process의 단계 $T$ 의 수가 매우 커질 때, 그 역과정인 Reverse Process도 충분히 가우시안으로 근사할 수 있다.  

> 그러니까 음.. 예를 들어서 탄산 음료의 병뚜껑을 처음 열었을 때, 탄산 분자가 공기 중으로 퍼져 나가는걸 상상해보자. 탄소 분자가 퍼져 나가면서 $\nabla t$ 후에 이동되는 위치 $x_{t+1}$ 은 현재 위치 $x_{t}$ 를 기준으로 하는 구체 영역 내의 한 점으로 생각될 수 있다. 물론 이를 정확히 예측하는 것은 거의 불가능하지만, 관측하는 시간을 찰나의 시간으로 줄인다면, 즉, $\nabla t \to 0$ 이라면, 다음 위치 $x_{t+1}$ 을 결정하는 $x_{t}$ 를 기준으로 하는 구체의 영역도 마찬가지로 줄어들 것이다. 이렇게 아~주 작게 줄어들면 탄소 분자의 위치 변화는 변동성이 줄어들고 선형성이 조금씩 증가하면서, 가우시안으로 근사할 수 있게 된다. [SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456) 여기서 지금처럼 $t=1,2,...,T$ 와 같은 이산형 $t$ 가 아닌 연속형 $t \in [1,T]$ 에서 이를 다루긴 하는데, 일단 근래에 따로 글로 작성하겠다..

아무튼 위 식에서 $p(x_{T}) =  \mathcal{N}(x_{T}; 0, \mathbf{I})$ 로 정의되고, 결국 우리가 학습한 모델이 Reverse Process 각 단계에서의 평균 $\mu_{\theta}$ 와 분산 $\Sigma_{\theta}$ 이 두 값을 출력해야 함을 알 수 있다. 

## Objective

목적함수는 NLL(Negative log-likelihood)을 최소화하는데, VAE에서와 마찬가지로 ELBO를 통해 이를 근사하여 최적화한다. 

$$
\begin{align}
    \mathbb{E}[-\log p_{\theta}(x_{0})] &\leq \mathbb{E}_{q}\left[-\log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_{0})} \right] = L \\
    L &= \mathbb{E}_{q}\left[-\log p(x_{T}) - \sum_{t \geq 1}\log \frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})} \right] \\
    &= \mathbb{E}_{q}\left[\underbrace{D_{KL}(q(x_{T}|x_{0}) || p(x_{T}))}_{L_{T}} + \sum_{t > 1} \underbrace{D_{KL}(q(x_{t-1} | x_{t}, x_{0}) || p_{\theta}(x_{t-1}|x_{t}))}_{L_{t-1}} \underbrace{-\log p_{\theta}(x_{0}|x_{1})}_{L_{0}} \right]
\end{align} \tag{1}
$$

자세한 과정은 [여기](#eq-1-nll-elbo) 에서 확인하자.  


위 식에서 $L_{t-1}$ 부분을 보면, 결국 우리의 모델이 예측할 $p_{\theta}(x_{t-1} \| x_{t})$ 가 $q(x_{t-1}\|x_{t}, x_{0})$ 와 비슷해져야 함을 알 수 있는데, $q(x_{t-1}\|x_{t}, x_{0})$ 은 가우시안의 형태로, $q(x_{t-1}\|x_{t})$ 에 추가적으로 $x_{0}$ 을 조건으로 둘 때 계산할 수 있다.   

$$
\begin{align}
    q(x_{t-1}|x_{t}, x_{0}) &= \mathcal{N}(x_{t-1}; \tilde{\mu}_{t}(x_{t}, x_{0}), \tilde{\beta}_{t}\mathbf{I}) \\
    \tilde{\mu}_{t}(x_{t}, x_{0}) &= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_{t} \\
    \tilde{\beta}_{t} &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \beta_{t}
\end{align} \tag{2}
$$

왜 이따구로 나오는지 계산 과정은 [이 형님의 블로그](https://joydeep31415.medium.com/the-math-behind-diffusion-models-ddpm-9fabe9c9f1d9)를 참고하자....
[여기](#eq-2-q_posterior)

이제 각 항들을 각각 살펴봐서 손실함수를 어떻게 계산해야 할 지 알아보자.

### $L_{T}$

논문에선 $\beta_{t}$ 를 상수로 정의하기 때문에, $q(x_{T}\|x_{0})$ 또한 학습 동안 상수로 유지된다. $p(x_{T})$ 도 $\mathcal{N}(0, \mathbf{I})$ 의 표준 정규 분포로 고정되므로, 따라서 논문에선 이를 무시한다.  

### $L_{1:T-1}$

학습을 통해서 $p_{\theta}(x_{t-1}\| x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))$ 를 $q(x_{t-1} \| x_{t}, x_{0})$ 와 최대한 가깝게 하도록 해야한다.  

그러기 위해선 $\mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)$ 를 앞서 살펴본 $q(x_{t-1} \| x_{t}, x_{0})$ 의 평균과 분산의 값과 비슷하게끔 해야하는데, 논문에선 $\Sigma_{\theta}(x_{t}, t) = \sigma_{t}^{2} \mathbf{I}$ 의 상수로 정의한다.  

$\sigma\_{t}^{2} = \beta\_{t}$ 또는 $\sigma\_{t}^{2} = \tilde{\beta}\_{t}$ 로 정의하는데, 두 값은 논문에서 비슷한 성능을 가진다고 한다.  

그러면 $p\_{\theta}(\mathbf{x}\_{t-1} \| \mathbf{x}\_{t}) = \mathcal{N}(\mathbf{x}\_{t-1};\mu\_{\theta}(\mathbf{x}\_{t},t), \sigma\_{t}^{2}\mathbf{I})$ 에 대해서, 앞선 $L\_{t-1}$ 을 다음과 같이 나타낼 수 있다.  

$$
L_{t-1} = \mathbb{E}_{q}\left[ \frac{1}{2\sigma_{t}^{2}}||\tilde{\mu}_{t}(\mathbf{x}_{t},\mathbf{x}_{0}) - \mu_{\theta}(\mathbf{x}_{t},t)||^{2} \right] + C \tag{3}
$$

이때 $C$ 는 학습 파라미터 $\theta$ 와 관련 없는 항들을 따로 뺀 것으로, 자세한 계산 과정은 [여기를 확인하자](#eq-3-l-based).

결국 우리가 학습할 모델이 $q(x_{t-1} \| x_{t}, x_{0})$ 의 평균값 $\tilde{\mu}\_{t}(x\_{t}, x\_{0})$ 를 예측하게 되는 문제로 축소된다.  

--- 

여기서 앞서서 $x_{t}(x_{0}, \epsilon) = \sqrt{1-\beta_{t}}x_{t-1} + \sqrt{\beta_{t}}\epsilon \quad ( \epsilon \sim \mathcal{N}(0, \mathbf{I}))$ 와 식 2의 $\tilde{\mu}\_{t}(x\_{t}, x\_{0})$ 을 통해 다음과 같이 나타낼 수 있다.  

$$
\begin{align}
L_{t-1} - C &= \mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \frac{1}{2\sigma_{t}^{2}}|| \tilde{\mu}_{t}\left( \mathbf{x}_{t}(\mathbf{x}_{0},\epsilon), \frac{1}{\sqrt{ \bar{\alpha}_{t} }}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon) - \sqrt{ 1-\bar{\alpha}_{t} }\epsilon) \right) -\mu_{\theta}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon), t) ||^{2} \right] \\
&= \mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \frac{1}{2\sigma_{t}^{2}} || \frac{1}{\sqrt{ \alpha_{t} }}\left( \mathbf{x}_{t}(\mathbf{x}_{0},\epsilon) - \frac{\beta_{t}}{\sqrt{ 1-\bar{\alpha}_{t} }}\epsilon \right) - \mu_{\theta}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon), t) ||^{2} \right] 
\end{align} \tag{4}
$$

자세한 과정은 [여기를 확인하자](#eq-4-reparameterization-with-epsilon)

즉, 모델은 $x_{t}$ 와 $t$ 가 입력으로 주어졌을 때 $\frac{1}{\sqrt{\alpha\_{t}}}\left(\mathbf{x}\_{t} - \frac{\beta\_{t}}{\sqrt{ 1 - \bar{\alpha}\_{t}}}\epsilon \right)$ 을 예측하는 것과 같다. 또한 여기서 샘플링된 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 를 제외하고, 나머지 $\alpha, \beta, \bar{\alpha}, x_{t}$ 값들은 모두 상수의 형태로 주어지므로, 다음과 같이 우리의 모델 $\mu_{\theta}$ 는 전체 평균 $\frac{1}{\sqrt{\alpha\_{t}}}\left(\mathbf{x}\_{t} - \frac{\beta\_{t}}{\sqrt{ 1 - \bar{\alpha}\_{t}}}\epsilon \right)$ 을 예측하는 것이 아닌, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 값만을 예측하게끔 범위를 축소시킬 수 있다.  

$$
\mu_{\theta}(\mathbf{x}_{t},t) = \frac{1}{\sqrt{ \alpha_{t} }} \left( \mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{ 1-\bar{\alpha}_{t} }}\epsilon_{\theta}(\mathbf{x}_{t},t) \right) \tag{5}
$$

즉, 위 식을 식 4에 적용해서, 다시 나타내면 다음과 같다. 

$$
\begin{align}
    &\mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \frac{\beta_{t}^{2}}{2\sigma_{t}^{2}\alpha_{t}(1-\bar{\alpha}_{t})} ||\epsilon - \epsilon_{\theta}(\sqrt{ \bar{\alpha}_{t} }\mathbf{x}_{0} + \sqrt{ 1-\bar{\alpha}_{t} }\epsilon,t) ||^{2} \right] \\
    =& \mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \lambda_{t} \cdot ||\epsilon - \epsilon_{\theta}(\sqrt{ \bar{\alpha}_{t} }\mathbf{x}_{0} + \sqrt{ 1-\bar{\alpha}_{t} }\epsilon,t) ||^{2} \right]
\end{align}
 \tag{6}
$$

역시나 자세한 계산 과정은 [여기를 확인하자](#eq-6-reconstruction-with-epsilon-approximator).

결국 $p_{\theta}$ 가 $q$ 와 비슷해지로고 학습한다는 것은, $x_{t}$ 에서 섞인 노이즈를 예측하는 것과 같음을 알 수 있다. 

### $L_{0}$

이미지 데이터는 $\set{0, 1, ..., 255}$ 의 정수로 구성되는데, 그에 반해 Reverse Process 모델 $p_{\theta}$ 의 출력은 연속적인 확률 흐름을 지닌 가우시안 PDF의 형태이다. 이러한 연속과 이산값의 괴리를 해결하기 위해서, 마지막 $t=1$ 에서 $t=0$ 으로 가는 단계 $L_{0}$ 에서는 특정 픽셀 값이 차지하는 영역의 면적인 적분을 구하는 방식으로 계산한다.  

$$
\begin{align}
p_{\theta}(\mathbf{x}_{0}\|\mathbf{x}_{1}) &= \prod_{i=1}^{D} \int_{\delta_{-}(x_{0}^{i})}^{\delta_{+}(x_{0}^{i})}\mathcal{N}(x;\mu_{\theta}^{i}(\mathbf{x}_{1},1), \sigma_{1}^{2})\,dx \\
\delta_{+}(x) &= \begin{cases}
\infty & \text{if }x=1 \\
x+ \frac{1}{255} & \text{if }x<1
\end{cases} \\
\delta_{-}(x) &= \begin{cases}
-\infty & \text{if }x=-1  \\
x- \frac{1}{255} & \text{if }x>-1
\end{cases}
\end{align} 
$$

이때 $D$는 데이터의 차원, $i$ 는 한 좌표축을 나타낸다.  
이 식은 우리의 모델이 Reverse Process의 마지막 단계에서 $x_{0}$ 을 예측할 때, 각 픽셀들의 특정 값의 주변 영역의 면적을 구하여 이산적인 값을 예측한다. 

### Simplified objective

앞서 주어진 $L$ 의 각 항들에 대해서 살펴봤는데, 사실 논문에선 다음과 같이 노이즈 자체만을 모델이 예측하는게 더 간단하고 성능이 잘 나온다고 한다.  

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_{0}, \epsilon}\left[\left|\left| \epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1 - \bar{\alpha}_{t}}\epsilon, t) \right|\right|^{2}\right] \tag{7}
$$

논문의 저자는 위와 같이 $\lambda_{t} = 1$ 로 설정했는데, 데이터의 소량의 노이즈만 낀 초기 단계 $t$ 에서의 손실 함수의 가중치가 감소되게 되었고, 데이터를 거의 알아보지 못할 정도로 노이즈가 낀 $t$ 단계에서의 노이즈를 제거하는 것을 중심으로 모델이 학습하는 결과를 낳게 되었다. 즉, 다시 말해서 작은 $t$ 에서 데이터를 거의 알아볼 수 있을 때가 아닌, 큰 $t$ 에서 데이터를 거의 알아볼 수 없을 정도로 노이즈가 낀 상태에서 노이즈를 제거하는 것에 초점을 두고 모델을 학습을 진행하게끔 되었다는 것이다. 아무튼 이렇게 했더니 더 좋은 화질의 결과물을 얻었다고 한다.  


## Implementation

최종적인 목적 함수 식 7에 의해, 학습 알고리즘은 다음과 같이 구성된다.  

![Train_algorithm](https://i.imgur.com/ZO9dWoJ.png)

각 학습스텝마다 $t \sim \mathcal{U}(1,T)$, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 를 샘플링하여, Closed-form으로 $x_{t}$ 를 계산한 다음, $x_{t}$ 와 $t$ 를 모델에 입력으로 넣어 출력한 결과를 $\epsilon$ 과의 MSE Loss를 계산하여 학습을 진행한다.  

![Sample_algorithm](https://i.imgur.com/dZZgaOi.png)

샘플링 과정은 위와 같은데, 표준 정규 분포로부터 샘플링된 무작위 $x_{T} \sim \mathcal{N}(0, \mathbf{I})$ 를 시작으로, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 을 샘플링하고 $x_{t}$ 를 Closed-form으로 계산한 뒤, 우리가 학습한 모델 $\epsilon_{\theta}$ 를 이용하여, $\mu\_{\theta}(\mathbf{x}\_{t},t) = \frac{1}{\sqrt{ \alpha\_{t} }} \left( \mathbf{x}\_{t} - \frac{\beta\_{t}}{\sqrt{ 1-\bar{\alpha}\_{t} }}\epsilon\_{\theta}(\mathbf{x}\_{t},t) \right)$ , $\Sigma\_{\theta} = \sigma\_{t}$ 를 통해, $x\_{t-1} = \mu\_{\theta}(\mathbf{x}\_{t},t) + \sigma \epsilon$ 을 계산한다. 이 과정을 $T$ 번 반복하여, 최종적으로 $x_{0}$ 을 얻는다.  


논문에서 모델 $\epsilon_{\theta}$ 의 구조는 UNet을 기본 베이스라인으로, 추가적으로 그룹 정규화(Group Normalization) 와 Self-attention이 특정 Feature map 크기에서 사용하였다. 이 때 Reverse Process의 각 단계 $t$ 에 대해서 총 $T$ 개의 모델이 아닌 파라미터를 공유하는 하나의 모델을 사용하고, 단계 $t$ 를 따로 구분하기 위해 $t$ 를 Transformer에서 사용된 Position Embedding을 수행하여 모델에 입력된다. 아무튼 자세한 구조는 논문을 확인하자...

![Train_result](https://i.imgur.com/qg9CvtI.png)

위 사진은 LSUN bedroom 데이터셋에서 $1.5e^{5}$ 스텝만큼 학습된 모델을 통해 샘플링한 이미지들로, VAE에서 흐릿한 이미지가 아닌 (DC)GAN 처럼 뚜렷한 이미지가 출력되는 것을 볼 수 있다. 

## 찌라시

최종적인 목적함수가 논문에선 Denoising Score Matching 과 같다고 언급한다. 또한 샘플링 과정도 Langevin Dynamics 와 같다고 언급한다. 그리고 


## 추가 설명

### FOrward Process 의 분산이 1로 고정되는 이유


### Closed-form of Forward Process 



### Eq. 1 NLL ELBO

$$
\begin{align}
L &= \mathbb{E}_{q}\left[ -\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} \right] \tag{17}\\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right] \tag{18} \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T}) -\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})} -\sum_{t > 1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right] \tag{19} \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T})-\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})} - \sum_{t > 1} \log p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \cdot \frac{q(\mathbf{x}_{t-1})}{q(\mathbf{x}_{t},\mathbf{x}_{t-1})} \right]  \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T})-\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})} - \sum_{t > 1} \log p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \cdot \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})} \right] \\
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T})-\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})} - \sum_{t > 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})} \cdot \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{0})}{q(\mathbf{x}_{t}|\mathbf{x}_{0})} \right] \tag{20}\\ 
&= \mathbb{E}_{q}\left[ -\log p(\mathbf{x}_{T})-\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{1}|\mathbf{x}_{0})} -\log \frac{p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}{q(\mathbf{x}_{T}|\mathbf{x}_{0})} - \sum_{t > 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})} \right] \\ 
&= \mathbb{E}_{q}\left[ -\log \frac{p(\mathbf{x}_{T})}{q(\mathbf{x}_{T}|\mathbf{x}_{0})} - \log p_{\theta(\mathbf{x}_{0}|\mathbf{x}_{1})} - \sum_{t>1}\log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})} \right] \tag{21}  \\
&= \mathbb{E}_{q}\left[ D_{KL}(q(\mathbf{x}_{T}|\mathbf{x}_{0}) || p(\mathbf{x}_{T})) - \log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1}) + \sum_{t>1}D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0}) || p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})) \right] \tag{22}
\end{align}
$$

### Eq. 2 q_posterior


### Eq. 3 L based $\mu$


### Eq. 4 reparameterization with epsilon


### Eq. 6 reconstruction with epsilon-approximator

~~추가 중~~


