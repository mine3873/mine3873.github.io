---
title: "Denoising Diffusion Probabilistic Models"
date: 2026-05-27 19:27:00 +0900
categories: [DL]
tags: [dl] 
math: true
mermaid: true
---

# Denoising Diffusion Probabilistic Models

> [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

## Introduction

좁은 엘레베이터 안, 누군가 방귀를 뀌었다. 방귀 분자는 공기 중으로 무질서하게 퍼지기 시작하고, 이내 엘레베이터 내부 전체를 가득 채운다. ~~똥! ㅋㅋ~~

아무튼 물리학에서는 이러한 현상을 확산(Diffusion)이라고 한다. 질서 정연하게 뭉쳐있던 분자들이 시간이 흐름에 따라 자연스럽게 무질서한 상태 (엔트로피가 증가하는 방향으로) 로 퍼져나가는 현상을 말한다. 결국 시간이 흐르면, 엘레베이터 안은 방귀 분자가 완전히 포화되어, 누가 뀌었는지 알지도 못하는 완전한 무작위 상태가 된다.  

그렇다면, 이 분자가 퍼지는 과정을 완벽하게 역추적한다면? 도대체 어떤 경우없는 사람이 방귀를 뀌었는지 알아낼 수 있는 초기 상태를 확인할 수 있다.  

**DDPM(Denoising Diffusion Probabilistic Models)** 은 이러한 점을 생성형 모델에 적용하였다.  

![forward process exp](https://i.imgur.com/sM6GzLs.gif)

분자들이 공기 중으로 퍼져 결국엔 완전히 포화되는 것처럼, 기존 이미지에 소량의 노이즈들이 더해져가며 결국 완전히 노이즈해지는 과정을 **Forward Process, $q$**라고 하고, 우리가 학습할 목표인 이 과정을 역으로 뒤집어 완전한 노이즈로부터 초기 상태로 되돌리는 과정을 **Reverse Process (Diffusion Process), $p_{\theta}$** 라고 한다.  

![DGM_both_process](https://i.imgur.com/6HtTRJd.png)

위 사진과 같이 전체 Diffusion의 구조는 Forward Process, Reverse Process로 구성되는데, 두 Process는 모두 Markov chain으로 정의된다.  

여기서 분자들이 퍼져 나가는 그 움직임을 수학적으로 정의해야 한다. 다음 상태 분자의 위치는 현재 상태 분자 위치를 기준으로 하는 특정 크기의 구체의 영역 내의 한 지점으로 생각할 수 있다. 하지만 그 중 어느 지점이 분자의 다음 위치가 될지 예측하는 것은 거의 불가능하다. 하지만 시간의 변화량 $\nabla t$ 을 충분히 작게하면, 즉, 특정 분자를 기준으로 아~~~주 미세한 찰나의 시간 동안 관찰한다면, 분자의 다음 위치를 나타내는 해당 구체의 크기도 마찬가지로 작아지게 될 것이고, 해당 구체의 크기를 충분히 줄이면 다음 분자의 위치는 가우시안 분포로 근사할 수가 있다! 즉, 분자가 퍼지는 Forward Process를 아주 미세한 가우시안 분포들로 정의할 수 있으면, 그 역과정인 Reverse Process 또한 가우시안으로 정의할 수 있다는 것이다. 자세한건 [여기서 확인하자]()

## Objective

Diffusion 모델은 $p_{\theta}(x_{0}) = \int p_{\theta}(x_{0:T}) \, dx_{1:T}$ 형태의 잠재 변수 모델로, 여기서 $p_{\theta}(x_{0:T})$ 가 **Reverse Process, $p_{\theta}$** 이다.  

$$
\begin{align}
    p_{\theta}(x_{0:T}) &= p(x_{T})\prod_{t=1}^{T} p_{\theta}(x_{t-1}\|x_{t}) \\
    p_{\theta}(x_{t-1}|x_{t}) &= \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t},t), \Sigma_{\theta}(x_{t},t))
\end{align}
$$

여기서 $T$ 시점의 완전한 노이즈 상태는 $p(x_{T}) = \mathcal{N}(x_{t-1}; 0, \mathbf{I})$ 로 정의된다. 

**Forward Process, $q$** 는 다음과 같이 정의된다.  

$$
\begin{align}
    q(x_{1:T}|x_{0}) &= \prod_{t=1}^{T} q(x_{t}\|x_{t-1}) \\
    q(x_{t}|x_{t-1}) &= \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1}, \beta_{t}\mathbf{I})
\end{align}
$$

여기서 $\beta_{t}$ 는 $0$ 에 가까운 값의 Forward Process의 분산으로, Forward Process의 각 단계에서, 이전 상태 $x_{t-1}$ 의 값을 미세하게 감소시키고 미세한 노이즈를 추가하는 것을 볼 수 있다.  

여기서 $x_{t} = \sqrt{1-\beta_{t}}x_{t-1} + \sqrt{\beta_{t}}\epsilon \quad ( \epsilon \sim \mathcal{N}(0, \mathbf{I}))$ 으로 나타낼 수 있는데, $x_{t}$ 의 분산은 $1$ 로 유지된다. 이를 통해 데이터의 스케일을 통제하여 $T$ 단계에서 완전한 노이즈 상태 $\mathcal{N}(0, \mathbf{I})$ 에 맞출 수 있고, 고정된 분산으로 학습도 쉬워진다.  

이러한 $\beta_{t}$ 값은 하이퍼파라미터로 상수로 고정되거나, 학습시킬 수 있는데, 이 논문에선 상수로 고정하였다.  

Forward Process의 가장 큰 특징은 특정 스텝의 $x_{t}$ 를 **단힌 형태 (Closed Form)** 로 계산할 수 있다는 것이다.  

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

목적함수는 NLL(Negative log-likelihood)를 최소화하는데, VAE에서와 마찬가지로 ELBO를 통해 이를 근사하여 최적화한다. 

$$
\begin{align}
    \mathbb{E}[-\log p_{\theta}(x_{0})] &\leq \mathbb{E}_{q}\left[-\log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_{0})} \right] = L \\
    L &= \mathbb{E}_{q}\left[-\log p(x_{T}) - \sum_{t \geq 1}\log \frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})} \right] \\
    &= \mathbb{E}_{q}\left[\underbrace{D_{KL}(q(x_{T}|x_{0}) || p(x_{T}))}_{L_{T}} + \sum_{t > 1} \underbrace{D_{KL}(q(x_{t-1} | x_{t}, x_{0}) || p_{\theta}(x_{t-1}|x_{t}))}_{L_{t-1}} \underbrace{-\log p_{\theta}(x_{0}|x_{1})}_{L_{0}} \right]
\end{align} \tag{1}
$$

자세한 과정은 [여기](#eq-1) 에서 확인하자.  


위 식에서 $q(x_{t-1} \| x_{t}, x_{0})$ 은 가우시안의 형태로, $x_{0}$ 을 조건으로 가질 때 계산할 수 있다.   

$$
\begin{align}
    q(x_{t-1}|x_{t}, x_{0}) &= \mathcal{N}(x_{t-1}; \tilde{\mu}_{t}(x_{t}, x_{0}), \tilde{\beta}_{t}\mathbf{I}) \\
    \tilde{\mu}_{t}(x_{t}, x_{0}) &= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}x_{0} + \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} x_{t} \\
    \tilde{\beta}_{t} &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \beta_{t}
\end{align} \tag{2}
$$

왜 이따구로 나오는지 계산 과정은 [이 형님의 블로그](https://joydeep31415.medium.com/the-math-behind-diffusion-models-ddpm-9fabe9c9f1d9)를 참고하자....

### $L_{T}$

논문에선 $\beta_{t}$ 를 상수로 정의하기 때문에, $q(x_{T}\|x_{0})$ 또한 학습 동안 상수로 유지된다. 따라서 학습 과정에서 무시할 수 있다.  

### $L_{1:T-1}$

학습을 통해서 $p_{\theta}(x_{t-1}\| x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))$ 를 $q(x_{t-1} \| x_{t}, x_{0})$ 와 최대한 가깝게 하도록 해야한다.  

그러기 위해선 $\mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t)$ 를 나타내야 하는데, 논문에선 $\Sigma_{\theta}(x_{t}, t) = \sigma_{t}^{2} \mathbf{I}$ 의 상수로 정의한다.  

$\sigma\_{t}^{2} = \beta\_{t}$ 또는 $\sigma\_{t}^{2} = \tilde{\beta}\_{t}$ 로 정의하는데, 전자는 원본 데이터 $x_{0}$ 이 완전환 무작위 노이즈 $\mathcal{N}(0, \mathbf{I})$ 라고 가정할 때, 즉 $x\_{0}$ 에 대한 정보가 아에 없을 때 최적의 값이고, 후자는 $x\_{0}$ 가 한 점에 고정되어 있을 때, 즉 $x\_{0}$ 에 대해 확실히 알 고 있는 상태에서 최적의 값으로, 각각 실제 이상적인 역과정 분산의 상한선과 하한선이다.  

그러면 $p\_{\theta}(\mathbf{x}\_{t-1} \| \mathbf{x}\_{t}) = \mathcal{N}(\mathbf{x}\_{t-1};\mu\_{\theta}(\mathbf{x}\_{t},t), \sigma\_{t}^{2}\mathbf{I})$ 에 대해서, 앞선 $L\_{t-1}$ 을 다음과 같이 나타낼 수 있다.  

$$
L_{t-1} = \mathbb{E}_{q}\left[ \frac{1}{2\sigma_{t}^{2}}||\tilde{\mu}_{t}(\mathbf{x}_{t},\mathbf{x}_{0}) - \mu_{\theta}(\mathbf{x}_{t},t)||^{2} \right] + C \tag{3}
$$

이때 $C$ 는 학습 파라미터 $\theta$ 와 관련 없는 항으로, 위 식은 결국 모델이 Forward Process의 사후 평균 $\tilde{\mu}_{t}$ 를 예측하도록 학습됨을 의미한다. 

여기서 앞서서 $x_{t}(x_{0}, \epsilon) = \sqrt{1-\beta_{t}}x_{t-1} + \sqrt{\beta_{t}}\epsilon \quad ( \epsilon \sim \mathcal{N}(0, \mathbf{I}))$ 와 식 2를 사용하여 다음과 같이 나타낼 수 있다.  

$$
\begin{align}
L_{t-1} - C &= \mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \frac{1}{2\sigma_{t}^{2}}|| \tilde{\mu}_{t}\left( \mathbf{x}_{t}(\mathbf{x}_{0},\epsilon), \frac{1}{\sqrt{ \bar{\alpha}_{t} }}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon) - \sqrt{ 1-\bar{\alpha}_{t} }\epsilon) \right) -\mu_{\theta}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon), t) ||^{2} \right] \\
&= \mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \frac{1}{2\sigma_{t}^{2}} || \frac{1}{\sqrt{ \alpha_{t} }}\left( \mathbf{x}_{t}(\mathbf{x}_{0},\epsilon) - \frac{\beta_{t}}{\sqrt{ 1-\bar{\alpha}_{t} }}\epsilon \right) - \mu_{\theta}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon), t) ||^{2} \right] 
\end{align} \tag{4}
$$

즉, 모델은 $x_{t}$ 와 $t$ 가 입력으로 주어졌을 때 $\frac{1}{\sqrt{\alpha\_{t}}}\left(\mathbf{x}\_{t} - \frac{\beta\_{t}}{\sqrt{ 1 - \bar{\alpha}\_{t}}}\epsilon \right)$ 을 예측하는 것과 같다. 더 나아가서 $\epsilon$ 을 제외한 나머지 모든 값들이 주어지기 때문에, 다음과 같다.  

$$
\mu_{\theta}(\mathbf{x}_{t},t) = \frac{1}{\sqrt{ \alpha_{t} }} \left( \mathbf{x}_{t} - \frac{\beta_{t}}{\sqrt{ 1-\bar{\alpha}_{t} }}\epsilon_{\theta}(\mathbf{x}_{t},t) \right) \tag{5}
$$

즉, 위 식을 식 4에 적용하면 다음과 같다.  

$$
\mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[ \frac{\beta_{t}^{2}}{2\sigma_{t}^{2}\alpha_{t}(1-\bar{\alpha}_{t})} ||\epsilon - \epsilon_{\theta}(\sqrt{ \bar{\alpha}_{t} }\mathbf{x}_{0} + \sqrt{ 1-\bar{\alpha}_{t} }\epsilon,t) ||^{2} \right] \tag{6}
$$

결국 $p_{\theta}$ 가 $q$ 와 가까워지도록 학습하는 것은, 초기 데이터 $x_{0}$ 에서 섞인 노이즈를 예측하는 문제가 된 것을 알 수 있다.  

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

위 식은 $t > 1$ 일 때, 앞서 살펴본 식 6의 가중치를 제거한 것과 같으며, $t=1$ 일 때 $L_{0}$ 과 같다.  
즉, 위 식이 최종적인 목적 함수로, 결국 노이즈가 확산되가는 과정의 역을 학습한다는 것은 주어진 이미지가 있을 때 이 이미지에 어떤 노이즈가 섞인 것인지 모델이 예측하는 것과 같음을 의미한다.  


## Implementation

최종적인 목적 함수 식 7에 의해, 학습 알고리즘은 다음과 같이 구성된다.  

![Train_algorithm](https://i.imgur.com/ZO9dWoJ.png)

논문에서 모델의 구조는 UNet을 기반으로 그룹 정규화(Group Normalization) 과 Self-attention이 사용하였다. 단계 $t \quad (1 \leq t \leq T)$ 에 대해서 총 $T$ 개의 모델이 아닌 파라미터를 공유하는 하나의 모델을 사용하고, 단계 $t$ 에 따라 구분하기 위해 $t$ 를 Transformer에서 사용된 Position Embedding을 수행하여 모델에 입력된다. 

![Train_result](https://i.imgur.com/qg9CvtI.png)

위 사진은 이렇게 학습된 Diffusion 모델이 출력한 이미지들로, VAE에서 흐릿한 이미지가 아닌 (DC)GAN 처럼 뚜렷한 이미지가 출력되는 것을 볼 수 있다. 

## vs. Other Generative Models

~~추가 중~~

## 찌라시

일단 딱 이렇게 주어졌을 때 이렇게 돼서 이러이렇게 됐다라는건 알겠는데, 왜 이렇게 주어졌는지가 드럽게 궁금하다. 근데 내 내공이 아직 너무나도 부족하다. 머리가 터질 것 같다. 공부할게 너무 많다. 근데 너무 궁금하다. 언젠가 하나씩 내가 도장깨기 해주마.


## 추가 설명

### Markov Chain의 각 과정이 가우시안 분포가 되는 이유?

앞서서 기체 분자 예시를 계속 이어나가서, 각 스텝의 변화량 $\nabla t$ 를 $0$ 에 가깝게 설정하면, 분자가 다음에 존재할 위치를 가우시안 분포 나타낼 수 있다고 했었다. 아 그렇구나가 아니고 왜 그런건데.

먼저 우리가 보는 거시적인 관점에선 $\nabla t \to 0$ 은 매우 짧은 순간이지만, 미시적인 분자 세계의 관점에서는 겁~~~~~나게 많은 분자들이 서로 충돌하는 아주 긴 시간이다.
구글에 검색해보니 상온, 1기압의 조건에서 공기 중의 기체 분자 1개가 다른 분자와 충돌하는 횟수는 1초에 $10^{9} \sim 10^{10}$ 번이라고 한다. ~~암튼 구글이 그럼~~  
아무튼 이때 각 분자가 튕겨 나가는 방향과 속도는 완전히 독립적인 무작위 확률변수이다. 여기서 통계학에서의 중심극한정리(CLT)에 따라, 이러한 독립적인 무작위 변수가 아주 많이~ 계속 더해지면 그 결과는 가우시안 분포로 수렴한다.  

그럼 분자들의 움직임이 수없이 더해져서 가우시안 분포가 된다면 $\nabla t \to 0$ 로 설정하는 이유는? 바로 $\nabla t$ 가 큰 값을 가지게 되면, 극단적으로 설명하자면 초기 상태에서 한 스텝 만에 $x_{T}$ 가 되버려 초기 상태 $x_{0}$ 를 아에 알아볼 수가 없다. 따라서 $\nabla t \to 0$ 로 설정하여 이전 단계의 상태 $x_{t-1}$ 를 어느정도만큼은 보존하게끔 한다. 

--- 

~~일단 대충 결론 부분만 훑어봤는데 다시 자세히 읽어서 다시 작성하겠다..~~

실제로 공간의 경계가 존재할 때 (여기선 이미지 픽셀의 값의 범위인 $[-1, 1]$.), 분자들의 무작위 궤적 집합이 연속적인 가우시안 확률 과정으로 수렴한다는 사실을 기능적 중심극한정리(Functional Central Limit Theorem, FCLT) 를 통해 증명해 두었다.  

> [Functional central limit theorem for Brownian
particles in domains with Robin boundary
condition](https://arxiv.org/pdf/1404.1442)

### Eq. 1

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

~~추가 중~~


