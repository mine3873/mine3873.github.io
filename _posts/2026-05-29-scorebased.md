---
title: "Generative Modeling by Estimating Gradients of the Data Distribution"
date: 2026-05-31 19:27:00 +0900
categories: [DL]
tags: [dl, generative-model,] 
math: true
mermaid: true
---

# Generative Modeling by Estimating Gradients of the Data Distribution

> [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600)

## Introduction

이 논문에서는 생성형 모델에서 우도함수 기반 (ex. VAE) 이나 Adversarial 훈련 프레임워크 (ex. GAN) 에 기반하지 않고, (Stein) Score 이라는 개념을 사용하여 학습하고 샘플링하는 방법을 설명한다.  

한번 드가보자. 

![DORORUN](https://i.imgur.com/zldQCVa.gif)

## Score-based Generative Modeling

일단 먼저 Score라는게 무엇이냐, 임의의 분포 $p(x)$ 의 Score은 $\nabla_{x} \log p(x)$ 로 정의되며, $\log p(x)$ 의 값이 가장 큰 방향, 즉, 최빈값(Mode)를 향하는 방향을 가리키는 벡터장을 의미한다. 

![쌈뽕한 Score img](https://blog.christianperone.com/wp-content/uploads/2024/11/gaussian_score_animation-1.gif)

위 사진은 이 형님의 블로그에서 쌈뽕해서 가져와봤다...  
[https://blog.christianperone.com/2024/11/the-geometry-of-data-part-ii/](https://blog.christianperone.com/2024/11/the-geometry-of-data-part-ii/)

아무튼튼 이러한 Score 값, 즉, 로그 확률 밀도의 최빈값을 향하는 기울기를 사용하여 기존 분포 $p(x)$ 를 유추할 수 있는데, 자세하게는 [여기](#score-matching)를 확인하자...  

잡설은 뒤로 미루고 결론만 말하자면 이 논문에서의 우리의 목적은 이러한 $\nabla_{x} \log p(x)$ 를 직접적으로 근사하는 Score network $s_{\theta}$ 를 학습하는 것이다. 이제 차근차근 알아보자. 

### Score Matching for Score Estimation

Score network $s_{\theta}$ 를 학습시키기 위해서, 목적함수로 다음과 같은 Score Matching 을 사용한다.  

$$
\begin{align}
    L &= \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)}\left[\left|\left| s_{\theta}(x) - \nabla_{x} \log p_{\text{data}}(x)  \right|\right|_{2}^{2}\right] \\
    &\cdots \quad \text{서커스 시작} \\
    &= \mathbb{E}_{p_{\text{data}}(x)}\left[\text{tr}(\nabla_{x} s_{\theta}(x)) + \frac{1}{2} ||s_{\theta}(x)||_{2}^{2}  \right]
\end{align} \tag*{1}
$$

> 서커스 과정은 [여기](#eq-1)를 확인하자. 

위 식에서 $\nabla_{x} s_{\theta}(x)$ 은 $s_{\theta}(x)$ 의 Jacobian 이다. 

원래 식에 있던 우리가 알고자했지만 알지 못하는 $\nabla_{x}\log p_{\text{data}}(x)$ 가 어디론가로 사라지고, 모델이 에측한 값만을 사용해서 최적화를 진행할 수 있게 되었다. 신기하다.  

아무튼 이 식을 최소화 한다는 것은, 각 항을 살펴보면, 전자의 $\text{tr}(\nabla_{x} s_{\theta}(x))$ 의 벡터장의 대각합이 나타내는 것은 Score 값들의 발산을 의미한다. 이 값들이 양의 값일 때 Score, 즉 화살표들이 사방으로 뿜어져 나가는 형태가 되고, 음의 값일 때 블랙홀 처럼 사방의 화살표들이 현재 점 $x$ 로 빨려들어오는 형태가 된다. 즉, 모든 화살표들이 실제 데이터가 존재하는 방향을 향하도록 하는 역할을 한다.  
 
후자는 Score 값 자체를 최소화한다고 볼 수 있는데, 벡터 장의 화살표들이 최빈값에 도달했을 때, Score 값이 0이 되게끔 하여, 각 Score 값이 무한하게 음수가 되는 것을 방지한다. 

그래서 이 두 항의 앙상블로, 모델 $s_{\theta}$ 이 데이터가 있는 곳으로 Score 를 향하게 하되, 최빈값에 도착하면 Score 값이 0이 되게끔 하는 실제 데이터의 Score 벡터장 $\nabla_{x} \log p_{\text{data}}(x)$ 의 근사 벡터장을 학습하게끔 한다.   

여기서 앞선 항 $\text{tr}(\nabla_{x} s_{\theta}(x))$ 을 계산하는게 거시기 한데, $x \in \mathbb{R}^{d}$ 에 대해서, $\nabla_{x} s_{\theta}(x)$ 을 계산하면 $d \times d$ Jacobian 행렬이 나온다. $\text{tr}(\nabla_{x} s_{\theta}(x))$ 이를 통해서 또 $d$ 개를 더해야 한다. 아무튼 만약 이미지 데이터를 예시로, $32 \times 32$ 크기의 RGB 채널 이미지라고 해도, $32 \times 32 \times 3 = 3072$ 번의 미분 계산이 필요하다. 

아무튼 그래서 이러한 점을 어떻게 다루는지 알아보자. 

#### Denoising score matching

Denoising score matching 은 $x$ 에 바로 score matching 을 사용하지 않고, $x$ 에 노이즈를 추가하여, 노이즈가 낀 $x$ 에 대해 Score matching 을 수행하는 것이다. 이때 이 노이즈는 $q\_{\sigma}(\tilde{x}\|x)$ 로 정의되며, 노이즈 낀 분포는 $q_{\sigma}(\tilde{x}) = \int q_{\sigma}(\tilde{x}\|x) p_{\text{data}}(x)\,dx$ 로 정의된다. 따라서 노이즈를 추가한 후의 목적 함수는 다음과 같다.

$$
\frac{1}{2}\mathbb{E}_{q_{\sigma}(\tilde{x}|x)p_{\text{data}}(x)}\left[ ||s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}|x)||_{2}^{2} \right]
$$

서커스 과정은 [여기를 확인하자..](#denoising-score-matching-서커스-과정)

아무튼 이 목적 함수를 최적화하는 파라미터 $\theta^{\star}$ 에 대해서, $s\_{\theta^{\star}}(\tilde{x}) = \nabla\_{\tilde{x}} \log q\_{\sigma}(\tilde{x})$ 가 최적의 모델이 된다. 왜 $\nabla\_{\tilde{x}} \log q\_{\sigma}(\tilde{x}\|x)$ 이 값을 예측하는게 아닌지는 [여기를 확인하자.....](#denoising-score-matching의-최적의-모델이-예측하는-값이-그렇고-그런-이유)  

여기서 추가되는 노이즈가 아~주 적은, 미미한 경우, 즉 노이즈를 추가하더라도 원본 데이터와 충분히 비슷한 경우에만 $\nabla\_{\tilde{x}} \log q\_{\sigma}(\tilde{x}) \approx \nabla\_{\tilde{x}} \log p_{\text{data}}(\tilde{x})$ 를 만족한다. 

#### Sliced score matching

Sliced score matching은 $\text{tr}(\nabla_{x} s_{\theta}(x))$ 에 무작위 투영 연산을 수행하여 근사하는 방식이다. 즉, 다음과 같다.  

$$
\mathbb{E}_{p_{v}}\mathbb{E}_{p_{\text{data}}}\left[ v^{T}\nabla_{x} s_{\theta}(x)v + \frac{1}{2}\|s_{\theta}(x)\|_{2}^{2}\right]
$$

이때 $p_{v}$ 는 무작위 백터의 다변량 표준 정규 분포와 같은 간단한 분포이다.  

$v^{T}\nabla_{x} s_{\theta}(x)v \to v^{T}(\nabla_{x} s_{\theta}(x)v)$ 여기서 $\nabla_{x} s_{\theta}(x)v$ 이 부분이 $d \times d$ 행렬과 $d \times 1$ 벡터의 곱셈으로, Jacobian-Vector Product, JYP 의 형태이다. 이를 통해서 앞서 살펴봤던 계산이 힘들었던 이유인 높은 차원의 입력 데이터에 대해서 $d \times d$ 연산을 수행할 필요 없이, $d \times 1$ 크기의 $\nabla_{x} s_{\theta}(x)v$ 를 바로 계산함으로써 연산량을 줄인다. 이를 위해서 Forward-mode Auto-differentiation 이 사용되는데, 이 방식은 역전파처럼 결과값에서 시작해서 입력값 방향으로 미분을 전파하는게 아닌, 입력값에서 출력값 방향으로 원본값과 미분값을 동시에 흘려보내는 방식이다. 예를 들어서...

처음에 입력 데이터 $x$ 를 넣을 때, $x$ 의 미분 방향 변수에 샘플링한 무작위 벡터 $v$ 를 넣는다. 그러면 
- 입력: $x$ 
- 미분 입력: $\dot{x} = v$  

여기서 신경망들을 거치면서 덧셈, 곱셈 등등 수행이 될텐데, 그럴 때마다 미분값도 동시에 계산해 나간다. 그러면 최종 출력으로는 다음과 같다.
- 출력: $s_{\theta}(x)$
- 미분 출력: $\nabla_{x}s_{\theta}(x) \cdot \dot{x}$

이 미분 출력에 $v$ 만 추가로 내적해주면 $v^{T}\nabla_{x} s_{\theta}(x)v$ 가 계산된다. 

앞선 방식과는 다르게 SLiced score matching 은 원본 데이터 $x$ 에 대한 score 을 추정하는데, 하지만 $v^{T}\nabla_{x} s_{\theta}(x)v$ 를 계산하는 연산을 추가로 필요로 한다는 점이 있다. 

### Sampling with Langevin Dynamics

Langevin Dynamics 는 다음과 같은 형태의 샘플링 과정이다. 

$$
\tilde{x}_{t} = \tilde{x}_{t-1} + \frac{\epsilon}{2}\nabla_{x} \log p(\tilde{x}_{t-1}) + \sqrt{\epsilon}z_{t}
$$

여기서 $\epsilon > 0$ 은 고정된 학습률(Step size)이고, 초기값 $\tilde{x}\_{0} \sim \pi(x)$ 로 정의되는데 $\pi$ 는 사전 분포이다. 그리고 $z\_{t} \sim \mathcal{N}(0,\mathbf{I})$ 이다.  

- $\tilde{x}_{t-1}$: 현재 위치
- $\frac{\epsilon}{2}\nabla_{x} \log p(\tilde{x}_{t-1})$: 현재 위치에서, 분포의 최빈값의 방향
- $\sqrt{\epsilon}z_{t}$: 약간 최빈값 방향으로 무작정 가지 않고, 무작위 노이즈 값을 통해 각 스텝 $t$ 마다 정신산만한 착한 친구처럼 아주 살짝 미세하게 무작위로 요동침. 이를 통해서 최빈값에 다다르더라도 요리조리 다니며 다른 최빈값으로 이동할 수 있음. 얘 덕분에 지나온 궤적이 분포 $p(x)$ 를 나타내는 느낌

마지막 항 $\sqrt{\epsilon}z_{t}$ 이게 설명이 살짝 거시기한데, 이 항이 없다면, 결국 초기 위치 $x_{0}$ 에서 샘플링을 시작하여, 분포의 최빈값의 방향으로 곧장 향하게만 될 것이다. 그리고 최빈값에 도착하면 더 이상 위치 변화가 없을 텐데 ($\nabla\_{x} \log p(\tilde{x}\_{t-1}) \approx 0$), 그냥 데이터 분포의 한 점으로 수렴하여 분포를 묘사한다는 것이다. 즉, 초기 위치가 다르더라도 같은 점으로 수렴해버려, 결국 어느 초기 위치에서 시작하든 아래와 같이 동일한 데이터만 샘플링될 것이다. 

| ![할짝할짝](https://i.imgur.com/wyjE1qf.gif) | ![할짝할짝](https://i.imgur.com/wyjE1qf.gif) | ![할짝할짝](https://i.imgur.com/wyjE1qf.gif) |
| :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
| ![할짝할짝](https://i.imgur.com/wyjE1qf.gif) | ![할짝할짝](https://i.imgur.com/wyjE1qf.gif) | ![할짝할짝](https://i.imgur.com/wyjE1qf.gif) |

최종 위치가 국소적 최대값(Local Maximum) 이나 안장점(Saddle point) 인 경우에도 멈춰설 수 있고.... 아 그냥 저 항이 없으면 확률 분포를 반영하지 못한다는 것이다. 

식을 보면 모델 $s_{\theta}$ 이 예측할 Score $\nabla\_{x} \log p(\tilde{x}\_{t-1})$ 만을 사용해서 $p(x)$ 에서 샘플링을 수행할 수 있음을 알 수 있는데, $\epsilon \to 0$, $T \to \infty$ 일 때, $\tilde{x}_{T}$ 의 분포는 $p(x)$ 와 같아진다. 

$\epsilon > 0$, $T < \infty$ 의 경우엔 그만큼 오차가 심해질텐데, 이를 교정하기 위해 Metropolis–Hastings algorithm 란걸 사용할 수 있지만, 논문에서는 딱히 안써도 된다고 한다.  

## Challenges of Scores-based Generative Modeling 

![애야..원래..생략](https://i.imgur.com/vWh7c8P.png)

출세하고 성공의 길인 줄 알았는데 아쉽게도 풀어야할 두 개의 족쇄가 남아있었다. 어떤건지 알아보자...

### The manifold hypothesis

일단 먼저 대부분의 데이터들은 거대한 고차원 공간 속에서 아주 작은 저차원 매니폴드(Manifold) 에만 몰려있다. (이를 Manifold Hypothesis 라고 한다.) 그러니까 예를 들어서... 다양한 도로롱들이, 아니다. 그냥 $32 \times 32$ 크기 RGB 채널의 고양이 이미지를 예시로 들자면, 고양이 데이터는 $32 \times 32 \times 3 = 3072$ 차원의 공간에서의 부분 공간이다. 3채널의 $32 \times 32$ 로 나타낼 수 있는 이미지들에 대해서 고양이 이미지로 분류되는 비율이 얼마나 될까. 그냥 겁~나게 적다는 것이다.  

이 때문에 2가지 문제가 생기는데, 먼저 Score $\nabla_{x} \log p_{\text{data}}(x)$ 를 정의할 수 없다는 것이다. 음.. 그러니까 이 거대한 우주에서 지구의 빛(Score, ~~그냥 흐린 눈으로 지구 빛이라고 하자.~~) 을 추적하여 지구 ($p_{\text{data}}$) 를 찾아야하는데, 우주는 겁~나게 넓으니까 저~ 먼 곳에서는 이 지구 빛을 볼 수 조차 없다. Score 값이 안보이는거다. 아무튼 내가 이해한 바는 이런데.. 계속 짱구 돌려보다가 나중에 수정하겠다..

그러니까.. $\nabla_{x} \log p_{\text{data}}(x) = \frac{\nabla_{x} p_{\text{data}}}{p_{\text{data}}}$ 로 정의되는데, 이 매니폴드와 멀리 떨어지면 $p_{\text{data}} \approx 0$ 이 되어서, $\frac{\nabla_{x} p_{\text{data}}}{p_{\text{data}}}$ 이게 정의가 되지 않는 것이다.  

..아무튼 다른 문제는 앞서 살펴본 우리의 목적함수에 대해서, $p_{\text{data}}$ 의 서포트가 전체 공간일 때만 일관된 Score 값을 제공하며, 저차원 매니폴드에 있다면 그렇지 않다는 것이다. 

여기서 분포의 서포트는, $p(x) > 0$, 즉, 데이터가 존재하는 영역을 말한다. 여기서 매니폴드는 이러한 서포트가 가지는 저차원 기하학 구조를 말한다... ~~아 뭔가 딱 감올 것 같은데 거시기하다. 나중에 제대로 다뤄보겠다..~~

![Fig1](https://i.imgur.com/niKs52k.png)

그래서 논문 저자 형님들이 이 족쇄가 어떤 영향을 주는지 실험해본 결과, 위 사진의 왼쪽과 같다. Sliced score matching 을 통해 학습을 진행했는데, 손실값이 감소하다 막 불규칙하게 요동치고 난리도 아니다.  

오른쪽은 소량의 가우시안 노이즈를 전체 공간에 추가하여, 노이즈가 섞인 데이터에 대해 동일하게 학습을 진행한 것이다. 보다시피 학습이 야무지게 수렴하는 것을 볼 수 있다. 여기서 가우시안 노이즈는 $\mathcal{0, 0.0001}$ 로 진짜 엄~청 소량인데, 그럼에도 저렇게 전혀 다른 결과가 나왔다는 것이다. 

### Low data density regions 

~~추가 중~~



## 추가 설명

### Score matching

Score matching 에 대해 알아보기 전에, Energy-based Model, EBM 이란걸 아주 사알짝만 다뤄보자면...  

어떠한 $p_{\text{data}}$ 분포가 있을 때, 이를 직관적으로 나타내기 쉬운 방법은 EBM 의 형태로 $p_{\text{data}}$ 를 근사하는 것이다. 즉, 

$$
p_{\text{data}} \approx p_{\theta}(x) = \frac{\exp(-f_{\theta}(x))}{Z} = \frac{\exp(-f_{\theta}(x))}{\int \exp(-f_{\theta}(x))\, dx}
$$

요런 식으로 $\theta$ 를 파라미터로 가지는 함수 $f_{\theta}(x)$ (에너지 함수 $E_{\theta}(x)$ ) 를 통해 기존 분포 $p_{\text{data}}$ 를 근사하는 방식으로, 에너지 함수는 데이터가 자연스러울 때, 낮은 값을 반환하고, 자연스럽지 않은 데이터일 때 높은 값을 반환한다.

아무튼 기존 Score matching은 이렇게 EBM 으로 근사한 모델의 Score, 즉, $-\nabla_{x} f_{\theta}(x)$ 를 기본 분포의 Score $\nabla_{x} p_{\text{data}}$ 와 같아지도록 파라미터 $\theta$ 를 조절하여 학습하는 것이다.  

하지만 이 논문에서는 이렇게 에너지 함수의 Score을 계산하여 기존 데이터 분포의 Score에 근사하는 식으로 에너지 함수를 학습하는 것이 아니라, 에너지 함수고 뭐고 그냥 데이터 분포의 Score 자체를 근사하는 $s_{\theta}$ 를 근사하는 것이다.  

$$
\begin{align}
    \text{EBM} &: \nabla_{x} f_{\theta}(x) \approx \nabla_{x} p_{\text{data}}(x) \\
    \text{score-based} &: s_{\theta}(x) \approx \nabla_{x} p_{\text{data}}(x)
\end{align}
$$

아무튼 이를 통해서 $\nabla_{x} f_{\theta}(x)$ 과 같이 에너지 함수를 따로 미분하는 계산을 따로 필요로 하지 않는다. 

### Eq. 1

$$
\begin{align}
    L &= \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)}\left[\left|\left| s_{\theta}(x) - \nabla_{x} \log p_{\text{data}}(x)  \right|\right|_{2}^{2}\right] \\

    &= \frac{1}{2} \int p_{\text{data}}(x)||s_{\theta}(x) - \nabla_{x} \log p_{\text{data}}(x) ||_{2}^{2} \, dx \quad \left(\mathbb{E}_{p_{\text{data}}(x)} = \int p_{\text{data}}(x)\,dx = 1 \right) \\

    &= \frac{1}{2} \int p_{\text{data}}(x) \left[||s_{\theta}(x)||_{2}^{2} + ||\nabla_{x} \log p_{\text{data}}(x)||_{2}^{2} - 2s_{\theta}(x)^{T}\nabla_{x} \log p_{\text{data}}(x) \right] \,dx \\

    &= \frac{1}{2} \int p_{\text{data}}(x) ||s_{\theta}(x)||_{2}^{2}\,dx + \frac{1}{2} \int p_{\text{data}}(x) ||\nabla_{x} \log p_{\text{data}}(x)||_{2}^{2} - \underbrace{\int p_{\text{data}}(x)s_{\theta}(x)^{T}\nabla_{x} \log p_{\text{data}}(x) \, dx}_{(1)} \\

    (1) &\implies \int p_{\text{data}}(x) \sum_{i=1}^{d} \left(s_{\theta,i}(x) \nabla_{x_{i}}\log p_{\text{data}}(x) \right) \, dx \\
    
    &= \sum_{i=1}^{d} \int s_{\theta,i}(x) \nabla_{x_{i}}p_{\text{data}}(x) \quad \left( \nabla_{x} \log p_{\text{data}}(x) = \frac{\nabla_{x} p_{\text{data}}(x)}{p_{\text{data}}(x)}\right)\\

    &= \sum_{i=1}^{d}\left[ \underbrace{s_{\theta}p_{\text{data}}(x)|_{-\infty}^{\infty}}_{(2)} - \int p_{\text{data}}(x)\nabla_{x_{i}}s_{\theta,i}(x)\,dx\right] \quad \left( \int \nabla v \cdot u  =  vu|_{a}^{b} - \int v \cdot \nabla u \right) \\
    
    &= - \sum_{i=1}^{d} \int p_{\text{data}}(x)\nabla_{x_{i}}s_{\theta,i}(x)\,dx \\

    &= - \mathbb{E}_{p_{\text{data}}(x)}\left[ \sum_{i=1}^{d} \nabla_{x_{i}}s_{\theta,i}(x)\right] = -\mathbb{E}_{p_{\text{data}}(x)}[\text{tr}(\nabla_{x_{i}}s_{\theta}(x))] \\

    \\
    L &= \frac{1}{2} \int p_{\text{data}}(x) (s_{\theta}(x))^{2}\,dx + \frac{1}{2} \int p_{\text{data}}(x) (\nabla_{x} \log p_{\text{data}}(x))^{2} + \mathbb{E}_{p_{\text{data}}(x)}[\text{tr}(\nabla_{x_{i}}s_{\theta}(x))] \, dx \\
    &= \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)}[(s_{\theta}(x))^{2}] + \underbrace{\frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)}[(\nabla_{x} \log p_{\text{data}}(x))^{2}]}_{\text{$\theta$ 와 관련 없으므로 훈련 중 무시가능}} + \mathbb{E}_{p_{\text{data}}(x)}[\text{tr}(\nabla_{x_{i}}s_{\theta}(x))] \\
    &= \mathbb{E}_{p_{\text{data}}(x)}[\text{tr}(\nabla_{x_{i}}s_{\theta}(x))] + \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)}[(s_{\theta}(x))^{2}]
\end{align}
$$

### Denoising score matching 서커스 과정

노이즈가 추가된 분포 $q_{\sigma}(\tilde{x}) = \int q_{\sigma}(\tilde{x}\|x) p_{\text{data}}(x)\,dx$ 에 score matching 은 다음과 같다.

$$
\begin{align}
    L &= \frac{1}{2}\mathbb{E}_{q_{\sigma}(\tilde{x})}\left[||s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}}\log q_{\sigma}(\tilde{x})||_{2}^{2} \right] \\
    &= \frac{1}{2} \int q_\sigma(\tilde{x}) \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) \right\|_2^2 \, d\tilde{x} \\
    &= \frac{1}{2} \int q_\sigma(\tilde{x}) \left\| s_\theta(\tilde{x})\right\|_2^2 \, d\tilde{x} - \underbrace{\int q_\sigma(\tilde{x})s_{\theta}(\tilde{x})^{T} \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) \, d\tilde{x}}_{(1)}  + \text{$\theta$ 와 관련 없는 상수} \\
    &\cdots\\
    (1) &\implies \int  q_\sigma(\tilde{x}) s_{\theta}(\tilde{x})^{T} \frac{\nabla_{\tilde{x}} q_{\sigma}(\tilde{x})}{q_{\sigma}(\tilde{x})} \, d\tilde{x} = \int s_{\theta}(\tilde{x})^{T}\nabla_{\tilde{x}} q_{\sigma}(\tilde{x}) \, d\tilde{x} \\
    &= \int s_{\theta}(\tilde{x})^{T} \nabla_{\tilde{x}} \left(\int q_{\sigma}(\tilde{x}|x) p_{\text{data}}(x)\,dx\right) \, d\tilde{x} \\
    &= \int \int p_{\text{data}}(x) \left(s_{\theta}(\tilde{x})^{T} \nabla_{\tilde{x}}q_{\sigma}(\tilde{x}|x) \right)\, dx \, d\tilde{x} \\
    &= \int \int p_{\text{data}}(x) q_{\sigma}(\tilde{x}|x) \left( s_{\theta}(\tilde{x})^{T} \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}| x) \right)\, dx \, d\tilde{x} \quad (\nabla q = q \nabla \log q) \\
    &\cdots\\
    L &= \frac{1}{2} \int q_\sigma(\tilde{x}) \left\| s_\theta(\tilde{x})\right\|_2^2 \, d\tilde{x} + \int \int p_{\text{data}}(x) q_{\sigma}(\tilde{x}|x) \left( s_{\theta}(\tilde{x})^{T} \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}| x) \right)\, dx \, d\tilde{x} + C\\
    &= \frac{1}{2} \int \int p_{\text{data}}(x) q_\sigma(\tilde{x}) \left\| s_\theta(\tilde{x})\right\|_2^2\, dx \, d\tilde{x} + \int \int p_{\text{data}}(x) q_{\sigma}(\tilde{x}|x) \left( s_{\theta}(\tilde{x})^{T} \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}| x) \right)\, dx \, d\tilde{x} + C \\
    &= \frac{1}{2} \int \int p_{\text{data}}(x) q_\sigma(\tilde{x})\left\| s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}| x) \right\|_2^2 \, dx \, d\tilde{x} \\
    &= \frac{1}{2}\mathbb{E}_{q_{\sigma}(\tilde{x}|x)p_{\text{data}}(x)}\left[ ||s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q\_{\sigma}(\tilde{x}|x)||_{2}^{2} \right]
\end{align}
$$

### Denoising score matching의 최적의 모델이 예측하는 값이 그렇고 그런 이유

$$
\frac{1}{2}\mathbb{E}_{q_{\sigma}(\tilde{x}|x)p_{\text{data}}(x)}\left[ ||s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}|x)||_{2}^{2} \right]
$$

목적함수를 다시 살펴보면 $s\_{\theta}(\tilde{x})$ 와 $\nabla\_{\tilde{x}} \log q\_{\sigma}(\tilde{x}\|x)$ 사이의 평균 제곱 오차, MSE 의 형태이다.   

여기서 우리의 모델 $s\_{\theta}(\tilde{x})$ 이 입력값으로 노이즈가 낀 $\tilde{x}$ 만을 가지며, 원본 데이터 $x$ 를 따로 입력 받지 못하는데, 통계에서, $X$ 를 모른 채 주어진 $Y$ 만 보고 오차 제곱 기댓값을 최소화하는 최적의 함수 $f^{\star}(Y) = \arg\min_f \mathbb{E}[(f(Y) - X)^2]$ 을 구하면, 이 값은 항상 조건부 기댓값 $\mathbb{E}[X\|Y]$ 가 된다.  

이를 $q_{\sigma}(x, \tilde{x}) = q_{\sigma}(\tilde{x}\|x)p_{\text{data}}(x)$ 를 통해 목적 함수에 대입하면, 특정 $\tilde{x}$ 가 주어졌을 때, 모델 $s\_{\theta}(\tilde{x})$ 이 가질 수 있는 최적의 값 $s_{\theta^\star}(\tilde{x})$ 은 다음과 같이 $x$ 에 대한 조건부 기댓값 형태로 정의된다.

$$
s_{\theta^\star}(\tilde{x}) = \mathbb{E}_{x \sim q_\sigma(x|\tilde{x})} \left[ \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}|x) \right]
$$

이 식을 베이즈 정리를 통해 서커스를 진행하면 다음과 같다.

$$
\begin{align}
    \mathbb{E}_{x \sim q_\sigma(x|\tilde{x})} \left[ \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}|x) \right] &= \int q_{\sigma}(x|\tilde{x}) \nabla_{\tilde{x}}\log q_{\sigma}(\tilde{x}|x) \, dx\\
    &= \int \frac{q_{\sigma}(\tilde{x}|x)p_{\text{data}}(x)}{q_{\sigma}(\tilde{x})} \frac{\nabla_{\tilde{x}}q_{\sigma}(\tilde{x}|x)}{q_{\sigma}(\tilde{x}|x)} \, dx\\
    &= \frac{1}{q_{\sigma}(\tilde{x})} \int p_{\text{data}}(x) \nabla_{\tilde{x}}q_{\sigma}(\tilde{x}|x) \, dx\\
    &= \frac{1}{q_{\sigma}(\tilde{x})}\nabla_{\tilde{x}} \underbrace{\int p_{\text{data}}(x) q_{\sigma}(\tilde{x}|x)}_{q_{\sigma}(\tilde{x})} \, dx\\
    &= \frac{\nabla_{\tilde{x}} q_{\sigma}(\tilde{x})}{q_{\sigma}(\tilde{x})} = \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x})
\end{align}
$$
