---
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
date: 2026-06-08 20:27:00 +0900
categories: [DL, Generative-model]
tags: [dl, generative-model, diffusion] 
math: true
mermaid: true
---

# High-Resolution Image Synthesis with Latent Diffusion Models

> [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
{: .prompt-info} 

![](https://i.imgur.com/F7tBMY8.gif)

## Introduction

기존 Diffusion model, DM은 우도함수 기반의 모델로, 뛰어난 성능을 보였다. 
하지만 단점이 있는데, 데이터를 복원할 때 Mode-covering behavior, 즉, 모든 영역을 커버하려는 성향을 가진다.  
다시 말해서 이미지의 경우 우리가 보이지 않는 아주 미세한 디테일도 학습하여 과도한 자원을 사용한다는 것이다.  
거기다가 DM은 학습과 추론 과정에서의 Backbone 모델의 반복적인 사용도 과도한 자원 사용의 요인 중 하나이다.  

때문에 DM의 성능을 가지면서도 어떻게 하면 사용되는 자원을 줄이기 위해 논문에서 어떻게 했는지 알아보자.  

그전에 Distortion 과 Rate에 대해 짧게만 알아보자..  
distortion, $\mathbf{D}$ 는 다음과 같다.

$$
\mathbf{D} = \mathbb{E}_{x\sim p(x), z\sim \mathcal{E}(z|x)}[-\log \mathcal{D}(x|z)] \propto \|x - \hat{x}\|^{2}
$$

이때 $p(x)$ 는 실제 데이터의 분포, $\mathcal{E}(z\|x)$ 는 원본 데이터 $x$ 가 주어졌을 때 인코더가 출력한 $z$ 의 분포, $\mathcal{D}(x\|z)$ 는 $z$ 가 주어졌을 때 디코더가 복원한 $x$ 의 분포이며, $\hat{x}$ 는 디코더에 잠재변수 $z$ 를 넣었을 때의 출력이다.  

Distortion은 간단하게 생각하면 기존 데이터 $x$ 와 모델이 뱉은 데이터 $\hat{x}$ 의 MSE 와 비례한다. 한마디로 모델이 얼마나 잘 복원했는가를 나타내는 지표이다.  

Rate, $\mathbf{R}$ 은 다음과 같다..

$$
\mathbf{R} = \mathbb{E}_{p(x)}[D_{KL}(\mathcal{E}(z|x) \| m(z))]
$$

이때 $m(z)$ 는 잠재공간에 대한 사전 분포이다.  

Rate는 인코더가 출력한 잠재 공간의 분포가 사전 분포와 얼마나 다른지 평균을 낸 값으로, 잠재 공간에 정보를 넣을 때 사용하는 데이터 용량 수로 볼 수 있다. 

![rate-distortion trade-off](https://i.imgur.com/UyMciOM.png)

위 사진은 픽셀 공간에서 rate-distortion trade-off 를 나타내는데, 우도함수 모델의 경우, 학습을 크게 2가지로 나눌 수 있다.   
첫번째는 Perceptual compression 으로, Rate 가 크게 줄어드는 구간을 말한다. 보면 Rate 가 크게 줄어도 선글라스 낀 남성의 전체적인 형태가 유지되는데, Distortion 도 0에 가까운 것을 볼 수 있다. 즉, 우리가 보기엔 거기서 거기인데 아주 세부적인 디테일만 처리하는 단계이다.

두번째는 Semantic compression 으로, Rate가 줄어들면서 Distortion이 크게 증가하는 구간을 말한다. 사진을 보면 Rate 가 0.5 이하로 줄어드니 이미지가 전체적으로 파괴되면서 선글라스 낀 남성이 다른 여자 얼굴로 변하는 것을 볼 수 있다. 즉, 이 부분에 데이터의 본질적인 정보가 압축되어 있는 것이다.  

기존 DM 은 이 두 구간을 픽셀 공간에서 모두 담당했는데, 이 논문에서 제시한 Latent Diffusion Models, LDMs 는 사소한 디테일 부분인 Perceptual compression 은 Auto-Encoder 를 사용하여 따로 처리하고, 이 Semantic compression 부분만 처리하게끔 하여, 학습에 필요한 자원을 줄이고자 한다. 

그러니까, Auto-Encoder의 인코더로 픽셀 공간 데이터를 핵심 정보를 보존하는 저차원의 잠재 공간으로 압축하고, 이 축소된 잠재 공간에서 DM을 수행한 후에, 다시 디코더로 데이터 공간으로 복원하는 것이다.  

기존 픽셀 데이터 공간보다 축소된 잠재 공간에서 DM 을 수행하기 때문에, 학습과 샘플링 모두 데이터 공간에서 수행하던 기존 DM 보다 확실히 적은 계산 자원을 사용한다. 

또한 하나의 Auto-Encoder 모델을 사용하여, 잠재 공간으로 축소와 복원을 학습해두면, 그 학습된 잠재 공간에서 목적에 맞는 다양한 DM 을 개별적으로 학습시킬 수 있다. (로컬 AI 이미지 생성 프로그램의 경우, 실제로 Stable diffusion v1.4, v1.5 등의 DM 들이 동일한 VAE 모델을 공유하기도 한다.)

이제 Auto-Encoder와 DM 모델 각각 어떻게 수행하는지 알아보자.

## Models

### Auto-Encoder

기존 데이터 $x \in \mathbb{R}^{H \times W \times C}$와 인코더 $\mathcal{E}$, 디코더 $\mathcal{D}$ 에 대해서, 그냥 기존 오토인코더와 같다.  

$$
\begin{align}
    \hat{x} &= \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x)) \\
    z &= \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c} \\
\end{align}
$$

여기서 $f = 2^{m} \quad (m \in \mathbb{N})$ 는 인코더의 다운 샘플링의 정도이고, $h = \frac{H}{f}$ , $w = \frac{W}{f}$ 로 정의된다.  

VQGAN 의 방식으로 오토인코더를 GAN에서의 Adversarial 학습 방식으로 다음과 같이 학습한다.

$$
L = \underset{\mathcal{E},\mathcal{D}}{\arg\min} \underset{D}{\max}\left(L_{\text{rec}}(x, \hat{x}) + \lambda(\underbrace{\log D(x) + \log ( 1 - D(\hat{x})))}_{L_{\text{GAN}}} + L_{\text{reg}}(x;\mathcal{E},\mathcal{D})\right)
$$
 
하나씩 하나씩 살펴보자. 

#### Reconstructin Loss

$$
L_{\text{rec}}(x, \hat{x}) = \sum_{l} \frac{1}{H_{l}W_{l}C_{l}} \|\Phi_{l}(x) - \Phi_{l}(\hat{x})\|_{2}^{2} 
$$

이때 $\Phi_{l}(.)$ 는 VGG-16 모델의 $l$ 번째 레이어에서의 Feature Map이다. 

VGG-16 은 간단하게 그냥 $3 \times 3$ 필터만 사용해서 층을 겁나게 쌓은 건데, $16$ 이 의미하는건 학습 가능한 레이어의 수가 $16$ 라는 것이다. 아무튼 겁나게 생략에서, VGG-16 모델은 총 5개의 레이어를 가지며, 특성 상 초기 레이어의 Feature Map 에선 점, 선 과 같은 Low-level 의 특징을 잡고, 후반 레이어로 갈 수록 High-level 의 특징을 잡는다. 

아무튼 그래서 저 식은 $x, \hat{x}$ 에 대한 각 레이어의 출력, Feature Map 끼리 비교하는 것이다. 아무튼 이를 통해서 VAE에서 봤듯이, 기존 L2 Loss 를 사용했을 때의 픽셀 하나하나 비교했을 때 나타난 이미지의 블러(?) 흐림(?) 과 다르게, 다양한 Level의 형태에서 비교하기 때문에, 맨 처음 이미지에서 봤듯이 Rate가 크게 감소할 때 퀄리티 박살나던 것을 어느정도 막을 수 있다.  

그래도 기존 L2 Loss $\|\|x - \hat{x}\|\|^{2}$ 처럼 픽셀 하나하나 대조하진 않기 때문에, 저기에 L1 Loss를 추가하기도 한다.  

아무튼 최종 식은 다음과 같다.

$$
\therefore L_{\text{rec}}(x, \hat{x}) = \|x - \hat{x}\|_{1} + \sum_{l} \frac{1}{H_{l}W_{l}C_{l}} \|\Phi_{l}(x) - \Phi_{l}(\hat{x})\|_{2}^{2} 
$$

#### GAN Loss

$$
L_{\text{GAN}} = \log \mathcal{D}(x) + \log ( 1 - \mathcal{D}(\hat{x}))
$$

이건 뭐... 전에 다뤘으니 패스하겠다. 절대 귀찮은 것이 아니다.  

전체 Loss 에서 가중치 $\lambda$ 가 곱해지는데, 보통 VQGAN 에서의 방식인 다음과 같은 Adaptive Weighting 을 사용한다.  

$$
\lambda = \frac{\nabla_{\mathcal{D}_{L}}[L_{\text{rec}}]}{\nabla_{\mathcal{D}_{L}}[L_{\text{GAN}}] + \delta}
$$

여기서 $\nabla_{\mathcal{D}_{L}}[.]$ 는 디코더의 마지막 레이어 $L$ 에 대한 Loss의 Gradient를 의미하고, $\delta \approx 10^{-6}$ 이다. 

#### Regularization

$$
L_{\text{reg}}(x;\mathcal{E},\mathcal{D})
$$

이건 Auto-Encoder의 정규화 항인데, 두 가지 방식을 논문에서 제시한다.  

하나는 기존 VAE 처럼 표준 정규 분포와의 KL-divergence 를 통해 정규화하거나,  
다른 하나는 VQGAN 그대로 Vector Quantization, VQ 를 하는 것이다.  

하나 씩 알아보자. 

##### KL-Regularization

> [VAE 참고](https://mine3873.github.io/posts/vae/#%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98-%EA%B3%84%EC%82%B0)
{: .prompt-info} 

그냥 VAE랑 똑같다. 인코더에서 평균과 분산을 출력한 후, $z = \mu + \sigma \odot \epsilon$ 로 재파라미터화 해서.. 진행하는거다. 아무튼 이러한 $\mu, \sigma$ 를 가지고 표준 정규 분포와 가까워지도록 하는데, 그러면 Loss는 다음과 같다.

$$
L_{\text{reg}}(x;E) = -D_{KL}((\mathcal{E}(z|x) \| \mathcal{N}(z;0,\mathbf{I})) = \frac{1}{2}\sum_{j=1}^{J}(1 + \log(\sigma_{j}^{2}) - \mu_{j}^{2} - \sigma_{j}^{2})
$$

자세한건 위 블로그 글을 보자...  

이렇게 정규화하긴 하지만, 확실하게 잠재 공간 $z$ 의 분산을 $1$ 로 설정하기 위해서 다음과 같은 작업을 추가하기도 한다.  

$$
\begin{align}
    \hat{\mu} &= \frac{1}{bchw}\sum_{b,c,h,w}z^{b,c,h,w} \\
    \hat{\sigma}^{2} &= \frac{1}{bchw}\sum_{b,c,h,w}(z^{b,c,h,w} - \hat{\mu})^{2} \\
    \therefore z &:= \frac{z}{\hat{\sigma}}
\end{align}
$$

여기서 보통 $\hat{\mu}, \hat{\sigma}$ 를 계산할 땐, 학습 데이터의 첫 번째 배치를 통해 계산하고, 이 값을 통해 모든 데이터에 대해 $z$ 값을 조정한다.  

이게 끝이다. 짜자잔. 

##### VQ-Regularization

Vector Quantization, VQ 도 쉽다. 그냥 연속적인 값을 가지는 잠재공간에 대해서, 이를 이산형 데이터로 나타내는 것이다. 이산형 데이터로 어떻게 나타내느냐, 이산형 데이터들이 있는 Codebook $\mathcal{Z}$ 란 개념이 있다. 여기엔 각 이산형 데이터 $z_{k} \in \mathbb{R}^{n_{z}} \quad (k = 1,...,\|\mathcal{Z}\|)$ 가 백과사전 마냥 나열되어 있다.(이때 $n_{z}$ 는 Codebook 내의 데이터의 수이다.) 그러면 연속형 데이터에서 Codebokk 내의 이산형 데이터로의 매핑 함수를 $q(.)$ 라고 했을 때, 다음과 같다.  

$$
q(\mathcal{E}(x)) = \left(\underset{z_{k} \in Z}{\arg\min} \| \mathcal{E}(x) - z_{k}\|\right) \in \mathbb{R}^{h \times w \times n_{z}}
$$

그냥 제일 가까운 값으로 바꾸는 것이다. 이러한 Codebook $\mathcal{Z}$ 도 학습의 대상인데, 다음과 같이 학습한다.  

$$
L_{\text{reg}}(x;\mathcal{E},\mathcal{Z}) = \|sg[\mathcal{E}(x)] - q(\mathcal{E}(x))\|_{2}^{2} + \|sg[q(\mathcal{E}(x))] - \mathcal{E}(x) \|_{2}^{2}
$$

여기서 $sg[.]$ 는 Stop-Gradient 로, 역전파에 사용되지 않도록 파이썬의 `.detach()` 와 같이 그냥 값만 같은 별개의 상수이다.  

식을 보면 인코더가 출력한 잠재 공간이 Codebook 과 가까워지게끔 하며, 동시에 그 반대도 수행하고 있음을 알 수 있다. 

아무튼 이를 통해서 인코더가 출력한 잠재 공간 들이 Codebook 근처로 달라붙으며 정규화를 수행한다. 

물론 DM은 연속형 실수 공간에서 활동하기 때문에, 잠재 공간을 이산형으로 나타내기 전에 DM 모델을 수행한 후에, 이산형으로 나타내어 디코더로 전달되게끔 한다.  

---

이렇게 학습하면?? 된다..  
![](https://i.imgur.com/F7tBMY8.gif)

### Latent Diffusion Models, LDMs

앞서 Auto-Encoder를 딱 학습을 끝냈다? 그러면 이제 학습된 잠재공간 내에서 DM 을 학습하면 된다.  
딱히 기존 DM 과 달라진건 없다. 그냥 기존 $x \in \mathbb{R}^{H \times W \times C}$ 에서가 아니라 $z \in \mathbb{R}^{h \times w \times c}$ 에서 모델을 돌리는 것이다.  

....DM 도 전에 살펴봤으니 생략하고, 전체 Loss는 다음과 같다.

$$
L_{LDM} = \mathbb{E}_{E(x), \epsilon \sim \mathcal{N}(0,1),t}\left[\| \epsilon - \epsilon_{\theta}(z_{t}, t)\|_{2}^{2}\right]
$$

여기서 $z_{t}$ 는 $x_{t}$ 와 같이 $t$ 단계에서의 노이즈가 섞인 잠재 공간 데이터이다.  

...짜잔, 끝이다. Conditional LDM 도 알아보자. 

### Conditional LDMs

![](https://i.imgur.com/3mg7isD.png)

Conditional LDMs 의 전체 구조는 위와 같다.  

어... 우리가 뭐 `달리는 도로롱 이미지 그려줘` 와 같은 텍스트 프롬프트나, 마스크 등의 조건부 $y$ 가 사용될 수 있는데, 이러한 조건 $y$ 가 주어졌을 때의 전체 Denoising 과정은 다음과 같다고 볼 수 있다.  

$$
p_{\theta}(z|y) = p(z_{T})\prod_{t=1}^{T} p_{\theta}(z_{t-1}|z_{t},y)
$$

이러한 조건을 모델에 접목시키기 위해 전용 인코더 $\tau_{\theta}(.)$ 를 도입하는데, 이를 통해 조건 $y$ 를 우리 입맛에 맞게 변환한 다음, 우리의 Backbone UNet 에 접목시킴으로써 수행할 수 있다.  

여기서 $\tau_{\theta}(y)$ 가 우리의 잠재 공간과 어떠한 위상 공간 관계를 가지느냐에 따라 다른 방식으로 모델에 접목시킬 수 있는데,  

음.. 예를 들어서 Inpainting 이나, img2img 와 같이, 조건 $y$ 가 동일한 위상 공간을 가지는 경우, 즉, 1:1 로 위치가 매칭되는 공간 정보를 사용할 때, $\tau_{\theta}(y) \in \mathbf{R}^{h \times w \times c'}$ 를 모델 UNet의 입력에 채널 차원에 concat하여 수행한다. 즉, Backbone UNet의 입력은 $\tilde{z}_{t} \in \mathbf{R}^{h \times w \times (c + c')}$ 와 같이 확장된 형태가 된다. 

반대로 다른 위상 공간을 가지는 경우, 예를 들어서 `달리는 도로롱 이미지 그려줘` 와 같은 텍스트 조건에 대해선, 구조가 완전히 다르기 때문에, 트랜스포머 기반의 $\tau_{\theta}$ 로, 시퀀셜로 매핑하여 사용한다.  

이때 기존 UNet 모델의 self-attention 블록 부분을 cross-attention 으로 바꾸어 사용하는데, $\varphi_{i}(z_{t}) \in \mathbf{R}^{N \times d_{\epsilon}^{i}}$ 가 $\epsilon_{\theta}$ 의 해당하는 부분 출력값을 펼친 것이고, $\tau_{\theta}(y) \in \mathbf{R}^{M \times d_{\tau}}$ 가 $\tau_{\theta}$ 에 의해 매핑된 조건 $y$ 라고 할 때, 각 Backbone UNet 의 Cross-attention 은 다음과 같이 수행된다.  

$$
\begin{align}
    \text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^{T}}{\sqrt{d}} \right) \cdot V \\
    Q &= W_{Q}^{(i)} \cdot \varphi_{i}(z_{t}) \quad \left(W_{Q}^{(i)} \in \mathbf{R}^{d \times d_{\epsilon}^{i}}\right) \\
    K &= W_{K}^{(i)} \cdot \tau_{\theta}(y) \quad \left(W_{K}^{(i)} \in \mathbf{R}^{d \times d_{\tau}}\right) \\
    V &= W_{V}^{(i)} \cdot \tau_{\theta}(y) \quad \left(W_{V}^{(i)} \in \mathbf{R}^{d \times d_{\tau}}\right) 
\end{align}
$$

그 외에도, $\tau_{\theta}(y)$ 가 클래스 라벨 처럼 단일 벡터인 경우엔, UNet의 각 ResBlock에 tiemstep $t$ 임베딩을 더하는데 꼽사리 껴서 같이 더하는 식으로, 특정 레이어 전체에 일괄적인 변환을 수행할 수도 있다.  

아무튼 Conditional LDMs Loss 는 다음과 같다.

$$
L_{LDM} = \mathbf{E}_{E(x), y, \epsilon \sim \mathcal{N}(0,1),t}\left[\|\epsilon - \epsilon_{\theta}(z_{t}, t, \tau_{\theta}(y)) \|_{2}^{2}\right]
$$

여기서 $\epsilon_{\theta}$ 와 인코더 $\tau_{\theta}$ 를 동시에 최적화한다.  

## 찌라시시

![](https://i.imgur.com/F7tBMY8.gif)

국토종주하고 몸살나서 조금 늦어졌는데, 다시 달려보도록 하겠다.  
역시나 글 가독성이 개나 줘버렸는데, 나름대로 초안쓰고 두 번 다시 회독하면서 수정한 것이다... 나름 최선이다. 그래도 계속 노력해보겠다.  

근데 그냥 내 공부 용인데 필요한가.. 짜피 나중에 논문 다시 보기 귀찮아서 정리해놓는거 + 내용 이해했는지 셀프체크 느낌이라..  

야 모르겠다 아무튼 여기서 Attention 부분은 사실 자세히 안다뤘는데, 논문 읽기 초기에 트랜스포머 공부하다가 머리 터질 뻔했던게 생각난다.  

그래도 개미 똥꾸멍만큼의 짬밥이 생긴 지금 다시 트랜스포머 읽어보면서 글 작성하도록 하겠다...  

지인짜 마지막으로 구현은 끝났는데, 학습 돌릴 타이밍이 거시기해서 아주 어어어언젠가는 여기 추가하도록 하겠다...
