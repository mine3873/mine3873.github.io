---
title: "Attention Is All You Need"
date: 2026-06-23 12:34:00 +0900
categories: [DL]
tags: [dl] 
math: true
---

# Attention Is All You Need

> [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
{: .prompt-info} 

## Introduction 

이 논문은 당시에 사용하던 RNN, CNN 기반의 모델이 아닌 새로운 seq2seq 모델의 아키텍처를 제시한다. 

여기서 seq2seq 모델은 시퀀스를 입력으로 받고 시퀀스를 출력하는 모델로, 이 논문에서 그랬듯이, 기계 번역 (한국어 -> 영어 등) 을 예시로 두고 가보자.

이 논문에서는 당시에 당시에 RNN 에 보조적으로만 사용되던 어텐션 메커니즘만을 사용한, 새로운 seq2seq 구조, 트랜스포머를 제시한다.  

![](https://i.imgur.com/Il1K5ph.png) 

일단 전체 구조는 위와 같은데, 일단 알아보자.   

## Previous seq2seq models

일단 이전 seq2seq 모델들이 뭐가 있는지 알아보자.. 딱히 안봐도 되긴하는데.. 그냥 트랜스포머 이전에 모델들이 간단하게 어떤 식인지, 어떤 문제가 있는지 알아보자..  

### RNNs seq2seq

일단 기존 seq2seq RNN 모델은 $t$ 시점의 $h_{t}$ 를 계산하기 위해, 이전 시점의 $h_{t-1}$ 과 $t$ 시점의 입력 $x_{t}$ 를 필요로 한다. 그래서 이 $h_{t}$ 를 사용하여 $y_{t}$ 를 생성한다. 

이를 $f_{\theta}: \mathbb{R}^{TD} \to \mathbb{R}^{T'C}$ 로 나타낼 수 있는데, 여기서 $D,C$ 는 각각 입력, 출력 벡터의 크기, $T,T'$ 는 각각 입력, 출력 시퀀스의 길이이다. 

여기서 두가지 경우로 나눌 수 있는데, $T = T'$, $T \neq T'$ 두 경우이다. 

먼저 두 시퀀스 길이가 같은 경우, 다음과 같다.  

![](https://i.imgur.com/axKwUgG.png)

$$
p(y_{1:T}|x_{1:T}) = \sum_{h_{1:T}}\prod_{t=1}^{T}p(y_{t}|h_{t})\mathbb{I}(h_{t} = f(h_{t-1},x_{t}))
$$

이때 $h_{1} = f(h_{0}, x_{1}) = f_{0}(x_{1})$ 로 정의된다.  

이 경우, $y_{1}, y_{2}, ...$ 이렇게 순차적으로 계산하고, 각 $y_{t}$ 는 $h_{t}$ 에만 의존하며, 각 $h_{t}$ 는 $h_{t-1}, x_{t}$ 에 의존하는 형태이다. 그렇기 때문에 병렬 연산이 불가능하고, 레이어를 순차적으로 거치며 생기는 기울기 폭발 또는 소실로, 문장 길이가 길어질 수록 앞에 나온 단어의 정보를 사용하기 위한 장기 의존성 능력이 이빠이 떨어진다.  

장기 의존성 능력이 떨어진다는 것은, 문장이 길어질 수록 뒤에 생성되는 단어들이 문장의 앞쪽 단어들을 확인하지 않는다는 것인데, `도로롱은 세계 최고의 귀염둥이 캐릭터인데 귀엽기만한게 아니라 맛있어보이기도 하는데 아니 짜장면 먹고싶다` 이렇게 문장이 길어지면서 뒤에 생성되는 단어들이 앞쪽의 `도로롱`이라는 정보를 참고하지 않게 되는 것이다.  

아무튼 $T \neq T'$ 경우는, 예를 들어서 `도로롱은 귀엽다` 와 `DORO is cute` 처럼, 한국어에서 영어로 번역할 때, 시퀀스 길이가 다른 경우를 말한다. 

![](https://i.imgur.com/nnuEaJq.png)

이 경우, 사진과 같이 Encoder-Decoder 구조를 사용하여, 입력 시퀀스의 전체적인 맥락 정보를 $c = h_{T}^{e}$ 와 같이 추출하고, 이 정보를 디코더에서 참고하여  $h_{t}^{d} = f_{d}(h_{t-1}^{d}, y_{t-1}, c)$ 로 출력 시퀀스를 출력한다.  

하지만 이렇게 하더라도 출력 시퀀스가 직접적으로 입력 시퀀스에 연결되지 않기에 병목 현상이 생긴다. 이 문맥 정보 $c$ 가 고정된 크기의 벡터여서, 문장의 길이가 어떻든 이를 같은 크기의 벡터에 정보를 우겨넣어야 해서, 정보 손실이 생긴다는 것이다.   

때문에 직접적으로 입력 시퀀스에 추가로 연결하여 이를 어느정도 해결하려 할 수 있지만, 그렇더라도 각 언어별로 문장의 주어, 동사 등의 서순이 달라, 이 서순, 즉, 단어의 순서를 추론할 수밖에 없다.  

### 1d CNNs seq2seq

1d CNNs seq2seq 모델은 다음과 같이 Causal CNN 을 사용하여 Markov 형태로 정의된다. 일반적인 CNN은 주변의 앞뒤 데이터를 같이 보는데, 이전 단어들만 보도록 하기 위해 Causal CNN 을 사용하여 미래 단어가 필터에 걸리지 않도록 차단한다.  

$$
p(y) = \prod_{t=1}^{T}p(y_{t}|y_{1:t-1}) = \prod_{t=1}^{T}\text{Cat}(y_{t}|\text{softmax}(\varphi(\sum_{\tau = 1}^{t-k}w^{T}y_{\tau:\tau + k})))
$$

이때 $w$ 는 $k$ 사이즈의 컨볼루션 필터이고, $\varphi(.)$ 는 활성화 함수이다. 그리고 표기적 간편을 위해 범주형 출력으로 나타낸다.   

![](https://i.imgur.com/xCTwfOn.png)

전체적인 맥락 정보를 가지기 위해 위 사진과 같이 Dilated 컨볼루션을 사용하여, 문장이 길어지더라도, 마지막 단어가 앞쪽의 단어 정보를 가지며 출력될 수 있게끔한다.  

하지만 필터 크기와 레이어의 층수에 따라 참고할 수 있는 앞쪽 단어들의 정보가 정해지기 때문에, 장기 의존성을 키우기 위해서는 레이어 층수를 이빠이 높일 수밖에 없다.  

## Attention

앞서 뭐..이런저런 모델을 살펴봤는데, 어텐션이 뭐하는 놈이길래 트랜스포머에서 사용하는지, 알아보자.  

기존 네트워크에서 각 레이어는 다음과 같이 나타낼 수 있다. 

$$
Z = \varphi(X \cdot W) \quad (X \in \mathbb{R}^{m \times v}, W \in \mathbb{R}^{m \times v'})
$$

여기서 $X$ 는 해당 레이어의 입력 벡터, $W$ 는 가중치, $\varphi(.)$ 는 활성화 함수이다.  

여기서 가중치가 입력 벡터 $X$ 에 의존하는 형태로 다음과 같이 나타낼 수 있다.

$$
Z = \varphi(X \cdot W(X))
$$

더 일반화해서, 다음과 같은 형태로 나타낼 수 있다.  

$$
Z = \varphi(V \cdot W(Q,K))
$$

여기서 $Q \in \mathbb{R}^{m \times q}$ 는 Query, 현재 입력이 문맥 파악을 위해 현재 찾고 있는 값이고, $K \in \mathbb{R}^{m \times q}$ 는 Key, 데이터가 가진 압축된(?) 데이터, $V \in \mathbb{R}^{m \times v}$ 는 Value, 실제로 다음 레이어로 전달되는 핵심 정보이다.  

그러니까... 흔히 사용되는 유튜브 예시로는, $Q$ 는 검색어, $K$ 는 썸네일이나 태그 같은 압축된 정보, $V$ 는 실제 그 영샹의 데이터이다.  

그래서 어텐션은 위와 같은 형태를 가지며, $Q$ 와 $K$ 와의 유사도를 계산하고, 이 유사도를 바탕으로 $V$ 와 연산하여 출력한다.    

일단 간단하게 Single output vector 의 경우, $q,k,v$ 에 대해서, 어텐션은 다음과 같이 계산된다.  

$$
\begin{align}
    \text{Attn}(q,(k_{1:m}, v_{1:m})) &= \sum_{i=1}^{m} \alpha_{i}(q, k_{1:m})v_{i} \in \mathbb{R}^{v} \\
    \alpha_{i}(q, k_{1:m}) &= \frac{\exp(a(q, k_{i}))}{\sum_{j=1}^{m}\exp(a(q, k_{j}))}
\end{align}
$$

여기서 $a(q, k_{i}) \in \mathbb{R}$ 는 어텐션 점수로 $q,k$ 의 유사도를 계산한다. 그리고   $\alpha_{i}(q, k_{1:m})$ 를 어텐션 가중치라고 하는데, $\alpha_{i}(q, k_{1:m}) > 0$, $\sum_{i=1}^{m}\alpha_{i}(q, k_{1:m}) = 1$ 를 만족시키기 위해 Softmax 를 사용한다.  

그래서 최종적으로는 $Q$ 와 $K$ 의 유사도가 높을 수록 어텐션 가중치가 높아지며, 이 값이 각 $V$ 의 요소에 곱한 형태가 된다.  

아무튼 형태를 보면 Dictionary lookup 과 비슷한데, 여기서 특정 부분에만 어텐션을 수행하도록 제한할 수 있다. 이를 Masked attention 이라고 하며, 우리가 어텐션을 수행할 때 무시하고자 하는 부분의 어텐션 점수 $a(q, k_{i})$ 를 $-10^{6}$ 과 같은 매우 작은 값으로 설정하여, 해당 부분의 어텐션 가중치가 $0$ 에 가까워지게끔 하여 수행할 수 있다.  

---

아무튼 더 일반화해서, $q \in \mathbb{R}^{q}$, $k \in \mathbb{R}^{k}$ 에 대해서, $W_{q} \in \mathbb{R}^{h \times q}$, $W_{k} \in \mathbb{R}^{h \times k}$ 를 사용하여 같은 공간으로 투영한 뒤, 다음과 같이 어텐션 점수를 계산한다. 

$$
a(q,k) = w_{v}^{T}\text{tanh}(W_{q}q + W_{k}k) \in \mathbb{R}
$$

근데 여기서 $q, k \in \mathbb{R}^{d}$ 와 같이 같은 크기를 가진다면, 다음과 같이 간단하게 나타낼 수 있다. 

$$
a(q,k) = \frac{q^{T}k}{\sqrt{d}} \in \mathbb{R}
$$

여기서 $q^{T}k$ 의 분산을 $1$ 로 설정하기 위해, $\sqrt{d}$ 로 나누어 준다.  

아무튼, 어텐션은 $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{m \times d}$, $V \in \mathbb{R}^{m \times v}$ 에 대해 다음과 같이 계산된다.   

![](https://i.imgur.com/Mpm8j3U.png)

$$
\therefore \text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d}}\right)V \in \mathbb{R}^{n \times v}
$$

### the alignment inference with attention in RNNs

앞선 RNNs seq2seq 에서의 각 시퀀스의 서순을 추론할 수 밖에 없던 문제를 어텐션을 사용해 해결할 수 있다고 했는데, 이 고정된 문맥 정보 $c$ 를 다음과 같이 동적인 문맥 정보 $c_{t}$ 로 나타낼 수 있다.   

$$
c_{t} = \sum_{i=1}^{T} \alpha_{i}(h_{t-1}^{d}, h_{1:T}^{e})h_{i}^{e}
$$

여기서 Query는 디코더의 이전 단계의 은닉 상태 $h_{t-1}^{d}$, Key는 인코더의 모든 은닉상태, Value는 마찬가지로 인코더의 모든 은닉상태이다.  

![](https://i.imgur.com/cl9KOXR.png)

그래서 전체구조는 위와 같은데, 뭔가 많이 돌아간거 같으니.. 일단 넘어가자.. 아무튼 트랜스포머는 이렇게 보조적으로만 사용되던 어텐션을 중점적으로 사용하고자 한 것이다...  

### Self-attention

앞서 RNNs seq2seq 에서 어텐션을 이용하여 인코더의 문맥 정보를 디코더에 사용하였는데, 같은 문장 내에서 단어들끼리의 문맥을 파악하고자 할 수 있다.  

예를 들어서, 다음 두 문장 예시를 보자. 

`The animal didn't cross the street because it was too tired.`  
`The animal didn't cross the street because it was too wide.`

이 경우, 두 문장에서의 `it` 이 가리키는 대상이 다르다. 전자는 `The animal`, 후자는 `the street` 을 지칭한다.  

이를 위해 사용되는 것이, Self-attention, 같은 문장 내에서 단어들끼리의 문맥 정보를 파악하는 것이다. 

입력 시퀀스 $x_{1:n} \in \mathbb{R}^{d}$ 가 주어졌을 때, 다음과 같다.  

$$
y_{i} = \text{Attn}(x_{i}, (x_{1:n}, x_{1:n})) \in \mathbb{R}^{d}
$$

![](https://i.imgur.com/BF5AqqT.png)

위 사진은 앞선 두 예시 문장에서 Self-attention 을 수행한 결과이다. 연결된 선이 진할 수록 높은 어텐션 가중치를 가지고 있다는 것인데, `it` 이 무엇을 나타내는지 쌈뽕하게 계산함을 볼 수 있다. 

### Multi-Head Attention, MHA

$Q,K$ 의 유사도를 계산하는 어텐션 점수에 대해서, 이 유사도의 기준에 따라 두 데이터간의 유사도가 달라질 수 있다.  

예를 들어, 사과와 바나나는 둘 다 같은 과일이라는 점에서 높은 유사도를 가지지만, 색깔을 기준으로 보면 전혀 다른 색으로 낮은 유사도를 가진다.   

이를 위해서, 서로 다른 기준의 유사도를 측정하는 여러 개의 어텐션 점수를 구하고자 Multi-head attention, MHA 를 정의하는데, 다음과 같다.  

$Q,K,V \in \mathbb{R}^{B \times d}$ 가 주어질 때, 독립적인 가중치 $W_{i}^{Q} \in \mathbb{R}^{d \times d_{k}}$, $W_{i}^{K} \in \mathbb{R}^{d \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d \times d_{v}}$ 를 사용하여, 서로 다른 저차원 공간으로 투영시킨 뒤에 어텐션을 수행한다.   

$$
\begin{align}
    h_{i} &= \text{Attn}(Q_{i}W_{i}^{Q}, (K_{i}W_{i}^{K},V_{i}W_{i}^{V})) \\
    h &= \text{MHA}(Q,(K,V)) = \begin{pmatrix}
        h_{1} \\ \vdots \\ h_{h}
    \end{pmatrix}W^{O}
\end{align}
$$

$h$ 의 유사도 기준에서 어텐션을 수행한 뒤, $W^{O} \in \mathbb{R}^{hd_{v} \times d}$ 를 사용하여 다시 $d$ 차원으로 확장시킨다.

## Transformer Architecture

![](https://i.imgur.com/Il1K5ph.png) 

앞서 다 알아봤으니, 다시 모델 구조를 보자.  

### Encoder & Decoder 

Encoder-Decoder 구조로 나뉘고, Encoder는 총 $N$ 개의 블록으로 구성되며, 각 블록은 Multi-head self-attention 과 Feed Forward 레이어로 구성된다. 그리고 각 레이어에서 Residual connection 을 사용한다. (이때 이를 위해서 모든 레이어의 출력 차원을 같은 $d$ 로 설정한다. 논문에서는 $d=512$ 를 사용한다.) 

즉, 다음과 같다.    

``` python
def EncoderBlock(X):
    h = LayerNorm(MHA(Q=X, K=X, V=X) + X)
    h = LayerNorm(FF(h) + h)
    return h

def Encoder(X, N):
    h = X
    for n in range(N):
        h = EncoderBlock(h)

    return h
```

논문에선 위와 같이 Post-layerNorm 을 사용하지만, 최근엔 주로 Pre-layerNorm 이 사용되긴 한다.  

Decoder 는 Encoder 와 구조가 비슷하지만, 앞에 Masked MHA 가 추가되었다.  
당연히 문장을 출력할 때, 순차적으로 생성해야 하는데, 다시 설명하자면..예를 들어서 `도로롱은 도로시보다 귀엽다.` 라는 문장을 생성할 때, `귀엽다` 라는 단어는 앞선 `도로롱은`, `도로시보다` 를 참고하여 생성되지, 뒤에 이어질 미지의 단어를 참고하지 않는다.  

이를 위해서, 어텐션 점수 $QK^{T}$ 의 정사각 행렬이 주어질 때, 해당하는 현재 단어가 뒤에 이어질 단어들과의 가중치 점수를 $-10^{6}$ 과 같은 아주 작은 값으로 설정하여, Softmax 함수 후에, 나온 어텐션 가중치가 $0$ 에 가까워지도록 설정한다. 즉, 강제적으로 뒤에 나올 단어들과의 유사도를 없애버리는 것이다. 이를 위해서 가중치 점수 행렬의 Upper Triangle 부분의 모든 요소를 매우 적은 값으로 설정한다.  

물론 앞서 `도로롱은 도로시보다 귀엽다.` 라는 문장을 추론할 때는 `도로롱은`, `도로롱은 도로시보다`, `도로롱은 도로시보다 귀엽다.` 처럼 순차적으로 생성하지만, 학습 과정에서는 이 전체 문장을 알고 있으므로, 앞선 Upper Triangle Mask 만 적용한 채, 전체 시퀀스를 병렬로 연산한다.  

그리고 두번째 MHA 에 대해선, Encoder의 출력, 문맥 정보를 $K,V$ 값으로 입력을 받는다.  

...다음과 같다. 

``` python
def DecoderBlock(Y, E, mask):
    h = LayerNorm(MHA(Q=Y, K=Y, V=Y, mask=mask) + Y)
    h = LayerNorm(MHA(Q=h, K=E, V=E, mask=None) + h)
    h = LayerNorm(FF(h) + h)
    return h

def Decoder(Y, E, N, mask):
    h = Y
    for n in range(N):
        h = DecoderBlock(h, E, mask)
    return h
``` 

### Feed-Forward, FF

이 Fully connected Feed-Forward 레이어는 두개의 선형 변환과, 그 사이에 ReLU 활성화 함수가 있는 형태이다. 

$$
FF(x) = \max(0, xW_{1} + b_{1})W_{2} + b_{2}
$$

여기서 보통 차원을 확장시키고, 다시 원래 차원으로 축소시키는데, 이전 MHA 레이어에선 단어 사이의 관계를 파악했다면, 여기선 단어 자체의 표현력을 비선형적으로 확장하게끔 한다.  

논문에서는 입력 차원이 $d=512$ 라면, $d_{\text{in}} = 2048$ 로 확장한 뒤, 다시 $d=512$ 차원으로 축소시킨다.  

선택적으로 이 선형 변환을 $1 \times 1$ 의 커널을 가진 컨볼루션으로 대체할 수도 있다. 

### Positional Encoding

attention 은 시퀀스의 단어의 순서를 무시하기 때문에, 따로 조치가 필요한데, 모델 구조를 보면, Encoder, Decoder 모두 Positional Encoding 를 사용하여 각 단어가 나타나는 순서를 표현한다.  Diffusion 에서 봤던 $t$ 와 비슷하다.  

단순히 이 단어의 순서를 정수로 나타내기 위해 $000, 001, 010, ...$ 과 같은 2진법 비트로 나타낼 수 있지만, 이 경우, 가장 오른쪽 비트에 비해 왼쪽 비트의 변환 빈번도가 적기 때문에, 논문에서는 다음과 같은 Sinusoidal basis 를 사용하여 위치 행렬 $P \in \mathbb{R}^{n \times d}$ 를 나타낸다. 이때 $n$ 은 시퀀스 길이, $d$ 는 각 위치 정보를 나타낼 때 사용하는 벡터의 길이이다.     

$$
\begin{align}
    p_{i,2j} &= \sin \left(\frac{i}{C^{2j / d}}\right) \\
    p_{i, 2j+1} &= \cos \left(\frac{i}{C^{2j / d}}\right)
\end{align}
$$

이때 $C$ 는 시퀀스의 최대 길이로, 논문에서는 $C=10000$ 으로 설정한다. 

이를 사용하면, 두 가지 이점이 있는데, 하나는 임의의 길이 $T' \leq C$ 에 대해서, $C$ 보다 작기만 하면 따로 학습 없이 위치를 바로 계산할 수 있다는 점이고, 다른 하나는 위치간의 상대적 거리를 선형 변환으로 나타낼 수 있다는 것이다. 즉, 특정 위치 $t$ 와 상대적 거리 $k$ 에 대해서, 선형 변환 $f$ 를 통해 $p_{t + k} = f(p_{t}) = wp_{t}$ 로 나타낼 수 있는데, 다음과 같다.  

$$
\begin{align}
    \begin{pmatrix}
        \sin(w(t+k)) \\ \cos(w(t+k))
    \end{pmatrix} &= \begin{pmatrix}
        \sin(wt)\cos(wk) + \cos(wt)\sin(wk) \\ \cos(wt)\cos(wk) - \sin(wt)\sin(wk)
    \end{pmatrix} \\
    &= \begin{pmatrix}
        \cos(wk) & \sin(wk) \\ -\sin(wk) & \cos(wk)
    \end{pmatrix}\begin{pmatrix}
        \sin(wt) \\ \cos(wt)
    \end{pmatrix}
\end{align}
$$

아무튼 인코더, 디코더는 각각 이렇게 계산된 $P$ 를 기존 입력에 더하여 $(X+P)$ 를 입력으로 가진다.   

``` python
POS(X) = X + P

...
Encoder(POS(X), ...)
...
Decoder(POS(X), ...)
...
```

## Compare with others

$x_{1:n} \to y_{1:n}$ 에 대해서, 트랜스포머가 왜 쌈뽕한지, 비교해보자.

일단 각 모델들은 속도와 표현력, 이 두 개의 trade-off 를 가지는데, 표현력은 임의의 두 입력 사이의 최대 경로 길이로 표현될 수 있다.  

|         Layer type          |  Complexity  | Sequential ops. | Max. path length |
| :-------------------------: | :----------: | :-------------: | :--------------: |
|       Self-attention        | $O(n^{2}d)$  |     $O(1)$      |      $O(1)$      |
|          Recurrent          | $O(nd^{2})$  |     $O(n)$      |      $O(n)$      |
|        Convolutional        | $O(knd^{2})$ |     $O(1)$      |  $O(\log_{k}n)$  |
| Self-attention (restricted) |   $O(rnd)$   |     $O(1)$      |     $O(n/r)$     |

|                                                                  |                                                                  |                                                                  |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| ![](https://i.imgur.com/p1zr2ec.png){: height="800" width="400"} | ![](https://i.imgur.com/22PLNTr.png){: height="800" width="400"} | ![](https://i.imgur.com/b6KrVKu.png){: height="800" width="400"} |

이 두 입력 사이의 최대 경로 길이가 짧다는 것은, 장기 의존성 능력이 좋아진다는 것으로, 진짜진짜 다시 말하지만, `도로롱 너무 귀여워 최고야 진짜 세계 최고 바보 멍청이처럼 생겼는데 묘하게 귀엽고 보면 볼수록 새로워 짜릿해 늘 도파민을 선사해 ....(생략)...` 처럼 문장의 길이가 길어질 수록, 여전히 뒤에 생성된 단어들이 앞쪽 단어들의 정보를 가지고 있다는 것으로 생각할 수 있다. (사실 저 문장 쓰고 싶어서 다시 언급해봤다.) 

아무튼 Self-attention 의 최대 경로 길이가 사진에서 봤듯이 모든 부분에서 $O(1)$ 이 된다. 그에 반해 RNN 기반 모델은 $O(n)$ 의 최대 경로 길이를 가지고 1d CNN 은 커널사이즈 $k$ 에 대해, $O(\log_{k}n)$ 의 최대 경로 길이를 가진다.  

그리고 계산 복잡성의 측면에서, 시퀀스의 길이 $n$ 이 임베딩 차원 $d$ 보다 작은 경우에, RNN과 CNN 보다 더 적은 계산 복잡도를 가짐을 알 수 있다.  

여기서 더 나아가서, 어텐션을 수행할 때, 모든 단어들에 대해서 계산하는 것이 아닌, 현재 단어를 기준으로 양방향으로 $r$ 개의 단어들에 대해서만 어텐션을 수행하도록 수정하여, 이 계산 복잡도를 줄일 수 있다. 하지만 이 경우, 기존 최대 경로 길이가 $O(n/n) \to O(n/r)$ 로 증가하긴 한다.  

..정리하자면, 트랜스포머는 어텐션을 통해 $O(1)$ 의 경로로 문장 내 모든 단어 정보를 참고할 수 있고, 병렬연산으로 인한 속도와, RNNs 에서의 고정된 문맥 벡터로 인한 병목 현상을 해결했다. 거기다 시퀀스 길이에 따른 계산 복잡도도 야무지다.  


## 찌라시시

아무튼 트랜스포머를 살펴봤는데, 학습 과정은 저 멀리 보내버리고, 모델 구조를 중점적으로 살펴보았다..  

어.. 트랜스포머 구조만 설명만 하자니 뭔가 허전해서... 이것저것 생각나는대로 적었는데, 그래서 그런지 평소 보다 더 글이 난해한 것 같다. 그냥 이건 이거다 이건 이거고 이건 요거다 반복인데..  

..모르겠다. 그냥 넘어가자. 역시나 나중에 수정할 부분 보이면 알아서.. 그때 수정하겠다. 

글이 너무 딱딱한 것 같아서 이거라도 추가하겠다.. 

![](https://i.imgur.com/wyjE1qf.gif)



