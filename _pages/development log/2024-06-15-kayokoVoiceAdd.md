---
title: "Kayoko AI with ChatGpt (2)"
tags:
    - project
date: "2024-06-15"

---

# 목표
---
카요코 보이스 기능 추가하기.
질문에 대한 답변의 텍스트를 딥러닝으로 학습한 카요코의 목소리가 출력될 것임.

# 구현 시작
---
## 준비
---
먼저 학습에 사용하기 위한 오디오 파일들을 준비할 것임.
![FIleList](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132298&authkey=%21AFSvyHfCw02rD7o&width=998&height=1190)

데이터 처리하기 편하도록 이름을 voiceFile로 통합하여 저장하였다.

이제 학습에 사용할 데이터파일을 만들기 위해 ChatGpt한테 부탁하여 다음과 같은 코드를 만들었다.
![DataCreate](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132301&authkey=%21ALzFoj63aALz9uQ&width=823&height=889)

이 코드를 실행하여
![DataResult](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132300&authkey=%21ABLG5D9E81ucro8&width=1058&height=931)
다음과 같은 텍스트 파일을 생성할 수 있었다.


## 모델 학습
---
엔비디아의 **tacotron2 모델**을 사용하여 학습을 진행하기로 했다.  
[tacotron2 링크](https://github.com/NVIDIA/tacotron2)

파일을 다운받고, tacotron2는 기본적으로 학습하는데 영어를 사용하기 때문에, japanese cleaner 따로 추가하여 코드를 수정하였다.  

![Deeplearning start](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132299&authkey=%21ANkHTROxrEJi_TQ&width=2552&height=1389)
코드 수정 완료 후, 성공적으로 학습을 시작할 수 있게 되었다.  
이 과정에서 수많은 오류를 직면하였다.  
라이브러리의 버전이 바뀌면서 함수가 사라지거나 하는 등의 이유로 나는 6시간을 고군분투했다.  
chatGpt의 도움이 없었다면 큰일날 뻔 했다.

![Deeplearning middle process](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132297&authkey=%21ALRVdngSEbQipKE&width=1890&height=978)
샤워하고 오니 대략 이만큼 진행되었다.  
보니까 컴퓨터 GPU 사용률이 90~100 % 사이를 왔다갔다하는데 글카의 비명소리가 들리는 것 같다.  
체크포인트를 따로 설정하지 않아서 이게 잘 되고 있는지를 모르겠다...

학습된 모델을 가지고 검증을 시도했으나.. 결과는 처참했다. 그래서 학습할 데이터의 양을 더 늘리기로 했다.  
카요코라는 캐릭터의 음성파일을 더 늘리기 위해  구글링 결과 카요코 asmr 1시간 짜리가 있었다.  
이를 다운받고 음성 구간을 나누어 다수의 음성파일로 만들 것이다. 

파이썬의 AudioSegment를 이용해서 음성 부분만 추출했는데, 결과가 만족스럽지 않아서 adobe media encoder를 통해서 asmr 음성 파일을 여러 구간으로 나눠준다. (손으로 직접)  
추가한 wav 파일들을 학습에 사용할 txt 파일에 추가해준다..
![updated List](https://onedrive.live.com/embed?resid=9EB8D569512A5F6B%2132307&authkey=%21AEHsvWTL9EcztM8&width=678&height=622)
총 486개의 5~10초 길이의 음성파일이 준비되었다. 

다시 학습을 진행해보자..

.
.
.
.
.

# 2024-06-18
---
며칠간을 씨름하였으나 tacotron2 모델은 오래되기도 해서, 방법을 바꾸기로 했다.  
https://github.com/coqui-ai/TTS
바로 coqui TTS의 기존에 학습된 모델에 파인 튜닝을 통해서 카요코의 목소리로 나타내기로 했다.  
내일 다시 오도록 하겠다...

# 2024-06-18
---
coqui 의 kokoro 라는 모델을 통해서 파인 튜닝을 진행하고 있다.  
시작한지 10분 채 되지 않았는데, 다음과 같은 결과가 나오고 있다.

<audio controls>
  <source src="https://github.com/mine3873/mine3873.github.io/blob/master/assets/wav/testAudio.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>