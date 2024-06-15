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
![FIleList](https://drive.google.com/uc?export=view&id=13VhJhSwLltaU1pIH-W4Tj4Y-RXHd-QlW)

데이터 처리하기 편하도록 이름을 voiceFile로 통합하여 저장하였다.

이제 학습에 사용할 데이터파일을 만들기 위해 ChatGpt한테 부탁하여 다음과 같은 코드를 만들었다.
![DataCreate](https://drive.google.com/uc?export=view&id=1MhwXG2eIP-OYSRr0UVK9fCJmp050xs79)

이 코드를 실행하여
![DataResult](https://drive.google.com/uc?export=view&id=18HlnEXHSvFRx8MYxPRroa4t3ZCj-RZ-K)
다음과 같은 텍스트 파일을 생성할 수 있었다.


## 모델 학습
---
엔비디아의 tacotron2 모델을 사용하여 학습을 진행하기로 했다.  
[tacotron2](https://github.com/NVIDIA/tacotron2)

파일을 다운받고, tacotron2는 기본적으로 학습하는데 영어를 사용하기 때문에, japanese cleaner 따로 추가하여 코드를 수정하였다.  

![Deeplearning start](https://drive.google.com/uc?export=view&id=1UnlO1AWrERg8OtIAxzvZfcoiAH5xgXqB)
코드 수정 완료 후, 성공적으로 학습을 시작할 수 있게 되었다.  
이 과정에서 수많은 오류를 직면하였다.  
라이브러리의 버전이 바뀌면서 함수가 사라지거나 하는 등의 이유로 나는 6시간을 고군분투했다.  
chatGpt의 도움이 없었다면 큰일날 뻔 했다.

![Deeplearning middle process](https://drive.google.com/uc?export=view&id=1frABSu2ROZfOaZIqfoPGOQQ5MtIVQ9y9)
샤워하고 오니 대략 이만큼 진행되었다.  
보니까 컴퓨터 GPU 사용률이 90~100 % 사이를 왔다갔다하는데 글카의 비명소리가 들리는 것 같다.

