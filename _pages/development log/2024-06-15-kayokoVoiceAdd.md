---
title: "Kayoko AI with ChatGpt (2)"
tags:
    - project
date: "2024-06-15"

---

# 목표
카요코 보이스 기능 추가하기.
질문에 대한 답변의 텍스트를 딥러닝으로 학습한 카요코의 목소리가 출력될 것임.

# 구현 시작
## 준비
먼저 학습에 사용하기 위한 오디오 파일들을 준비할 것임.
![FIleList](https://drive.google.com/file/d/13VhJhSwLltaU1pIH-W4Tj4Y-RXHd-QlW/view?usp=sharing)

데이터 처리하기 편하도록 이름을 voiceFile로 통합하여 저장하였다.

이제 학습에 사용할 데이터파일을 만들기 위해 ChatGpt한테 부탁하여 다음과 같은 코드를 만들었다.
![DataCreate](https://drive.google.com/file/d/1MhwXG2eIP-OYSRr0UVK9fCJmp050xs79/view?usp=sharing)
이 코드를 실행하여
![DataResult](https://drive.google.com/file/d/18HlnEXHSvFRx8MYxPRroa4t3ZCj-RZ-K/view?usp=sharing)
다음과 같은 텍스트 파일을 생성할 수 있었다.
