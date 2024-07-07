---
title: "Kayoko AI with ChatGpt (2)"
tags:
    - project
    - kayoko
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

# 2024-06-20
---
다음 링크를 참고 해서, chatgpt의 도움을 받아 코드를 완성했다.
[참고 링크](https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/xtts_v2/train_gpt_xtts.py)

``` Python
import os
from pydub import AudioSegment
from trainer import Trainer, TrainerArgs
import sys

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# 현재 작업 디렉토리를 PYTHONPATH에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
tts_module_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(tts_module_path)

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

RUN_NAME = "GPT_XTTS_v2.0_Japanese_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True
BATCH_SIZE = 3
GRAD_ACUMM_STEPS = 84

config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="custom_japanese_dataset",
    path="C:/Users/aj200/Desktop/VS/Data/mine/MyTTSDataset/",
    meta_file_train="C:/Users/aj200/Desktop/VS/Data/mine/MyTTSDataset/metadata.csv",
    language="ja",
)

DATASETS_CONFIG_LIST = [config_dataset]

CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

SPEAKER_REFERENCE = [
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile0_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile1_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile22.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile512_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile82_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile56_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile97.wav"
]
LANGUAGE = config_dataset.language

# '。' 를 기준으로 텍스트를 나눠 문장의 길이를 줄임.
def split_text(text, delimiter='。'):
    sentences = text.split(delimiter)
    return [s + delimiter for s in sentences if s]

# 텍스트를 분할하는 기준에 맞춰 오디오도 같이 분할
def split_audio(audio_path, split_durations):
    audio = AudioSegment.from_wav(audio_path)
    segments = []
    start = 0
    for duration in split_durations:
        end = start + duration
        segments.append(audio[start:end])
        start = end
    return segments

# 텍스트와 오디오 파일을 분할하는 함수
def split_text_and_audio(sample, delimiter='。'):
    text = sample['text']
    audio_file = sample['audio_file']
    speaker_name = sample['speaker_name']
    language = sample['language']

    split_texts = split_text(text, delimiter)
    
    audio = AudioSegment.from_wav(audio_file)
    total_duration = len(audio)
    split_durations = [total_duration * len(t) / len(text) for t in split_texts]

    split_audios = split_audio(audio_file, split_durations)
    
    return [{'text': t, 'audio_file': audio_file.replace(".wav", f"_{i}.wav"), 'speaker_name': speaker_name, 'language': language} 
            for i, (t, a) in enumerate(zip(split_texts, split_audios))]

def main():
    # 모델 파라미터 설정
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6초
        min_conditioning_length=66150,  # 3초
        debug_loading_failures=False,
        max_wav_length=255995,  # 약 11.6초
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # 오디오 설정 정의
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # 학습 파라미터 설정
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=0,  # 멀티프로세싱 비활성화
        num_eval_loader_workers=0,  # 멀티프로세싱 비활성화
        eval_split_max_size=256,
        print_step=100,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "やあ、先生。今日はどうしたの？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "うーん... 音楽のCDを集めるのが好きだよ。特にヘビーメタルとかね。先生も好きなものはあるの？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "これくらいでいい？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "なんか用？まあ、問題があったら解決してあげるけど。",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "耳かきって？そういう趣味はないけど、なんでそんなこと聞くの？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    model = GPTTrainer.init_from_config(config)

    try:
        train_samples, eval_samples = load_tts_samples(config_dataset, eval_split = True, eval_split_max_size = config.eval_split_max_size, eval_split_size = config.eval_split_size,)

        if len(train_samples) == 0:
            raise ValueError("Training samples are empty.")
        if len(eval_samples) == 0:
            raise ValueError("Evaluation samples are empty.")

        # 학습 시작
        trainer = Trainer(
            TrainerArgs(
                restore_path = None,
                skip_train_epoch = False,
                start_with_eval = START_WITH_EVAL,
                grad_accum_steps = GRAD_ACUMM_STEPS,
            ),
            config,
            output_path=OUT_PATH,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        trainer.fit()
    
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()

```

다음 오디오는 모델 평가 과정에서 생성된 음성이다.  
<audio controls>
  <source src="https://github.com/mine3873/mine3873.github.io/raw/master/assets/wav/individualAudio.wav" type="audio/wav">
  Your browser does not support the audio element.  
</audio>    
<audio controls>
  <source src="https://github.com/mine3873/mine3873.github.io/raw/master/assets/wav/individualAudio_1.wav" type="audio/wav">
  Your browser does not support the audio element.  
</audio>


이제 학습한 모델을 기존 코드에 합쳐준다.. 
``` python
import sys
import atexit
from PyQt5.QtWidgets import QMainWindow, QApplication, QTextEdit, QLineEdit, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtCore import QTimer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.options import Options
import subprocess
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import os
import torch
import torchaudio
from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts
import sounddevice as sd


CONFIG_PATH = "TTS/voice/tts/custum_ja/config.json"
TOKENIZER_PATH = "TTS/voice/tts/XTTS_v2.0_original_model_files/vocab.json"
XTTS_CHECKPOINT = "TTS/voice/tts/custum_ja/checkpoint_9300.pth"
SPEAKER_REFERENCE = "TTS/voice/tts/custum_ja/voiceFile0.wav"

def play_kayoko_TTS(text):
    out = model.inference(
        text,
        "ja",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7, # Add custom parameters here
    )
    
    audio_data = torch.tensor(out["wav"]).numpy()
    
    volume_factor = 0.3  # For example, reduce volume to 50%
    audio_data = audio_data * volume_factor
    sd.play(audio_data, samplerate=24000)

# TTS 
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])



subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chromeCookie"')
option = Options()
option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)
    # 웹사이트 접속
driver.get('https://chatgpt.com/g/g-1aLqNB1zW-kayoko')



chat_button_xpath = '//button[@class="mb-1 me-1 flex h-8 w-8 items-center justify-center rounded-full bg-black text-white transition-colors hover:opacity-70 focus-visible:outline-none focus-visible:outline-black disabled:bg-[#D7D7D7] disabled:text-[#f4f4f4] disabled:hover:opacity-100 dark:bg-white dark:text-black dark:focus-visible:outline-white disabled:dark:bg-token-text-quaternary dark:disabled:text-token-main-surface-secondary"]'

def addOneLenDiv():
    lendiv = lendiv + 1

def input_Text(text):
    inputarea = driver.find_element(By.ID, "prompt-textarea")
    inputarea.send_keys(text)
    driver.find_element("xpath",chat_button_xpath).click()

def get_latest_answer():
    selector = ".markdown.prose.w-full.break-words.dark\:prose-invert.dark"
    current_count = len(driver.find_elements(By.CSS_SELECTOR, selector))
    
    WebDriverWait(driver, 20).until(
        lambda x: len(x.find_elements(By.CSS_SELECTOR, selector)) > current_count
    )
    #time.sleep(4)
    
    button = driver.find_element("xpath",chat_button_xpath)
    while(button.is_enabled()):
        pass
    latest_answer_div = driver.find_elements(By.CSS_SELECTOR, selector)[-1]
    p_text = latest_answer_div.find_elements(By.TAG_NAME, "p")
    return p_text
        
    

def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class ChatUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set the background image
        self.initUI()

    def initUI(self):
        # Window settings
        self.setGeometry(100, 100, 1280, 960)
        self.setWindowTitle('Chat UI Example')
        QFontDatabase.addApplicationFont('./GodoB.ttf')
        
        # Set the background
        self.backgorund_label = QLabel(self)
        background_image_path = resource_path('img/kayoko.jpg')
        self.backgorund_label.setPixmap(QPixmap(background_image_path))
        self.backgorund_label.setScaledContents(True)
        self.backgorund_label.setFixedSize(1280,960)
        
        # Create the name area
        self.nameArea = QLabel("카요코",self)
        self.nameArea.setFixedSize(1000,70)
        self.nameArea.setStyleSheet(
            "background-color: rgba(255,255,255,0);"
            "border-width: 0px 0px 3px 0px;"
            "border-style: solid;"
            "border-color: rgba(138,138,138,200);"
            "color: white;"
        )
        self.nameArea.setFont(QFont('고도 B',40))
        self.nameArea.move(140,575)
        
        # Create the explaination area
        self.explain = QTextEdit(self)
        self.explain.setReadOnly(True)
        self.explain.append("흥신소 68")
        self.explain.setFixedSize(500,60)
        self.explain.move(320,590)
        self.explain.setStyleSheet(
            "background-color: rgba(255,255,255,0);"
            "border-style: solid;"
            "border-color: rgba(255,255,255,0);"
            "color: rgba(74,162,188,200);"
        )
        self.explain.setFont(QFont('고도 B',25))
        
        # Create the answer area
        self.answerArea = QTextEdit(self)
        self.answerArea.setReadOnly(True)
        
        self.answerArea.setLineWrapMode(1)
        self.answerArea.setLineWrapColumnOrWidth(1)
        
        self.answerArea.setStyleSheet(  
            "background-color: rgba(255,255,255,0);"
            "border-color: rgba(255,255,255,0);"
            "border-style: solid;"
            "font-weight: bold;"
            "color: white;"
                                      )
        self.answerArea.setFixedSize(1000,220)
        self.answerArea.move(140,660)
        self.answerArea.setFont(QFont('고도 B',25))
        
        # Create the chat display area
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background-color: qLineargradient(y1: 0, y2: 1, stop: 0 rgba(20, 27, 59, 0), stop: 0.1 rgba(20, 27, 59, 160));"
            "border-color: rgba(255,255,255,0);"
            "border-style: solid;"
            "font-weight: bold;"
            "font-size: 20px;"
            "color: rgba(255,255,255,200);"
        )
        self.chat_display.setFixedSize(1280,450)
        self.chat_display.move(0,510)
        
        
        
        
        # Create the input text box
        self.text_input = QLineEdit(self)
        self.text_input.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200);"
            "border-color: rgba(255,255,255,0);"
            "border-radius: 20px;"
        ) 
        self.text_input.returnPressed.connect(self.on_enter)
        self.text_input.setFixedSize(1000,60)
        self.text_input.move(140,880)
        self.text_input.setFont(QFont('고도 B', 25))
        self.text_input.setTextMargins(20,0,0,0)
        
        layout = QVBoxLayout()
        layout.addWidget(self.nameArea)
        layout.addWidget(self.text_input)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.answerArea)
        layout.addWidget(self.explain)
        
        self.chat_display.lower()
        self.backgorund_label.lower()
        
        self.setLayout(layout)
        self.show()

    
    def show_Answer(self):
        answer_arr = get_latest_answer()
        answer = answer_arr[0].text
        ja_answer = answer_arr[1].text
        play_kayoko_TTS(ja_answer)
        #answer = get_latest_answer()
        self.answerArea.setText('')
        self.text = answer
        self.idx = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.typeLetter)
        self.timer.start(80)
        
        self.text_input.clear()
        
    def typeLetter(self):
        if self.idx < len(self.text):
            self.answerArea.insertPlainText(self.text[self.idx])
            self.idx += 1
        else:
            self.timer.stop()
        
    def on_enter(self):
        # Get text from input box and display it in the chat display area
        input_textAPP = self.text_input.text()
        input_Text(input_textAPP)
        self.show_Answer()

  # 상태 확인 간격
    def closeEvent(self, event):
        # 윈도우가 닫힐 때 웹드라이버 종료
        driver.quit()
        event.accept()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatUI()
    sys.exit(app.exec_())
    
# Register the driver.quit function to be called on program exit
atexit.register(driver.quit)
```
생각보다 음성의 품질이 좋지 않아서 만족스럽지가 않다.  
따로 방법을 찾아야겠다..
