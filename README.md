# 🎮 PickGame - AI 게임 추천 서비스

BERT 기반 자연어처리를 활용한 게임 추천 웹서비스

## 📋 서비스 소개

사용자의 자연어 질문을 이해하고, 훈련된 BERT 모델을 통해 개인화된 게임을 추천하는 AI 서비스입니다.

- 자연어 질문 처리 ("협동 게임 추천해줘", "RPG 게임 뭐가 좋을까?" 등)
- BERT 기반 게임 추천 (KcBERT 모델 파인튜닝)
- 웹 인터페이스 및 API 서비스 제공

## 📁 프로젝트 구조

```
PickGame/
├── app.py                 # Flask 웹서버
├── predict.py             # BERT 모델 추론 엔진
├── train_model.py         # 모델 훈련 스크립트
├── game_qa_data.csv       # 게임 Q&A 데이터셋
├── requirements.txt       # 패키지 의존성
├── templates/
│   └── index.html        # 메인 웹 인터페이스
└── model/                # 훈련된 모델 저장소
```

## 🚀 실행 방법

```bash
# 패키지 설치
pip install -r requirements.txt

# 웹서비스 시작
python app.py
```

접속: `http://localhost:5000`

## ✨ 주요 기능

### 웹 인터페이스
- PickGame 브랜딩 및 반응형 UI
- 실시간 AI 게임 추천
- 직관적인 질문-답변 인터페이스

### AI 추론 엔진
- BERT QA 모델 기반 질문-답변 매칭
- 한국어 자연어처리 특화 (KcBERT)
- 실시간 추론 처리

### API 서비스
- `POST /api/predict` - 게임 추천 API
- `GET /health` - 서비스 상태 확인
- `GET /model-status` - 모델 상태 확인

## 🎯 사용 예시

**질문:** "협동 게임 추천해줘"  
**답변:** "It Takes Two는 두 명이 협력해서 퍼즐을 풀어나가는 협동 어드벤처 게임입니다."

**질문:** "혼자 할 수 있는 RPG 게임 뭐가 좋을까?"  
**답변:** "엘든 링은 자유도가 높은 오픈월드 RPG로 광활한 세계를 탐험할 수 있습니다."

## 🔧 기술 스택

- **AI/ML**: PyTorch, Transformers, KcBERT
- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript

