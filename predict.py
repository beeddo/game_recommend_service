# predict.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering
import os
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ratsnlp import nlpbook
from ratsnlp.nlpbook.qa import QADeployArguments

class GameRecommendationPredictor:
    def __init__(self, model_path="./model", qa_data_path="augmented_game_qa_data.csv"):
        self.model_path = model_path
        
        # ratsnlp 설정
        self.args = QADeployArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_model_dir=model_path,
            max_seq_length=512,
            max_query_length=64
        )
        
        self.model_loaded = False
        
        # QA 데이터 로드 및 전처리
        try:
            self.qa_data = pd.read_csv(qa_data_path)
            print(f"✅ QA 데이터 로드 완료: {len(self.qa_data)}개 항목")
            
            # TF-IDF 벡터화를 위한 준비
            self.vectorizer = TfidfVectorizer()
            self.qa_vectors = self.vectorizer.fit_transform(self.qa_data['question'].fillna(''))
            print("✅ TF-IDF 벡터화 완료")
            
        except FileNotFoundError:
            print(f"⚠️ QA 데이터 파일을 찾을 수 없습니다: {qa_data_path}")
            raise FileNotFoundError(f"훈련용 데이터 파일이 필요합니다: {qa_data_path}")
        
        # 훈련된 모델 로드 (필수)
        self.load_trained_model()
        
        # GPU 사용 가능 시
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_loaded:
            self.model.to(self.device)
            self.model.eval()
            print(f"🚀 모델이 {self.device}에서 실행됩니다")
    
    def load_trained_model(self):
        """훈련된 모델을 로드 (필수 요구사항)"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"❌ 훈련된 모델 폴더가 없습니다: {self.model_path}\n"
                f"💡 먼저 다음을 실행하여 모델을 훈련해주세요:\n"
                f"   python train_model.py"
            )
        
        try:
            print("🤖 훈련된 BERT 모델을 로드합니다...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForQuestionAnswering.from_pretrained(self.model_path)
            self.model_loaded = True
            print("✅ 훈련된 모델 로드 완료!")
        except Exception as e:
            raise RuntimeError(
                f"❌ 모델 로드 중 오류 발생: {str(e)}\n"
                f"💡 모델 파일이 손상되었을 수 있습니다. 다시 훈련해주세요:\n"
                f"   python train_model.py"
            )
    
    def find_best_context(self, question: str) -> str:
        """질문과 가장 유사한 컨텍스트를 찾아 반환"""
        # 질문을 TF-IDF 벡터로 변환
        query_vector = self.vectorizer.transform([question])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.qa_vectors).flatten()
        
        # 가장 유사한 상위 5개 답변 선택
        top_indices = similarities.argsort()[-5:][::-1]
        top_answers = self.qa_data.iloc[top_indices]['answer'].tolist()
        
        # 선택된 답변들을 하나의 컨텍스트로 결합
        return " ".join(top_answers)
    
    def format_answer(self, answer: str) -> str:
        """답변을 보기 좋게 포맷팅"""
        # 문장 단위로 분리
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        # 포맷팅된 답변 생성
        formatted = "🎮 게임 추천 결과:\n\n"
        
        # 각 문장을 불릿 포인트로 표시하고, 긴 문장은 여러 줄로 나누기
        for i, sentence in enumerate(sentences, 1):
            # 문장이 너무 길면 적절히 나누기
            if len(sentence) > 100:
                words = sentence.split()
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > 100:
                        formatted += f"• {' '.join(current_line)}\n"
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    formatted += f"• {' '.join(current_line)}\n"
            else:
                formatted += f"• {sentence}.\n"
        
        # 추가 정보
        formatted += "\n💡 더 자세한 정보가 필요하시다면 구체적으로 질문해주세요!"
        
        return formatted

    def inference_fn(self, question: str, context: str = None) -> str:
        """훈련된 BERT 모델을 이용한 게임 추천 추론"""
        if not question or not question.strip():
            return "질문을 입력해주세요."
        
        if not self.model_loaded:
            raise RuntimeError("❌ 훈련된 모델이 로드되지 않았습니다.")
            
        # 컨텍스트가 제공되지 않으면 자동으로 찾기
        if context is None:
            context = self.find_best_context(question)
        
        try:
            # 질문 토큰화 및 자르기
            truncated_query = self.tokenizer.encode(
                question,
                add_special_tokens=False,
                truncation=True,
                max_length=self.args.max_query_length
            )
            
            # 입력 인코딩
            inputs = self.tokenizer.encode_plus(
                text=truncated_query,
                text_pair=context,
                truncation="only_second",
                padding="max_length",
                max_length=self.args.max_seq_length,
                return_token_type_ids=True,
            )
            
            # 텐서로 변환 및 디바이스로 이동
            model_inputs = {k: torch.tensor([v]).to(self.device) for k, v in inputs.items()}
            
            # BERT 모델 추론 실행
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                start_pred = outputs.start_logits.argmax(dim=-1).item()
                end_pred = outputs.end_logits.argmax(dim=-1).item()
                
                # 답변 추출
                if start_pred <= end_pred and end_pred < len(inputs['input_ids']) and start_pred >= 0:
                    pred_text = self.tokenizer.decode(
                        inputs['input_ids'][start_pred:end_pred+1],
                        skip_special_tokens=True
                    ).strip()
                    
                    if pred_text and len(pred_text) > 2:
                        return self.format_answer(pred_text)
                    else:
                        return "죄송합니다. 적절한 답변을 찾을 수 없습니다. 다른 방식으로 질문해보세요."
                else:
                    return "죄송합니다. 적절한 답변을 찾을 수 없습니다. 다른 방식으로 질문해보세요."
                
        except Exception as e:
            raise RuntimeError(f"❌ 모델 추론 중 오류 발생: {str(e)}")

# 전역 예측기 인스턴스 (싱글톤 패턴)
_predictor = None

def get_answer(question: str) -> str:
    """Flask 앱에서 사용할 인터페이스 - 훈련된 모델만 사용"""
    global _predictor
    
    try:
        if _predictor is None:
            _predictor = GameRecommendationPredictor()
        
        return _predictor.inference_fn(question)
    
    except (FileNotFoundError, RuntimeError) as e:
        error_msg = str(e)
        print(error_msg)
        return f"❌ 모델 오류: 훈련된 모델이 필요합니다. 먼저 'python train_model.py'를 실행해주세요."
    
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        return "❌ 시스템 오류가 발생했습니다. 관리자에게 문의하세요."
