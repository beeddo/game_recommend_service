import os
import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from functools import lru_cache
import hashlib

app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 로드
df = pd.read_csv('game_qa_data.csv')

# 모델과 토크나이저 로드 (임베딩용)
model_name = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 임베딩 캐시를 위한 해시 함수
def get_text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# 캐시된 임베딩 저장소
embedding_cache = {}

def get_embedding_cached(text):
    """캐시를 사용한 임베딩 계산"""
    text_hash = get_text_hash(text)
    
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    
    # 캐시에 없으면 새로 계산
    embedding = get_embedding(text)
    
    # 캐시 크기 제한 (최대 1000개)
    if len(embedding_cache) > 1000:
        # 가장 오래된 항목 제거
        oldest_key = next(iter(embedding_cache))
        del embedding_cache[oldest_key]
    
    embedding_cache[text_hash] = embedding
    return embedding

def get_embedding(text):
    """텍스트의 BERT 임베딩을 구합니다."""
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] 토큰의 임베딩을 사용
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding

def get_embedding_batch(texts, batch_size=32):
    """배치 단위로 임베딩을 계산합니다 (성능 최적화)"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 배치 토크나이징
        inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, 
                          truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] 토큰의 임베딩을 사용
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def extract_keywords(text):
    """질문에서 주요 키워드를 추출합니다."""
    text_lower = text.lower()
    
    # 멀티플레이어 관련 키워드
    multiplayer_keywords = ['멀티', '친구', '함께', '협동', '같이', '팀', '둘이', '여럿', '다같이', '협력']
    
    # 장르 관련 키워드
    genre_keywords = {
        'horror': ['공포', '호러', '무서운', '스릴', '긴장'],
        'survival': ['생존', '서바이벌', '살아남', '극한'],
        'action': ['액션', '전투', '싸움', '격투'],
        'rpg': ['rpg', '롤플레잉', '캐릭터', '레벨업', '스킬'],
        'strategy': ['전략', '전술', '계획', '지휘'],
        'puzzle': ['퍼즐', '수수께끼', '문제', '논리'],
        'simulation': ['시뮬레이션', '시뮬', '경영', '건설', '농장'],
        'adventure': ['어드벤처', '모험', '탐험', '여행'],
        'racing': ['레이싱', '경주', '드라이빙', '자동차'],
        'sports': ['스포츠', '축구', '농구', '야구', '운동']
    }
    
    # 플랫폼 관련 키워드
    platform_keywords = {
        'pc': ['pc', '컴퓨터', '윈도우'],
        'console': ['ps', 'xbox', 'switch', '콘솔', '플스', '엑박'],
        'mobile': ['모바일', '핸드폰', '스마트폰', '안드로이드', 'ios']
    }
    
    # 난이도 관련 키워드
    difficulty_keywords = {
        'easy': ['쉬운', '간단한', '초보', '캐주얼'],
        'hard': ['어려운', '하드코어', '도전적인', '극한', '고수']
    }
    
    # 플레이 스타일 키워드
    playstyle_keywords = {
        'solo': ['혼자', '솔로', '싱글', '개인'],
        'competitive': ['경쟁', '대전', 'pvp', '랭킹'],
        'casual': ['캐주얼', '편안한', '힐링', '여유'],
        'hardcore': ['하드코어', '진지한', '몰입']
    }
    
    found_keywords = {
        'multiplayer': [kw for kw in multiplayer_keywords if kw in text_lower],
        'genre': {},
        'platform': {},
        'difficulty': {},
        'playstyle': {}
    }
    
    # 장르 키워드 검사
    for genre, keywords in genre_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            found_keywords['genre'][genre] = found
    
    # 플랫폼 키워드 검사
    for platform, keywords in platform_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            found_keywords['platform'][platform] = found
    
    # 난이도 키워드 검사
    for diff, keywords in difficulty_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            found_keywords['difficulty'][diff] = found
    
    # 플레이 스타일 키워드 검사
    for style, keywords in playstyle_keywords.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            found_keywords['playstyle'][style] = found
    
    return found_keywords

def filter_candidates_by_keywords(user_question, df):
    """키워드 기반으로 후보를 필터링합니다."""
    user_keywords = extract_keywords(user_question)
    original_size = len(df)
    filtered_df = df.copy()
    
    # 1. 멀티플레이어 vs 솔로 플레이 필터링 (상호 배타적)
    if user_keywords['multiplayer']:
        print(f"멀티플레이어 키워드 감지: {user_keywords['multiplayer']}")
        multiplayer_mask = df['question'].str.contains('|'.join(['멀티', '친구', '함께', '협동', '같이', '팀', '둘이']), na=False)
        multiplayer_answer_mask = df['answer'].str.contains('|'.join(['멀티', '협동', '팀', '친구', '함께', '협력', 'co-op', 'Co-op']), na=False)
        negative_mask = ~df['answer'].str.contains('|'.join(['혼자', '솔로', '싱글', '개인', 'single']), na=False)
        filtered_df = df[(multiplayer_mask | multiplayer_answer_mask) & negative_mask]
        print(f"멀티플레이어 필터링: {original_size} -> {len(filtered_df)}개")
    
    elif user_keywords['playstyle'].get('solo'):
        print(f"솔로 플레이 키워드 감지: {user_keywords['playstyle']['solo']}")
        solo_mask = df['question'].str.contains('|'.join(['혼자', '솔로', '싱글']), na=False)
        solo_answer_mask = df['answer'].str.contains('|'.join(['싱글', '혼자', '개인']), na=False)
        # 멀티플레이어 게임 제외
        negative_mask = ~df['answer'].str.contains('|'.join(['멀티', '협동', '팀', 'co-op']), na=False)
        filtered_df = df[(solo_mask | solo_answer_mask) | negative_mask]
        print(f"솔로 플레이 필터링: {original_size} -> {len(filtered_df)}개")
    
    # 2. 장르 필터링 (기존 필터링된 결과에 추가 적용)
    if user_keywords['genre']:
        for genre, keywords in user_keywords['genre'].items():
            print(f"{genre} 장르 키워드 감지: {keywords}")
            
            # 장르별 마스크 생성
            if genre == 'horror':
                genre_mask = filtered_df['answer'].str.contains('|'.join(['공포', '호러', 'horror', 'Horror']), na=False, case=False)
            elif genre == 'survival':
                genre_mask = filtered_df['answer'].str.contains('|'.join(['생존', '서바이벌', 'survival', 'Survival']), na=False, case=False)
            elif genre == 'action':
                genre_mask = filtered_df['answer'].str.contains('|'.join(['액션', 'action', 'Action', '전투']), na=False, case=False)
            elif genre == 'rpg':
                genre_mask = filtered_df['answer'].str.contains('|'.join(['RPG', 'rpg', '롤플레잉']), na=False, case=False)
            elif genre == 'strategy':
                genre_mask = filtered_df['answer'].str.contains('|'.join(['전략', 'strategy', 'Strategy', '전술']), na=False, case=False)
            elif genre == 'simulation':
                genre_mask = filtered_df['answer'].str.contains('|'.join(['시뮬레이션', 'simulation', 'Simulation']), na=False, case=False)
            else:
                continue
            
            # 필터링 결과가 너무 적으면 적용하지 않음 (최소 30개 유지)
            potential_filtered = filtered_df[genre_mask]
            if len(potential_filtered) >= 30:
                filtered_df = potential_filtered
                print(f"{genre} 장르 필터링 적용: {len(filtered_df)}개")
            else:
                print(f"{genre} 장르 필터링 스킵: 후보가 너무 적음 ({len(potential_filtered)}개)")
    
    # 3. 난이도 필터링 (선택적)
    if user_keywords['difficulty']:
        for diff, keywords in user_keywords['difficulty'].items():
            print(f"{diff} 난이도 키워드 감지: {keywords}")
            if diff == 'hard':
                # 하드코어 관련 키워드가 있는 게임 우선
                hard_mask = filtered_df['answer'].str.contains('|'.join(['하드코어', 'hardcore', '도전적', '어려운']), na=False, case=False)
                potential_filtered = filtered_df[hard_mask]
                if len(potential_filtered) >= 20:
                    filtered_df = potential_filtered
                    print(f"하드코어 난이도 필터링 적용: {len(filtered_df)}개")
    
    print(f"최종 필터링 결과: {original_size} -> {len(filtered_df)}개 후보 ({(1-len(filtered_df)/original_size)*100:.1f}% 감소)")
    return filtered_df

# 미리 질문들의 임베딩을 계산 (배치 처리로 최적화)
print("질문 임베딩을 계산 중...")
questions = df['question'].tolist()

# 배치 단위로 임베딩 계산
batch_size = 16  # GPU 메모리에 맞게 조정
question_embeddings = []

for i in range(0, len(questions), batch_size):
    batch_questions = questions[i:i+batch_size]
    batch_embeddings = get_embedding_batch(batch_questions, batch_size=len(batch_questions))
    question_embeddings.extend(batch_embeddings)
    
    if i % (batch_size * 10) == 0:
        print(f"진행률: {i}/{len(questions)} ({i/len(questions)*100:.1f}%)")

question_embeddings = np.array(question_embeddings)
print("임베딩 계산 완료!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': '질문을 입력해주세요.'}), 400
        
        # 키워드 기반 필터링
        filtered_df = filter_candidates_by_keywords(question, df)
        
        # 필터링된 데이터가 있으면 해당 인덱스만 사용
        if len(filtered_df) < len(df):
            filtered_indices = filtered_df.index.tolist()
            filtered_embeddings = question_embeddings[filtered_indices]
        else:
            filtered_indices = list(range(len(df)))
            filtered_embeddings = question_embeddings
        
        # 입력 질문의 임베딩 계산
        user_embedding = get_embedding_cached(question)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(user_embedding, filtered_embeddings)[0]
        
        # 상위 3개 유사한 질문 찾기
        top_local_indices = np.argsort(similarities)[::-1][:3]
        best_local_idx = top_local_indices[0]
        best_global_idx = filtered_indices[best_local_idx]
        best_similarity = similarities[best_local_idx]
        
        # 디버깅 정보 - 상위 3개 결과 출력
        print(f"사용자 질문: {question}")
        print(f"필터링된 후보 수: {len(filtered_df)}")
        print("상위 3개 매칭 결과:")
        for i, local_idx in enumerate(top_local_indices):
            global_idx = filtered_indices[local_idx]
            sim = similarities[local_idx]
            matched_q = df.iloc[global_idx]['question']
            print(f"  {i+1}. 유사도: {sim:.3f} - {matched_q}")
        
        # 더 엄격한 유사도 임계값 적용
        threshold = 0.6  # 0.5에서 0.6으로 상향
        if best_similarity < threshold:
            answer = f"죄송합니다. 해당 질문에 대한 적절한 게임 추천을 찾지 못했습니다. (유사도: {best_similarity:.3f}) 다른 방식으로 질문해보시겠어요?"
        else:
            answer = df.iloc[best_global_idx]['answer']
        
        return jsonify({
            'question': question,
            'answer': answer,
            'similarity': float(best_similarity),
            'matched_question': df.iloc[best_global_idx]['question'],
            'filtered_candidates': len(filtered_df)
        })
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🎮 PickGame - AI 게임 추천 웹서비스를 시작합니다 (훈련된 모델 필요)...")
    app.run(host='0.0.0.0', debug=True)

