import pandas as pd
import numpy as np
from typing import List, Tuple
import random
from googletrans import Translator

def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 증강을 수행하여 데이터셋을 확장"""
    augmented_data = []
    
    # 기존 데이터 복사
    augmented_data.extend(df.to_dict('records'))
    
    # 1. 질문 변형
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        
        # 질문 패턴 변형
        patterns = [
            "어떤 게임을 추천해주세요?",
            "추천 게임이 있을까요?",
            "이런 게임을 찾고 있어요:",
            "다음과 같은 게임을 원해요:"
        ]
        
        for pattern in patterns:
            new_question = f"{pattern} {question}"
            augmented_data.append({
                'question': new_question,
                'answer': answer
            })
    
    # 2. 답변 변형
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        
        # 문장 순서 변경
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if len(sentences) > 1:
            random.shuffle(sentences)
            new_answer = '. '.join(sentences) + '.'
            augmented_data.append({
                'question': question,
                'answer': new_answer
            })
    
    # 3. 게임 특성 조합
    game_attributes = {
        '장르': ['RPG', '액션', '어드벤처', '전략', '시뮬레이션', '스포츠'],
        '플랫폼': ['PC', '모바일', '콘솔'],
        '가격대': ['무료', '저렴한', '중간', '고가'],
        '난이도': ['쉬운', '보통', '어려운']
    }
    
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        
        # 랜덤하게 특성 조합
        genre = random.choice(game_attributes['장르'])
        platform = random.choice(game_attributes['플랫폼'])
        price = random.choice(game_attributes['가격대'])
        difficulty = random.choice(game_attributes['난이도'])
        
        new_question = f"{genre} 장르의 {platform}용 {price} {difficulty} 게임을 추천해주세요."
        augmented_data.append({
            'question': new_question,
            'answer': answer
        })
    
    # 중복 제거
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df = augmented_df.drop_duplicates()
    
    return augmented_df

def main():
    # 데이터 로드
    df = pd.read_csv('game_qa_data.csv')
    print(f"원본 데이터: {len(df)}개")
    
    # 데이터 증강
    augmented_df = augment_data(df)
    print(f"증강 후 데이터: {len(augmented_df)}개")
    
    # 결과 저장
    augmented_df.to_csv('augmented_game_qa_data.csv', index=False, encoding='utf-8')
    print("증강된 데이터가 저장되었습니다.")

if __name__ == "__main__":
    main() 