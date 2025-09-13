import torch
import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

class SimpleGameDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        answer = str(self.answers[idx])
        
        # 질문과 답변을 함께 인코딩
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 간단한 start/end 위치 설정
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # [SEP] 토큰 위치 찾기
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_token_id).nonzero().flatten()
        
        if len(sep_positions) >= 2:
            start_pos = sep_positions[0].item() + 1
            end_pos = min(sep_positions[1].item() - 1, self.max_length - 1)
        else:
            start_pos = len(self.tokenizer.encode(question, add_special_tokens=True))
            end_pos = min(start_pos + 10, self.max_length - 1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_pos, dtype=torch.long),
            'end_positions': torch.tensor(end_pos, dtype=torch.long)
        }

def simple_train(csv_file="game_qa_data.csv", model_dir="./model", epochs=3, batch_size=4):
    print("🎮 간단한 게임 추천 모델 훈련 시작!")
    
    # 데이터 로드
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일이 없습니다: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"📊 데이터 로드: {len(df)}개")
    
    # 토크나이저와 모델 로드
    print("🤖 모델 로드 중...")
    model_name = "beomi/kcbert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    # 데이터 분할
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    
    train_questions, val_questions, train_answers, val_answers = train_test_split(
        questions, answers, test_size=0.2, random_state=42
    )
    
    # 데이터셋과 데이터로더
    train_dataset = SimpleGameDataset(train_questions, train_answers, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🚀 훈련 시작 (디바이스: {device})")
    
    # 훈련 루프
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 완료 - 평균 손실: {avg_loss:.4f}")
    
    # 모델 저장
    print(f"💾 모델 저장 중: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print("✅ 훈련 완료!")
    return True

if __name__ == "__main__":
    success = simple_train()
    if success:
        print("\n🎉 성공! 이제 웹서비스를 실행해보세요:")
        print("python app.py")
    else:
        print("\n❌ 훈련 실패") 