import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# 게임 추천 데이터셋 클래스
class GameQADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # BERT QA 형식에 맞게 입력 구성
        inputs = self.tokenizer(
            question,
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # start_positions과 end_positions 찾기
        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        answer_start = inputs['input_ids'][0].tolist().index(answer_tokens[0])
        answer_end = answer_start + len(answer_tokens) - 1

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'start_positions': torch.tensor(answer_start),
            'end_positions': torch.tensor(answer_end)
        }

def train_model(csv_file_path="augmented_game_qa_data.csv", model_save_path="./model"):
    print("게임 추천 BERT 모델 훈련을 시작합니다!")
    print("=" * 60)
    
    # 하이퍼파라미터 설정
    model_name = "klue/bert-base"
    max_seq_length = 300
    batch_size = 2
    learning_rate = 5e-5
    epochs = 3
    seed = 42
    
    # 랜덤 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 데이터 로드
    try:
        df = pd.read_csv(csv_file_path)
        print(f"데이터 로드 완료: {len(df)}개 항목")
        
        if len(df) == 0:
            print("❌ CSV 파일이 비어있습니다.")
            return False
            
        if 'question' not in df.columns or 'answer' not in df.columns:
            print("❌ CSV 파일에 'question', 'answer' 컬럼이 필요합니다.")
            print(f"현재 컬럼: {list(df.columns)}")
            return False
            
        print(f"데이터 미리보기:")
        print(f"- 질문 예시: {df['question'].iloc[0]}")
        print(f"- 답변 예시: {df['answer'].iloc[0][:50]}...")
        print()
            
    except Exception as e:
        print(f"❌ CSV 파일 읽기 오류: {e}")
        return False
    
    # 토크나이저 준비
    print("🔤 토크나이저를 준비합니다...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=300)
        print("✅ 토크나이저 로드 완료")
    except Exception as e:
        print(f"❌ 토크나이저 로드 실패: {e}")
        return False
    
    # 데이터를 train/val로 분할
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed)
    
    print(f"데이터 분할:")
    print(f"- 훈련 데이터: {len(train_df)}개")
    print(f"- 검증 데이터: {len(val_df)}개")
    print()
    
    # 데이터셋 준비
    train_dataset = GameQADataset(
        train_df['question'].tolist(),
        train_df['answer'].tolist(),
        tokenizer,
        max_seq_length
    )
    val_dataset = GameQADataset(
        val_df['question'].tolist(),
        val_df['answer'].tolist(),
        tokenizer,
        max_seq_length
    )
    
    # 데이터로더 준비
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0
    )
    
    # 모델 초기화
    print("BERT 모델을 초기화합니다...")
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False
    
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 모델을 GPU로 이동
    model = model.to(device)
    
    # 훈련 시작
    print("\n훈련을 시작합니다!")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        # 훈련 루프
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            
            # 손실이 유효한 경우에만 역전파
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_train_loss += loss.item()
                train_steps += 1
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_pbar.set_postfix({'loss': loss.item() if not torch.isnan(loss) else 'nan'})
        
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else float('inf')
        
        # 검증 루프
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for batch in val_pbar:
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
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_val_loss += loss.item()
                    val_steps += 1
                
                val_pbar.set_postfix({'loss': loss.item() if not torch.isnan(loss) else 'nan'})
        
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"✅ 모델이 저장되었습니다: {model_save_path}")
    
    print("\n훈련이 완료되었습니다!")
    return True

if __name__ == "__main__":
    train_model() 