import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # [G] 환각 여부 (Context vs Answer)
    answer_relevancy,   # [G] 답변 관련성 (Question vs Answer)
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import visualize

load_dotenv()

# --- [설정] ---
INPUT_FILE = 'made_by_PDFRAG.json' # 평가할 결과 파일
OUTPUT_FILE = 'ragas_evaluation_result.csv'
FILE_TYPE = "PDF" 

def prepare_eval_dataset(input_file):
    """
    표준 JSON 리스트([...]) 형식의 결과 파일을 읽어 RAGAs 평가용 Dataset으로 변환합니다.
    파일에 'context' 필드가 (문자열로) 있으면 사용하고, 없으면 빈 리스트로 처리합니다.
    """
    print(f"1. 데이터셋 로드 중... ({input_file})")
    
    data_buffer = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_buffer = json.load(f) # 표준 JSON 리스트 로드
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파싱 실패: {e}")
        return None
    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다.")
        return None

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],     # RAGAs는 리스트[str]을 요구함
    }

    print(f"   총 {len(data_buffer)}개 퀴즈(각 4개 선지)를 평가 데이터로 변환합니다...")
    
    for quiz in data_buffer:
        question_text = quiz.get('question', '')
        choices = quiz.get('choices', [])
        generated_explanations = quiz.get('explanation', [])
        
        # [핵심] 저장된 context 문자열 가져오기 (없으면 빈 문자열)
        context = quiz.get('context')

        # 데이터셋 펼치기 (Flatten)
        for i, choice_text in enumerate(choices):
            generated_text = generated_explanations[i]
            
            # A. 평가용 질문
            full_question = f"질문: {question_text}"
            
            # B. 데이터 추가
            ragas_data["question"].append(full_question)
            ragas_data["answer"].append(generated_text)
            
            if type(context) == str:
                ragas_data["contexts"].append([context]) 
            elif type(context) == list:
                ragas_data["contexts"].append(context) 
            else:
                print("Context type이 잘못되었습니다.")
                return None 

    return Dataset.from_dict(ragas_data)

def main_eval():
    # 1. 데이터셋 준비
    eval_dataset = prepare_eval_dataset(INPUT_FILE)
    
    if not eval_dataset or len(eval_dataset) == 0:
        print("평가할 데이터가 없습니다.")
        return

    print(f"\n2. RAGAs 평가 시작 (데이터 개수: {len(eval_dataset)}개)...")
    
    # [수정] 명시적으로 모델 전달 (오류 방지)
    evaluator_llm = ChatOpenAI(model="gpt-4o")
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # 2. 평가 실행
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness, 
            answer_relevancy, 
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False
    )

    # 3. 결과 출력
    print("\n--- 평가 결과 (Scores) ---")
    print(result)

    df = result.to_pandas()
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n상세 평가 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    main_eval()
    