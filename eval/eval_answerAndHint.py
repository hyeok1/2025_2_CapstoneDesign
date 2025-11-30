import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,       
    answer_relevancy,   
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import visualize

load_dotenv()

# --- [설정] ---
INPUT_FILE = 'made_by_PDFRAG2.json' 
OUTPUT_FILE1 = 'ragas_evaluation_result_answer.csv' # 결과 파일명 변경
OUTPUT_FILE2 = 'ragas_evaluation_result_hint.csv' # 결과 파일명 변경

def prepare_separate_datasets(input_file):
    """
    JSON 파일을 읽어 '해설 평가용'과 '힌트 평가용' 두 개의 데이터셋을 생성합니다.
    """
    print(f"1. 데이터셋 로드 및 분리 중... ({input_file})")
    
    data_buffer = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_buffer = json.load(f)
            if not isinstance(data_buffer, list):
                if isinstance(data_buffer, dict):
                    data_buffer = [data_buffer]
                else:
                    return None, None
    except Exception as e:
        print(f"오류: {e}")
        return None, None

    # 두 개의 데이터셋을 위한 딕셔너리
    expl_data = {"question": [], "answer": [], "contexts": []}
    hint_data = {"question": [], "answer": [], "contexts": []}

    for quiz in data_buffer:
        question_text = quiz.get('question', '')
        real_answer = quiz.get('answer', '')
        
        # 각각의 텍스트 가져오기
        explanation_text = quiz.get('explanation', '')
        hint_text = quiz.get('hint', '')
        
        retrieved_contexts = quiz.get('contexts', [])

        if not question_text: continue

        # 1. 해설(Explanation) 데이터셋 구성
        # (RAGAs의 'answer' 컬럼에 해설을 넣음)
        expl_data["question"].append(question_text)
        expl_data["answer"].append(explanation_text)
        expl_data["contexts"].append(retrieved_contexts)

        # 2. 힌트(Hint) 데이터셋 구성
        hint_data["question"]
        hint_data["answer"].append(hint_text)
        hint_data["contexts"].append(retrieved_contexts)


    return Dataset.from_dict(expl_data), Dataset.from_dict(hint_data)

def main_eval():
    # 1. 데이터셋 준비
    expl_dataset, hint_dataset = prepare_separate_datasets(INPUT_FILE)
    
    if not expl_dataset or not hint_dataset:
        print("데이터셋 생성 실패.")
        return

    # --- [수정] Timeout 설정 추가 ---
    # timeout=300 (5분)으로 설정하여 응답이 늦어도 기다리도록 합니다.
    # max_retries=3 (실패 시 3번 재시도)을 추가하여 안정성을 높입니다.
    
    print("평가용 모델을 로드합니다 (Timeout: 300s)...")
    
    evaluator_llm = ChatOpenAI(
        model="gpt-4o", 
        timeout=300,        # [중요] 타임아웃 5분으로 연장
        max_retries=3       # [중요] 실패 시 자동 재시도
    )
    
    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        timeout=300,        # [중요] 타임아웃 5분으로 연장
        max_retries=3
    )
    
    metrics = [faithfulness, answer_relevancy]

    # ---------------------------------------------------------
    # 2-1. 해설(Explanation) 평가 실행
    # ---------------------------------------------------------
    print(f"\n[1/2] 해설(Explanation) 평가 시작 ({len(expl_dataset)}개)...")
    
    # [팁] 평가 데이터가 많다면 raise_exceptions=False가 필수입니다.
    result_expl = evaluate(
        dataset=expl_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=3)
    )
    df_expl = result_expl.to_pandas()
    df_expl['type'] = 'explanation'

    # ---------------------------------------------------------
    # 2-2. 힌트(Hint) 평가 실행
    # ---------------------------------------------------------
    print(f"\n[2/2] 힌트(Hint) 평가 시작 ({len(hint_dataset)}개)...")
    
    result_hint = evaluate(
        dataset=hint_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=3)
    )
    df_hint = result_hint.to_pandas()
    df_hint['type'] = 'hint'

    df_expl.to_csv(OUTPUT_FILE1, index=False, encoding="utf-8-sig")
    df_hint.to_csv(OUTPUT_FILE2, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main_eval()
