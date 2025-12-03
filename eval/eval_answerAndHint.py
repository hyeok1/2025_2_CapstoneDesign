import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

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

    expl_data = {"question": [], "answer": [], "contexts": []}
    hint_data = {"question": [], "answer": [], "contexts": []}

    for quiz in data_buffer:
        question_text = quiz.get('question', '')        
        explanation_text = quiz.get('explanation', '')
        hint_text = quiz.get('hint', '')
        retrieved_contexts = quiz.get('contexts', [])

        if not question_text: continue

        # 1. 해설(Explanation) 데이터셋
        expl_data["question"].append(question_text)
        expl_data["answer"].append(explanation_text)
        expl_data["contexts"].append(retrieved_contexts)

        # 2. 힌트(Hint) 데이터셋
        # [오타 수정] hint_data["question"] -> append
        hint_data["question"].append(question_text)
        hint_data["answer"].append(hint_text)
        hint_data["contexts"].append(retrieved_contexts)

    return Dataset.from_dict(expl_data), Dataset.from_dict(hint_data)


# [수정] 파라미터 추가
def main_eval(file_type, chunk_size, chunk_overlap, embedding_size):
    config_str = f"CS{chunk_size}_CO{chunk_overlap}_ES{embedding_size}"
    if file_type == 'PDF':
        input_file = os.path.join('Result', 'RAG', 'PDF', f'PDFRAG_answerAndHint_{config_str}.json')
    elif file_type == 'Markdown':
        input_file = os.path.join('Result', 'RAG', 'Markdown', f'MarkdownRAG_answerAndHint_{config_str}.json')
    else:
        print("알 수 없는 파일 타입입니다.")
        return

    # 출력 파일 (평가 결과 파일)
    output_file_expl = os.path.join('Result', 'RAGAS', 'Explanation', f'Eval_Expl_{file_type}_{config_str}.csv')
    output_file_hint = os.path.join('Result', 'RAGAS', 'Hint', f'Eval_Hint_{file_type}_{config_str}.csv')

    # 1. 데이터셋 준비
    expl_dataset, hint_dataset = prepare_separate_datasets(input_file)
    
    if not expl_dataset or not hint_dataset:
        print(f"데이터셋 생성 실패. 파일({input_file})이 존재하는지 확인하세요.")
        return

    print("평가용 모델을 로드합니다 (Timeout: 300s)...")
    
    evaluator_llm = ChatOpenAI(
        model="gpt-4o", 
        timeout=300,        
        max_retries=3       
    )
    
    # [수정] 평가용 임베딩 모델도 embedding_size에 맞춰 설정 (중요!)
    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=embedding_size, # 평가 시에도 차원 일치 필요
        timeout=300,        
        max_retries=3
    )
    
    metrics = [faithfulness, answer_relevancy]

    # 2-1. 해설 평가
    print(f"\n[1/2] 해설(Explanation) 평가 시작 ({len(expl_dataset)}개)...")
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

    # 2-2. 힌트 평가
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

    # 3. 결과 저장
    df_expl.to_csv(output_file_expl, index=False, encoding="utf-8-sig")
    df_hint.to_csv(output_file_hint, index=False, encoding="utf-8-sig")

    print("\n--- 평가 결과 (해설) ---")
    print(result_expl)

    print("\n--- 평가 결과 (힌트) ---")
    print(result_hint)

    
    print(f"\n평가 완료! 결과 파일 저장됨:")
    print(f" - {output_file_expl}")
    print(f" - {output_file_hint}")

if __name__ == "__main__":
    # 테스트용 기본값
    main_eval('Markdown', 1000, 50, 3072)