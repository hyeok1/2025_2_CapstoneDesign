import os, json, time
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from util import format_docs, getJsonPath, save_results, read_context, make_contextList
from db_manager import get_vectorstore

def process_quizzes(json_file_paths: list[Path], context, rag_chain, output_filename: str) -> dict:
    all_explanations = {}

    print(f"총 {len(json_file_paths)}개의 JSON 퀴즈 파일을 찾았습니다.")
    print("=" * 60)

    for json_file_path in json_file_paths:
        filename_key = json_file_path.name

        if filename_key == output_filename:
            continue

        print(f"\n[--- JSON 파일 처리 시작: {filename_key} ---]")

        if filename_key not in all_explanations:
            all_explanations[filename_key] = []

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                quizzes = json.load(f)

            processed_quizzes_for_this_file = []

            for i, quiz in enumerate(quizzes):
                quiz_id = quiz.get('id', f'quiz_{i}')
                question = quiz['question']
                correct_answer = quiz['answer']
                choices_list = quiz['choices']

                contextStr = context.invoke({"question" : question})
                explanations_list = []

                for choice_text in choices_list:

                    input_data = {
                        "question": question,
                        "choice": choice_text,
                        "answer": correct_answer
                    }

                    explanation = rag_chain.invoke(input_data)
                    explanations_list.append(explanation)
                    time.sleep(1)

                quiz['explanation'] = explanations_list
                quiz['context'] = contextStr

                # 2. [수정] 'paragraph' 키 삭제 (만약 존재한다면)

                if 'paragraph' in quiz:
                    quiz.pop('paragraph')

                processed_quizzes_for_this_file.append(quiz)
                print(f"Processed Quiz ID: {quiz_id}")

            all_explanations[filename_key] = processed_quizzes_for_this_file
        except json.JSONDecodeError:
            print(f"  오류: {json_file_path.name} 파일이 올바른 JSON 형식이 아닙니다.")

        except Exception as e:
            print(f"  오류: {json_file_path.name} 처리 중 문제 발생: {e}")

    print("\n[--- 모든 JSON 파일 처리 완료 ---]")
    return all_explanations

def main_rag(fileType : str):
    # --- 단계 1-5: DB 준비 ---
    print("DB 매니저를 통해 벡터 DB를 준비합니다...")

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
    vectorstore = get_vectorstore(embeddings, fileType)

    if not vectorstore:
        print("벡터 스토어 준비에 실패했습니다. 스크립트를 종료합니다.")
        return # exit() 대신 return 사용

    retriever = vectorstore.as_retriever(
        search_type = "mmr", # "유사성" 대신 "MMR" 사용
        search_kwargs = {
            "k": 5,           # 1. 한 번에 가져올 청크 개수 (기본 4 -> 5)
            "fetch_k": 20     # 2. MMR 계산을 위해 먼저 20개 후보를 조회
        }
    )

    print("검색기 생성 완료.")

    # --- 단계 6 : 프롬프트 생성(Create Prompt) ---
    RAG_TEMPLATE_FILE = r"template\template_RAG.txt"
    template = read_context(RAG_TEMPLATE_FILE)

    if not template:
        print(f"'{RAG_TEMPLATE_FILE}' 로드 실패. 스크립트를 종료합니다.")
        return

    prompt = PromptTemplate.from_template(template)

    # --- 단계 7 : 언어모델(LLM) 생성 ---
    llm = ChatOpenAI(model_name = "gpt-4o", temperature = 0)

    context = (lambda x: x['question'])  | retriever | format_docs

    #--- 단계 8 : RAG 체인 정의 ---
    rag_chain_per_choice = (
    {
        "context": context,
        "question": lambda x: x['question'],
        "choice": lambda x: x['choice'],
        "answer": lambda x: x['answer'],
    }
    | prompt
    | llm
    | StrOutputParser()
    )
    
    if fileType == 'PDF':
        output_filename = 'made_by_PDFRAG.json'
    elif fileType == 'Markdown':
        output_filename = 'made_by_MarkdownRAG.json'
    else:
        print('사전에 정의하지 않은 파일명입니다.')
        return None

    # --- 단계 10: 메인 실행 흐름 ---
    json_file_paths = getJsonPath()
    #json_file_paths = [Path('example.json')]
    all_explanations = process_quizzes(
        json_file_paths,
        context, 
        rag_chain_per_choice,
        output_filename
    )

    save_results(all_explanations, output_filename)

if __name__ == "__main__":
    print("이 스크립트는 직접 실행할 수 없습니다.")
    print("`python main.py rag` 명령어를 사용하세요.")

