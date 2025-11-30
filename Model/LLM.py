import os, json, time
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from util import getJsonPath, save_results, read_context

def process_quizzes(json_file_paths: list[Path], base_chain, output_filename: str) -> dict:
        # ... (이전과 동일한 함수 내용) ...
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
                    explanations_list = []

                    for choice_text in choices_list:
                        input_data = {
                            "question": question,
                            "choice": choice_text,
                            "answer": correct_answer
                        }
                        explanation = base_chain.invoke(input_data)
                        explanations_list.append(explanation)
                        time.sleep(1)
                    
                    quiz['explaination'] = explanations_list
                    processed_quizzes_for_this_file.append(quiz)
                     
                    print(f"Processed Quiz ID: {quiz_id}")

                all_explanations[filename_key] = processed_quizzes_for_this_file
            except json.JSONDecodeError:
                print(f"  오류: {json_file_path.name} 파일이 올바른 JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"  오류: {json_file_path.name} 처리 중 문제 발생: {e}")

        print("\n[--- 모든 JSON 파일 처리 완료 ---]")
        return all_explanations

def main_base_llm():

    # --- 단계 1 : 프롬프트 생성 (Base LLM용) ---
    BASE_TEMPLATE_FILE = r"template\template_LLM.txt"
    template = read_context(BASE_TEMPLATE_FILE)

    if not template:
        print(f"'{BASE_TEMPLATE_FILE}' 로드 실패. 스크립트를 종료합니다.")
        return

    prompt = PromptTemplate.from_template(template)

    # --- 단계 2 : 언어모델(LLM) 생성 ---
    llm = ChatOpenAI(model_name = "gpt-4o", temperature = 0)

    # --- 단계 3 : Base LLM 체인 정의 ---
    base_chain_per_choice = (
        prompt
        | llm
        | StrOutputParser()
    )

    # --- 단계 5: 메인 실행 흐름 ---
    output_filename = 'made_by_LLM.json'
    json_file_paths = getJsonPath() 
    #json_file_paths = [Path('example.json')]
    all_explanations = process_quizzes(
        json_file_paths, 
        base_chain_per_choice, 
        output_filename
    )
    save_results(all_explanations, output_filename)

# [신규] 이 파일을 직접 실행하는 것을 방지
if __name__ == "__main__":
    print("이 스크립트는 직접 실행할 수 없습니다.")
    print("`python main.py base` 명령어를 사용하세요.")