import json
from pathlib import Path
from langchain_core.documents import Document

PDF_FILEPATHS = [
    'Database_RAG\PDF\EPS_Notion.pdf',
    'Database_RAG\PDF\KIIP_Notion_DB_Advanced.pdf',
    'Database_RAG\PDF\KIIP_Notion_DB.pdf'
]

MARKDOWN_FILEPATHS = [
    'Database_RAG\Markdown\EPS_Notion.md',
    'Database_RAG\Markdown\KIIP_Notion_DB_Advanced.md',
    'Database_RAG\Markdown\KIIP_Notion_DB.md'
]

def make_contextList(docs : list[Document]) -> list:
    return [doc.page_content.replace("\n", "") for doc in docs]

def format_docs(docs: list[Document]) -> str:
    # (이 함수는 변경 없음)
    return "\n\n".join(doc.page_content for doc in docs)

def read_context(filepath: str) -> str:
    """
    지정된 경로의 파일에서 문자열을 읽어옵니다.

    Args:
        filepath: 파일 경로

    Returns:
        파일의 전체 내용 (문자열)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            template_str = f.read()
        return template_str
    except FileNotFoundError:
        print(f"오류: 프롬프트 템플릿 파일을 찾을 수 없습니다. (경로: {filepath})")
        return "" # 빈 문자열 반환
    except Exception as e:
        print(f"오류: 프롬프트 템플릿 파일 로드 중 문제 발생: {e}")
        return ""

def getJsonPath():
    # 1. 현재 이 스크립트 파일(MakeDB.py)이 있는 디렉토리를 기준으로 삼습니다.
    script_dir = Path(__file__).parent 

    # 2. 검색을 시작할 상위 폴더 경로를 만듭니다. (Dataset/Culture)
    search_dir = script_dir / 'CLIcK' /'Dataset' / 'Culture' / 'KIIP'

    # 3. search_dir와 그 모든 하위 폴더에서 '*.json' 패턴의 파일을 찾습니다.
    #    rglob은 제너레이터(generator)이므로 list()로 감싸서 리스트로 만듭니다.
    json_file_paths = list(search_dir.rglob('*.json'))

    return json_file_paths

def save_results(data, output_filename: str):
    """
    처리된 결과를 JSON 파일로 저장합니다.
    """
    
    final_data_to_save = data # 기본값
    merged_list = []
    
    # 딕셔너리의 '값' (퀴즈 리스트) 들만 순회하며 'merged_list'에 확장
    for quiz_list in data.values():
        merged_list.extend(quiz_list)
    
    final_data_to_save = merged_list # 저장할 데이터를 병합된 리스트로 교체
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # final_data_to_save (병합된 리스트 또는 원본)를 저장합니다.
            json.dump(final_data_to_save, f, ensure_ascii=False, indent=4)
        print(f"\n[저장 완료] 모든 해설이 '{output_filename}' 파일에 성공적으로 저장되었습니다.")
    
    except Exception as e:
        print(f"[저장 실패] 결과 파일 저장 중 오류 발생: {e}")