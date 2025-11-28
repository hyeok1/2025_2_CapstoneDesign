import argparse
import sys
import os
from dotenv import load_dotenv

try:
    from RAG import main_rag
    from LLM import main_base_llm
    from db_manager import force_rebuild_db
except ImportError as e:
    print(f"오류: 필요한 모듈을 임포트할 수 없습니다: {e}")
    sys.exit(1)

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("경고: .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")

    # --- Argparse CLI 설정 ---
    parser = argparse.ArgumentParser(
        description="K-Culture 퀴즈 RAG 평가 시스템 CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # [수정] 전역 --type 인자를 여기에서 삭제합니다.
    
    subparsers = parser.add_subparsers(
        dest="command", 
        required=True,
        help="실행할 작업을 선택하세요."
    )

    # 1. 'rag' 명령어 설정
    parser_rag = subparsers.add_parser(
        "rag", 
        help="RAG 모델 (PDF/MD 참조)을 사용하여 퀴즈 해설을 생성합니다."
    )
    # [수정] 'rag' 명령어에만 속하는 --type 인자를 추가합니다.
    parser_rag.add_argument(
        '--type',
        type=str,
        default='Markdown', # 기본값은 Markdown DB
        choices=['PDF', 'Markdown'],
        help="사용할 지식 베이스 소스 타입을 선택합니다. (기본값: Markdown)"
    )
    
    # 2. 'base' 명령어 설정
    parser_base = subparsers.add_parser(
        "base", 
        help="Base LLM (일반 지식)을 사용하여 퀴즈 해설을 생성합니다."
    )
    # (base 명령어는 DB를 사용하지 않으므로 --type 인자가 필요 없습니다.)

    # 3. 'build_db' 명령어 설정
    parser_build = subparsers.add_parser(
        "build_db", 
        help="벡터 DB(FAISS 인덱스)를 강제로 삭제하고 새로 생성합니다."
    )
    # [수정] 'build_db' 명령어에만 속하는 --type 인자를 추가합니다.
    parser_build.add_argument(
        '--type',
        type=str,
        default='Markdown', # 기본값은 Markdown DB
        choices=['PDF', 'Markdown'],
        help="재생성할 지식 베이스 소스 타입을 선택합니다. (기본값: Markdown)"
    )
    # ----------------------------

    args = parser.parse_args()

    # --- 명령어 실행 ---
    if args.command == "rag":
        print(f"--- RAG ({args.type}) 기반 해설 생성을 시작합니다 ---")
        main_rag(args.type) # args.type을 main_rag로 전달
        print(f"--- RAG ({args.type}) 기반 해설 생성 완료 ---")
    
    elif args.command == "base":
        print("--- Base LLM 기반 해설 생성을 시작합니다 ---")
        main_base_llm()
        print("--- Base LLM 기반 해설 생성 완료 ---")
        
    elif args.command == "build_db":
        print(f"--- 벡터 DB ({args.type}) 강제 재생성을 시작합니다 ---")
        force_rebuild_db(args.type) # args.type을 force_rebuild_db로 전달
        print(f"--- 벡터 DB ({args.type}) 강제 재생성 완료 ---")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()