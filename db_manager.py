import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from util import read_context

# --- 기본 경로 설정 ---
# 현재 파일(db_manager.py)의 위치를 기준으로 경로를 설정합니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 벡터 DB가 저장될 상위 폴더
BASE_DB_DIR = os.path.join(SCRIPT_DIR, "VectorDB")

# PDF 파일 목록
PDF_FILEPATHS = [
    os.path.join(SCRIPT_DIR, 'Database_RAG', 'PDF', 'EPS_Notion.pdf'),
    os.path.join(SCRIPT_DIR, 'Database_RAG', 'PDF', 'KIIP_Notion_DB_Advanced.pdf'),
    os.path.join(SCRIPT_DIR, 'Database_RAG', 'PDF', 'KIIP_Notion_DB.pdf')
]

# Markdown 파일 목록
MARKDOWN_FILEPATHS = [
    os.path.join(SCRIPT_DIR, 'Database_RAG', 'Markdown', 'EPS_Notion.md'),
    os.path.join(SCRIPT_DIR, 'Database_RAG', 'Markdown', 'KIIP_Notion_DB_Advanced.md'),
    os.path.join(SCRIPT_DIR, 'Database_RAG', 'Markdown', 'KIIP_Notion_DB.md')
]
# ------------------------------------

def get_db_path(fileType: str, chunk_size: int, chunk_overlap: int, embedding_size: int) -> str:
    """
    설정값에 따라 고유한 DB 폴더 경로를 생성하여 반환합니다.
    예: VectorDB/faiss_pdf_c1000_o50_e3072
    """
    folder_name = f"faiss_{fileType.lower()}_c{chunk_size}_o{chunk_overlap}_e{embedding_size}"
    return os.path.join(BASE_DB_DIR, folder_name)


def get_ChunkDocuments(fileType: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    파일 타입에 따라 문서를 로드하고, 지정된 chunk_size와 chunk_overlap으로 분할합니다.
    """
    all_split_documents = [] 

    if fileType == "PDF":
        print("PDF 문서 로드 중...")
        all_docs = []
        for filepath in PDF_FILEPATHS:
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"'{filepath}' 로드 완료. (페이지 수: {len(docs)})")
            except Exception as e:
                print(f"경고: '{filepath}' 로드 중 오류: {e}")

        if not all_docs:
            print("오류: 로드할 PDF 문서가 없습니다.")
            return None

        print(f"\n단계 2: (Recursive) 문서 분할 중... (Size: {chunk_size}, Overlap: {chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        all_split_documents = text_splitter.split_documents(all_docs)
        
    elif fileType == "Markdown":
        print("\nMarkdown 문서 로드 및 2단계 분할 중...")
        
        # 1. 헤더 분할기 정의
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        
        # 2. 텍스트 분할기 정의 (매개변수 적용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

        for filepath in MARKDOWN_FILEPATHS:
            try:
                # util.py의 함수를 이용해 텍스트 로드
                markdown_document_text = read_context(filepath)
                if not markdown_document_text: 
                    print(f"경고: '{filepath}' 파일이 비어있거나 로드 실패.")
                    continue
                
                # 1차: 헤더 기준 분할
                header_chunks = markdown_splitter.split_text(markdown_document_text)
                
                # 2차: 크기 기준 분할
                final_chunks = text_splitter.split_documents(header_chunks)

                # 메타데이터에 원본 파일 경로 추가
                for chunk in final_chunks:
                    chunk.metadata["source"] = filepath
                
                all_split_documents.extend(final_chunks)
                print(f"'{filepath}' 로드 및 분할 완료. ({len(final_chunks)} chunks)")
            except Exception as e:
                print(f"경고: '{filepath}' 처리 중 오류: {e}")

        if not all_split_documents:
            print("오류: 로드/분할할 Markdown 문서가 없습니다.")
            return None
    
    else:
        print(f"오류: 알 수 없는 fileType '{fileType}'입니다.")
        return None

    return all_split_documents


def get_vectorstore(embeddings_model: OpenAIEmbeddings, fileType: str, chunk_size: int, chunk_overlap: int, embedding_size: int) -> FAISS:
    """
    지정된 설정(fileType, size, overlap, embedding)에 맞는 벡터 DB를 로드하거나 새로 생성합니다.
    """
    
    # 1. 설정에 맞는 고유한 저장 경로 생성
    save_path = get_db_path(fileType, chunk_size, chunk_overlap, embedding_size)

    # 2. 해당 설정의 DB가 이미 있는지 확인하고 로드
    if os.path.exists(save_path):
        print(f"'{save_path}'에서 기존 벡터 DB를 로드합니다...")
        try:
            vectorstore = FAISS.load_local(
                save_path, 
                embeddings_model, 
                allow_dangerous_deserialization=True
            )
            print("벡터 DB 로드 완료.")
            return vectorstore
        except Exception as e:
            print(f"기존 DB 로드 실패 (설정 불일치 가능성): {e}")
            print("새로 생성을 시도합니다.")

    # 3. 없으면 새로 생성
    print(f"기존 DB가 없으므로 새로 생성합니다: {save_path}")
    
    # 문서 로드 및 분할 (설정값 전달)
    all_split_documents = get_ChunkDocuments(fileType, chunk_size, chunk_overlap)
    
    if not all_split_documents:
        return None

    print(f"\n총 {len(all_split_documents)}개의 문서 조각 생성 완료.")
    print("\n단계 3-4: 임베딩 생성 및 벡터 DB 저장 (API 호출 발생)...")
    
    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(all_split_documents, embeddings_model)
    
    # 저장 경로가 없으면 생성 후 저장
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    vectorstore.save_local(save_path)
    print(f"벡터 DB를 '{save_path}'에 저장 완료.")
    
    return vectorstore


def force_rebuild_db(fileType: str, embedding_size: int, chunk_size: int, chunk_overlap: int):
    """
    특정 설정의 DB 폴더만 강제로 삭제하고 재생성합니다.
    """
    # 1. 삭제할 구체적인 경로 계산
    path_to_delete = get_db_path(fileType, chunk_size, chunk_overlap, embedding_size)

    # 2. 해당 폴더 삭제
    if os.path.exists(path_to_delete):
        print(f"'{path_to_delete}' 폴더를 삭제합니다...")
        try:
            shutil.rmtree(path_to_delete)
            print("폴더 삭제 완료.")
        except Exception as e:
            print(f"폴더 삭제 중 오류 발생: {e}")
            return

    # 3. DB 재생성
    print(f"({fileType}) DB 재생성을 시작합니다... (Size: {chunk_size}, Overlap: {chunk_overlap}, Emb: {embedding_size})")
    try:
        # dimensions 파라미터 사용
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=embedding_size)
        
        # get_vectorstore 호출 (이제 모든 파라미터를 전달)
        get_vectorstore(embeddings, fileType, chunk_size, chunk_overlap, embedding_size)
        
    except Exception as e:
        print(f"DB 재생성 중 오류 발생: {e}")

if __name__ == "__main__":
    load_dotenv()
    # 테스트 실행 예시 (원하는 설정으로 변경하여 테스트 가능)
    # force_rebuild_db("Markdown", 3072, 1000, 50)
    print("이 파일을 직접 실행하면 테스트 코드가 작동합니다. main.py를 통해 실행하는 것을 권장합니다.")