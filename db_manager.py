import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from util import read_context

# --- [신규] 파일 타입별 상수 정의 ---
PDF_FAISS_SAVE_PATH = "faiss_index_pdf"
MARKDOWN_FAISS_SAVE_PATH = "faiss_index_markdown"

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
# ------------------------------------

def get_ChunkDocuments(fileType : str) -> list:
    
    all_split_documents = [] # 최종 분할된 문서를 담을 리스트

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

        print("\n단계 2: (Recursive) 문서 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, 
            chunk_overlap = 50
        )
        all_split_documents = text_splitter.split_documents(all_docs)
    elif fileType == "Markdown":
        headers_to_split_on = [
            ("#", "H1"), ("##", "H2"), ("###", "H3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            # 분할할 헤더를 지정합니다.
            headers_to_split_on = headers_to_split_on,
            # 헤더를 제거하지 않도록 설정합니다.
            strip_headers=False,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500, 
            chunk_overlap = 50
        )

        for filepath in MARKDOWN_FILEPATHS:
            try:
                markdown_document_text = read_context(filepath)
                
                # 2b. 헤더 기준으로 1차 분할 (큰 덩어리 Document 리스트)
                header_chunks = markdown_splitter.split_text(markdown_document_text)
                
                # 2c. 크기 기준으로 2차 분할 (작은 덩어리 Document 리스트)
                final_chunks = text_splitter.split_documents(header_chunks)

                # 2d. (개선) 메타데이터에 원본 파일 경로 추가 (매우 중요)
                for chunk in final_chunks:
                    chunk.metadata["source"] = filepath
                
                # 2e. 최종 리스트에 추가
                all_split_documents.extend(final_chunks)
                print(f"'{filepath}' 로드 및 2단계 분할 완료. (청크 수: {len(final_chunks)})")

            except Exception as e:
                print(f"경고: '{filepath}' 처리 중 오류: {e}")

        if not all_split_documents:
            print("오류 : 로드/분할할 Markdown 문서가 없습니다.")
            return None
    
    else:
        print(f"오류: 알 수 없는 fileType '{fileType}'입니다.")
        return None

    return all_split_documents

def get_vectorstore(embeddings_model: OpenAIEmbeddings, fileType: str) -> FAISS:
    """
    fileType에 따라 로컬에 저장된 FAISS 벡터 DB를 로드하거나,
    없으면 새로 생성하여 반환합니다.
    """
    
    # 1. fileType에 따라 경로와 파일 리스트를 결정합니다.
    if fileType == "PDF":
        save_path = PDF_FAISS_SAVE_PATH
    elif fileType == "Markdown":
        save_path = MARKDOWN_FAISS_SAVE_PATH
    else:
        print(f"오류: 알 수 없는 fileType '{fileType}'입니다. None을 반환합니다.")
        return None

    # 2. 결정된 경로(save_path)에 DB가 있는지 확인하고 로드합니다.
    if os.path.exists(save_path):
        print(f"'{save_path}'에서 기존 {fileType} 벡터 DB를 로드합니다...")
        vectorstore = FAISS.load_local(
            save_path, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        print("벡터 DB 로드 완료.")
        return vectorstore

    # 3. DB가 없으면 fileType에 맞는 로직으로 새로 생성합니다.
    print(f"기존 {fileType} DB를 찾을 수 없습니다. '{save_path}'에 새로 생성합니다.")
    
    all_split_documents = get_ChunkDocuments(fileType)

    # 4. (공통) 임베딩 생성 및 저장
    print(f"\n총 {len(all_split_documents)}개의 문서 조각 생성 완료.")
    print("\n단계 3-4: 임베딩 생성 및 벡터 DB 저장 (API 호출 발생)...")
    
    vectorstore = FAISS.from_documents(all_split_documents, embeddings_model)
    vectorstore.save_local(save_path) # 결정된 경로에 저장
    
    print(f"벡터 DB를 '{save_path}'에 저장 완료.")
    
    return vectorstore

def force_rebuild_db(fileType: str):
    """
    fileType에 따라 기존 FAISS 폴더를 강제로 삭제하고 DB를 새로 생성합니다.
    """
    # 1. fileType에 따라 삭제할 경로를 결정
    path_to_delete = ""
    if fileType == "PDF":
        path_to_delete = PDF_FAISS_SAVE_PATH
    elif fileType == "Markdown":
        path_to_delete = MARKDOWN_FAISS_SAVE_PATH
    else:
        print(f"오류: 알 수 없는 fileType '{fileType}'입니다. 재생성을 중단합니다.")
        return

    # 2. 기존 폴더 삭제
    if os.path.exists(path_to_delete):
        print(f"'{path_to_delete}' 폴더를 삭제합니다...")
        try:
            shutil.rmtree(path_to_delete)
            print("폴더 삭제 완료.")
        except Exception as e:
            print(f"폴더 삭제 중 오류 발생: {e}")
            return

    # 3. DB 재생성 (get_vectorstore 호출)
    print(f"({fileType}) DB 재생성을 시작합니다...")
    try:
        # (API 키는 main.py에서 load_dotenv()로 이미 로드됨)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # 고성능 모델 권장
        get_vectorstore(embeddings, fileType) # fileType을 전달
    except Exception as e:
        print(f"DB 재생성 중 오류 발생: {e}")
        print("OPENAI_API_KEY가 .env 파일에 설정되어 있는지 확인하세요.")

if __name__ == "__main__":
    # 이 파일을 직접 실행하면 Markdown DB를 강제로 새로 생성합니다.
    load_dotenv()
    for header in get_ChunkDocuments("Markdown"):
        print(f"{header.page_content}")
        print(f"{header.metadata}", end="\n=====================\n")