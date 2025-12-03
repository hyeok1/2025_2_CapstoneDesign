import json
from pathlib import Path
from langchain_core.documents import Document
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

chunk_sizes = [250, 500, 1000]      # 열(Col)
chunk_overlaps = [50, 100]     # 행(Row)
emb_sizes = [512, 1024, 2048, 3072] # X축
file_type = "PDF" # 또는 "Markdown" (파일 naming convention에 따름)

PDF_FILEPATHS = [
    'Datasets\Database_RAG\PDF\EPS_Notion.pdf',
    'Datasets\Database_RAG\PDF\KIIP_Notion_DB_Advanced.pdf',
    'Datasets\Database_RAG\PDF\KIIP_Notion_DB.pdf'
]

MARKDOWN_FILEPATHS = [
    'Datasets\Database_RAG\Markdown\EPS_Notion.md',
    'Datasets\Database_RAG\Markdown\KIIP_Notion_DB_Advanced.md',
    'Datasets\Database_RAG\Markdown\KIIP_Notion_DB.md'
]

def make_contextList(docs : list[Document]) -> list:
    return [doc.page_content.replace("\n", "") for doc in docs]

def format_docs(docs: list[Document]) -> str:
    # (이 함수는 변경 없음)
    return "\n\n".join(doc.page_content for doc in docs)

def read_context(filepath: str) -> str:
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
    script_dir = Path(__file__).parent 
    search_dir = script_dir / 'Datasets' / 'Quiz'
    json_file_paths = list(search_dir.rglob('*.json'))

    return json_file_paths

def save_results(data, output_filename: str):
    final_data_to_save = data # 기본값
    merged_list = []
    
    # 딕셔너리의 '값' (퀴즈 리스트) 들만 순회하며 'merged_list'에 확장
    for quiz_list in data.values():
        merged_list.extend(quiz_list)
    
    final_data_to_save = merged_list # 저장할 데이터를 병합된 리스트로 교체
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_data_to_save, f, ensure_ascii=False, indent=4)
        print(f"\n[저장 완료] 모든 해설이 '{output_filename}' 파일에 성공적으로 저장되었습니다.")
    
    except Exception as e:
        print(f"[저장 실패] 결과 파일 저장 중 오류 발생: {e}")

def extract_RAGAS(file_name):
    df = pd.read_csv(file_name)
    avg_faithfulness = df['faithfulness'].mean()
    avg_answer_relevancy = df['answer_relevancy'].mean()

    return avg_faithfulness, avg_answer_relevancy

def load_all_experiment_data():
    data_map = {}
    
    for emb in emb_sizes:
        expl_faith = np.zeros((len(chunk_overlaps), len(chunk_sizes)))
        expl_rel = np.zeros((len(chunk_overlaps), len(chunk_sizes)))
        hint_faith = np.zeros((len(chunk_overlaps), len(chunk_sizes)))
        hint_rel = np.zeros((len(chunk_overlaps), len(chunk_sizes)))
        
        for r_idx, overlap in enumerate(chunk_overlaps):
            for c_idx, size in enumerate(chunk_sizes):
                config_str = f"CS{size}_CO{overlap}_ES{emb}"
                
                file_expl = os.path.join('Result', 'RAGAS', 'Explanation', f'Eval_Expl_{file_type}_{config_str}.csv')
                file_hint = os.path.join('Result', 'RAGAS', 'Hint', f'Eval_Hint_{file_type}_{config_str}.csv')
                
                ef, er = extract_RAGAS(file_expl)
                hf, hr = extract_RAGAS(file_hint)
                
                expl_faith[r_idx, c_idx] = ef
                expl_rel[r_idx, c_idx] = er
                hint_faith[r_idx, c_idx] = hf
                hint_rel[r_idx, c_idx] = hr
        
        data_map[emb] = {
            'EF': expl_faith, 
            'ER': expl_rel, 
            'HF': hint_faith, 
            'HR': hint_rel
        }
        
    return data_map

def prepare_dataframe(data_map):
    records = []
    for emb in emb_sizes:
        d = data_map[emb]
        for r_idx, overlap in enumerate(chunk_overlaps):
            for c_idx, size in enumerate(chunk_sizes):
                records.append({
                    'Embedding': emb,
                    'Size': size,
                    'Overlap': overlap,
                    'Expl Faith': d['EF'][r_idx, c_idx],
                    'Expl Rel': d['ER'][r_idx, c_idx],
                    'Hint Faith': d['HF'][r_idx, c_idx],
                    'Hint Rel': d['HR'][r_idx, c_idx]
                })
    return pd.DataFrame(records)

def plot_trend_lines(df):
    filename = os.path.join('Result', 'RAGAS', 'Image', 'Evaluation_embedding_ES.png')
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle('Impact of Embedding Size on RAG Performance by Chunk Strategy', fontsize=20, fontweight='bold')

    for r_idx, overlap in enumerate(chunk_overlaps):
        for c_idx, size in enumerate(chunk_sizes):
            ax = axes[r_idx, c_idx]
            subset = df[(df['Size'] == size) & (df['Overlap'] == overlap)]
            x = subset['Embedding']
            
            ax.plot(x, subset['Expl Faith'], marker='o', label='Expl: Faithfulness', color='blue', linewidth=2, linestyle='-')
            ax.plot(x, subset['Expl Rel'],   marker='s', label='Expl: Relevancy',   color='green', linewidth=2, linestyle='--')
            ax.plot(x, subset['Hint Faith'], marker='^', label='Hint: Faithfulness', color='purple', linewidth=2, linestyle='-')
            ax.plot(x, subset['Hint Rel'],   marker='D', label='Hint: Relevancy',   color='orange', linewidth=2, linestyle='--')

            ax.set_title(f'Chunk Size: {size} / Overlap: {overlap}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Embedding Size (Dimension)')
            ax.set_ylabel('Score')
            ax.set_ylim(0.4, 1.0) 
            ax.set_xticks(emb_sizes)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])    
    plt.savefig(filename)

def plot_heatmaps(data_map):
    for emb_size, data in data_map.items():
        filename = os.path.join('Result', 'RAGAS', 'Image', f'rag_result_heatmap_e{emb_size}.png')
        df_ef = pd.DataFrame(data['EF'], index=chunk_overlaps, columns=chunk_sizes)
        df_er = pd.DataFrame(data['ER'], index=chunk_overlaps, columns=chunk_sizes)
        df_hf = pd.DataFrame(data['HF'], index=chunk_overlaps, columns=chunk_sizes)
        df_hr = pd.DataFrame(data['HR'], index=chunk_overlaps, columns=chunk_sizes)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Performance Analysis - Embedding Size {emb_size}', fontsize=16, fontweight='bold')
        
        fmt = ".4f"
        kws = {"size": 13, "weight": "bold"}

        def plot(ax, df, title, color):
            sns.heatmap(df, annot=True, fmt=fmt, cmap=color, ax=ax, annot_kws=kws)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.set_xlabel('Chunk Size')
            ax.set_ylabel('Chunk Overlap')

        plot(axes[0,0], df_ef, 'Explanation: Faithfulness', 'Blues')
        plot(axes[0,1], df_hf, 'Hint: Faithfulness', 'Purples')
        plot(axes[1,0], df_er, 'Explanation: Relevancy', 'Greens')
        plot(axes[1,1], df_hr, 'Hint: Relevancy', 'Oranges')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        plt.close() # 메모리 해제

if __name__ == "__main__":
    data_map = load_all_experiment_data()
    
    df = prepare_dataframe(data_map)
    
    plot_trend_lines(df)
    plot_heatmaps(data_map)