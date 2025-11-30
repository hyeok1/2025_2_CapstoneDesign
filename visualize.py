import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 1. 임베딩 모델/사이즈 비교 (막대그래프) 함수
# ==========================================
def plot_embedding_performance(labels, faith_scores, ans_rel_scores, output_filename='rag_embedding_chart.png'):
    """
    임베딩 사이즈(또는 모델)에 따른 성능 변화를 그룹 막대그래프로 시각화합니다.
    """
    print(f"Generating Bar Chart: {output_filename}...")
    
    x = np.arange(len(labels))  # 라벨 위치
    width = 0.35  # 막대 너비

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 막대 그리기
    rects1 = ax.bar(x - width/2, faith_scores, width, label='Faithfulness', color='skyblue', alpha=0.9)
    rects2 = ax.bar(x + width/2, ans_rel_scores, width, label='Answer Relevancy', color='lightcoral', alpha=0.9)

    # 차트 꾸미기
    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_xlabel('Embedding Model / Size')
    ax.set_title('RAG Performance by Embedding Size')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15) # 여유 공간 확보
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 막대 위에 점수 표시
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(output_filename)
    print("Done.\n")


# ==========================================
# 2. 청크 설정 최적화 (히트맵) 함수
# ==========================================
def plot_chunking_heatmaps(sizes, overlaps, faith_data, ans_rel_data, output_filename='rag_chunking_heatmap.png'):
    """
    Chunk Size(X축)와 Overlap(Y축)에 따른 성능 변화를 히트맵으로 시각화합니다.
    [수정] Y축이 아래에서 위로 갈수록 커지도록 설정합니다.
    """
    print(f"Generating Heatmaps: {output_filename}...")

    # 데이터프레임 생성
    # (행 인덱스가 overlaps이므로, heatmap은 위에서부터 overlaps[0], overlaps[1]... 순으로 그립니다)
    df_faith = pd.DataFrame(faith_data, index=overlaps, columns=sizes)
    df_ans_rel = pd.DataFrame(ans_rel_data, index=overlaps, columns=sizes)

    # 그래프 설정 (1행 2열)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 2-1. Faithfulness 히트맵
    sns.heatmap(df_faith, annot=True, fmt=".3f", cmap="Blues", ax=axes[0],
                vmin=0.5, vmax=1.0)
    axes[0].set_title('Faithfulness Score')
    axes[0].set_xlabel('Chunk Size')
    axes[0].set_ylabel('Chunk Overlap')
    
    # [핵심 수정] Y축 반전 (아래에서 위로 커지게)
    axes[0].invert_yaxis()

    # 2-2. Answer Relevancy 히트맵
    sns.heatmap(df_ans_rel, annot=True, fmt=".3f", cmap="Greens", ax=axes[1],
                vmin=0.5, vmax=1.0)
    axes[1].set_title('Answer Relevancy Score')
    axes[1].set_xlabel('Chunk Size')
    axes[1].set_ylabel('Chunk Overlap')
    
    # [핵심 수정] Y축 반전 (아래에서 위로 커지게)
    axes[1].invert_yaxis()

    plt.suptitle('RAG Performance by Chunking Strategy', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print("Done.\n")


# ==========================================
# 3. 메인 실행부 (데이터 입력)
# ==========================================
if __name__ == "__main__":
    
    # --- [데이터 1] 임베딩 모델 비교 데이터 ---
    emb_labels = ['text-embedding-3-small\n(1536 dim)', 'text-embedding-3-large\n(3072 dim)']
    emb_faithfulness = [0.85, 0.94]      # 실제 실험 결과 입력
    emb_relevancy = [0.88, 0.96]         # 실제 실험 결과 입력

    # 막대그래프 생성 실행
    plot_embedding_performance(emb_labels, emb_faithfulness, emb_relevancy)


    # --- [데이터 2] 청크 전략 비교 데이터 ---
    # 실험한 설정값들
    chunk_sizes = [300, 500, 1000] 
    chunk_overlaps = [50, 100, 200]

    # Faithfulness 결과 행렬 (행: Overlap, 열: Size)
    # 예: faith_matrix[0][0]은 Overlap=50, Size=300일 때의 점수
    faith_matrix = np.array([
        [0.82, 0.88, 0.85],  # Overlap 50
        [0.84, 0.92, 0.89],  # Overlap 100
        [0.83, 0.90, 0.86]   # Overlap 200
    ])

    # Answer Relevancy 결과 행렬
    ans_rel_matrix = np.array([
        [0.80, 0.85, 0.81],  # Overlap 50
        [0.82, 0.89, 0.83],  # Overlap 100
        [0.81, 0.87, 0.82]   # Overlap 200
    ])

    # 히트맵 생성 실행
    plot_chunking_heatmaps(chunk_sizes, chunk_overlaps, faith_matrix, ans_rel_matrix)
    
    print("모든 시각화 파일이 저장되었습니다.")