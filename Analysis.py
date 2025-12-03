import pandas as pd
import re

def calculate_final_quiz_difficulty(file_path):
    # 1. 데이터 로드
    df = pd.read_csv(file_path)

    # 2. 컬럼명 정리 (Q1, Q2... 형식으로 자동 변환)
    new_cols = {
        df.columns[0]: 'Timestamp',
        df.columns[1]: 'Proficiency',
        df.columns[2]: 'Residence',
        df.columns[-1]: 'Feedback'
    }
    
    # 문항 수 계산 (전체 컬럼 - 메타 정보 컬럼 4개) // 2
    num_questions = (len(df.columns) - 4) // 2
    
    for i in range(1, num_questions + 1):
        ans_idx = 3 + (i-1)*2
        diff_idx = 4 + (i-1)*2
        new_cols[df.columns[ans_idx]] = f'Q{i}_Answer'
        new_cols[df.columns[diff_idx]] = f'Q{i}_Difficulty'
        
    df.rename(columns=new_cols, inplace=True)

    # 3. 데이터 전처리
    # 3-1. 난이도 숫자 추출 (예: "(2) 쉬움" -> 2)
    def extract_difficulty(val):
        if pd.isna(val): return None
        match = re.search(r'\((\d)\)', str(val))
        return int(match.group(1)) if match else None

    # 3-2. 한국어 능력 그룹화
    def simplify_prof(val):
        if pd.isna(val): return "Unknown"
        if "Advanced" in val: return "Advanced"
        if "Intermediate" in val: return "Intermediate"
        if "Beginner" in val: return "Beginner"
        return "Unknown"

    for i in range(1, num_questions + 1):
        df[f'Q{i}_Difficulty_Num'] = df[f'Q{i}_Difficulty'].apply(extract_difficulty)
    
    df['Proficiency_Group'] = df['Proficiency'].apply(simplify_prof)

    # 4. 가중치 및 정답 키 설정
    # 고급: 3점, 중급: 2점, 초급: 1점 (신뢰도 가중치)
    prof_weights = {'Advanced': 3, 'Intermediate': 2, 'Beginner': 1, 'Unknown': 0}
    df['Prof_Weight'] = df['Proficiency_Group'].map(prof_weights)

    # 고급 사용자의 최빈값(Mode)을 정답으로 추론
    advanced_users = df[df['Proficiency_Group'] == 'Advanced']
    answer_key = {}
    for i in range(1, num_questions + 1):
        ans_col = f'Q{i}_Answer'
        if not advanced_users.empty:
            mode_val = advanced_users[ans_col].mode()
            answer_key[ans_col] = mode_val[0] if not mode_val.empty else df[ans_col].mode()[0]
        else:
            answer_key[ans_col] = df[ans_col].mode()[0]

    # 5. PADI 지수 및 1-5점 척도 계산
    results = []
    total_weight = df['Prof_Weight'].sum()

    for i in range(1, num_questions + 1):
        q_ans_col = f'Q{i}_Answer'
        q_diff_col = f'Q{i}_Difficulty_Num'
        
        # (1) 보정된 오답률 (Weighted Error Rate)
        is_wrong = (df[q_ans_col] != answer_key[q_ans_col]).astype(int)
        weighted_error_rate = (is_wrong * df['Prof_Weight']).sum() / total_weight
        
        # (2) 보정된 주관적 난이도 (Weighted Subjective Difficulty)
        # 1~5점 척도를 0~1로 정규화: (점수 - 1) / 4
        norm_rating = (df[q_diff_col] - 1) / 4
        weighted_subj_diff = (norm_rating * df['Prof_Weight']).sum() / total_weight
        
        # (3) Raw PADI Score (0~1)
        raw_padi = (weighted_error_rate + weighted_subj_diff) / 2
        
        # (4) 최종 난이도 (1~5점 척도 변환)
        # 공식: 1 + (Raw_PADI * 4)
        difficulty_1to5 = 1 + (raw_padi * 4)
        
        results.append({
            'Question': f'Q{i}',
            'Weighted_Error_Rate': round(weighted_error_rate, 4),
            'Weighted_Subj_Diff_Norm': round(weighted_subj_diff, 4),
            'Raw_PADI': round(raw_padi, 4),
            'Final_Difficulty_1to5': round(difficulty_1to5, 2)
        })

    # 결과 데이터프레임 생성 및 정렬
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Final_Difficulty_1to5', ascending=False)
    
    return result_df

# 코드 실행 (예시)
if __name__ == "__main__":
    final_df = calculate_final_quiz_difficulty('Datasets\Data.csv')
    print(final_df[['Question', 'Final_Difficulty_1to5', 'Weighted_Error_Rate']].to_string(index=False))