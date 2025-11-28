import os
import wikipedia
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def wikipedia_to_single_pdf(keywords, filename="wikipedia_collection.pdf", lang="ko"):
    #1. 한글 폰트 등록
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
    pdfmetrics.registerFont(TTFont("Malgun", font_path))

    #2. 기본 스타일 정의
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="KTitle", fontName="Malgun", fontSize=20, spaceAfter=12))
    styles.add(ParagraphStyle(name="KHeading", fontName="Malgun", fontSize=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="KBody", fontName="Malgun", fontSize=11, leading=18))

    wikipedia.set_lang(lang)
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []

    for i, keyword in enumerate(keywords):
        try:
            page = wikipedia.page(keyword)
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[{keyword}] 여러 결과 중 첫 번째 사용: {e.options[0]}")
            page = wikipedia.page(e.options[0])
        except wikipedia.exceptions.PageError:
            print(f"[{keyword}] 페이지를 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 제목
        story.append(Paragraph(page.title, styles["KTitle"]))
        story.append(Spacer(1, 0.25 * inch))

        # 요약
        story.append(Paragraph("요약:", styles["KHeading"]))
        try:
            summary_text = wikipedia.summary(keyword, sentences=5)
        except:
            summary_text = "요약 정보를 가져올 수 없습니다."
        story.append(Paragraph(summary_text, styles["KBody"]))
        story.append(Spacer(1, 0.3 * inch))

        # 본문
        story.append(Paragraph("본문:", styles["KHeading"]))
        story.append(Paragraph(page.content.replace("\n", "<br />"), styles["KBody"]))

        # 페이지 구분
        if i < len(keywords) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"✅ '{filename}' 파일로 저장 완료! ({len(keywords)}개의 키워드 포함)")

def extract_choices_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 모든 choice들을 하나의 리스트로 모으기
    all_choices = []
    for item in data:
        all_choices.extend(item["choices"])

    # 중복 제거
    unique_choices = sorted(set(all_choices))
    return unique_choices

# 1. 현재 이 스크립트 파일(MakeDB.py)이 있는 디렉토리를 기준으로 삼습니다.
script_dir = Path(__file__).parent 

# 2. 검색을 시작할 상위 폴더 경로를 만듭니다. (Dataset/Culture)
search_dir = script_dir / 'CLIcK' /'Dataset' / 'Culture' 

# 3. search_dir와 그 모든 하위 폴더에서 '*.json' 패턴의 파일을 찾습니다.
#    rglob은 제너레이터(generator)이므로 list()로 감싸서 리스트로 만듭니다.
json_file_paths = list(search_dir.rglob('*.json'))

print(extract_choices_from_json(json_file_paths[0]))