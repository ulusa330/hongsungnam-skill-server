import os
import re
import json
import numpy as np
from flask import Flask, request, jsonify
from openai import OpenAI
from pathlib import Path
from datetime import date as dt_date
import threading

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

FALLBACK_MSG = "죄송합니다. 잠시 후 다시 질문해 주세요.\n📞 문의: 02-776-8405 (오전 11시~오후 4시)"

NEWSPAPER_FILTERS = {
    '중앙일보': ['중앙일보', '중앙'],
    '가톨릭신문': ['가톨릭신문', '가톨릭 신문'],
    '경향신문': ['경향신문', '경향 신문', '경향'],
}
YOUTUBE_SERIES_FILTERS = {
    '맹모닝 상담소': ['맹모닝', '맹모닝 상담소', '맹모닝상담소'],
    '마태오묵상집': ['마태오묵상', '마태오 묵상', '마태오묵상집'],
    '요한묵상집': ['요한묵상', '요한 묵상', '요한묵상집'],
    '창세기묵상집': ['창세기묵상', '창세기 묵상', '창세기묵상집'],
    '10분 강의': ['10분 강의', '10분강의', '십분 강의'],
    '톡쏘는 영성심리': ['톡쏘는', '영성심리'],
    '사수영': ['사수영', '사제와 수도자', '사제와수도자'],
    'cpbc': ['cpbc특강', 'cpbc뉴스', 'cpbc', '가톨릭 청춘어게인', '청춘어게인'],
}
MONTHLY_LECTURE_PATTERN = re.compile(r'\[?\d{6}\]?')
SOURCE_TYPE_FILTERS = {
    'column': ['칼럼', '신문', '기고', '신문 칼럼', '신문칼럼'],
    'youtube': ['유튜브', '영상', '동영상', '강의 영상'],
}
BOOK_SOURCE_TYPES = ['book_hong', 'book_bible', 'book_spiritual']
SCHEDULE_KEYWORDS = ['강의 일정', '특강 일정', '다음 강의', '강의 날짜', '다음 특강', '몇월', '몇 월', '다음 특강 언제', '특강 있나요', '특강 있어요', '특강 언제', '특강 있', '특강 일정']
LECTURE_QUERY_KEYWORDS = ['월특강', '특강 요약', '특강요약', '특강영상', '특강 영상', '월 특강', '특강 보고', '특강 알려', '요약해줘', '요약해 줘', '요약 해줘', '특강을 요약', '특강 내용', '요약정리', '요약 정리', '특강을 보여', '특강 정리']

db = None
SCHEDULE = None

def load_schedule():
    global SCHEDULE
    schedule_path = Path("./schedule.json")
    if not schedule_path.exists():
        SCHEDULE = None
        return
    try:
        with open(schedule_path, 'r', encoding='utf-8') as f:
            SCHEDULE = json.load(f)
        try:
            _next = SCHEDULE.get("next_lecture", {})
            if _next.get("status") == "confirmed":
                _month = str(int(_next.get("date", "")[5:7]))
                SCHEDULE_KEYWORDS.append(f"{_month}월 특강")
                SCHEDULE_KEYWORDS.append(f"{_month}월에도 특강")
                SCHEDULE_KEYWORDS.append(f"{_month}월 일정")
        except Exception:
            pass
        print("schedule.json 로드 완료")
    except Exception as e:
        print(f"schedule.json 로드 오류: {e}")
        SCHEDULE = None

def load_db():
    global db
    embeddings_path = Path("./vectordb_홍성남신부/embeddings.npz")
    metadata_path = Path("./vectordb_홍성남신부/metadata.json")
    if not embeddings_path.exists() or not metadata_path.exists():
        print("VectorDB 파일 없음")
        return None
    try:
        data = np.load(embeddings_path, mmap_mode='r')
        embeddings = data['embeddings']
        with open(metadata_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        db = {
            'embeddings': embeddings,
            'metadata': saved.get('metadata', []),
            'documents': saved.get('documents', []),
        }
        print(f"VectorDB 로드 완료: {len(embeddings)}개")
        return db
    except Exception as e:
        print(f"VectorDB 로드 오류: {e}")
        return None

def get_schedule_text():
    if SCHEDULE is None:
        return None
    today = dt_date.today()
    lecture = SCHEDULE.get('next_lecture', {})
    regular = SCHEDULE.get('regular_schedule', {})
    if lecture.get('status') == 'confirmed':
        date = lecture.get('date', '')
        year = date[:4] if date else ''
        month = str(int(date[5:7])) if date else ''
        day = str(int(date[8:10])) if date else ''
        dow = lecture.get('day_of_week', '')
        t_start = lecture.get('time_start', '')
        t_end = lecture.get('time_end', '')
        h_start = int(t_start.split(":")[0]) if t_start else 0
        period_start = "오후" if h_start >= 12 else "오전"
        h_start_12 = h_start - 12 if h_start > 12 else h_start
        h_end = int(t_end.split(":")[0]) if t_end else 0
        h_end_12 = h_end - 12 if h_end > 12 else h_end
        time_str = f"{period_start} {h_start_12}시~{h_end_12}시"
        location = lecture.get('location', '')
        fee = lecture.get('fee', '')
        contact = lecture.get('contact', '')
        title = lecture.get('title', '영성심리특강')
        note = lecture.get('note', '')
        try:
            lecture_date = dt_date(int(year), int(month), int(day))
        except Exception:
            lecture_date = None
        if lecture_date and today > lecture_date:
            return f"아쉽게도 {month}월 {title}은 이미 종료되었습니다. 다음 달 강의 일정은 아직 등록되지 않았습니다. 문의: {contact}"
        text = f"다음 특강 일정을 알려드립니다.\n"
        text += f"📅 날짜: {year}년 {month}월 {day}일 ({dow}요일)\n"
        text += f"⏰ 시간: {time_str}\n"
        text += f"📍 장소: {location}\n"
        text += f"💰 회비: {fee}\n"
        text += f"📞 문의: {contact} (오전 11시~오후 4시)\n"
        if note:
            text += f"✏️ 비고: {note}"
        return text
    else:
        pattern = regular.get('pattern', '매월 셋째 주 토요일')
        time = regular.get('time', '오후 3시')
        location = regular.get('location', '가톨릭회관')
        contact = regular.get('contact', '02-776-8405')
        return f"현재 확정된 강의 일정이 없습니다. 정기적으로 {pattern} {time}, {location}에서 진행됩니다. 문의: {contact}"

def get_schedule_prompt_text():
    if SCHEDULE is None:
        return ""
    today = dt_date.today()
    lecture = SCHEDULE.get('next_lecture', {})
    regular = SCHEDULE.get('regular_schedule', {})
    if lecture.get('status') == 'confirmed':
        date = lecture.get('date', '')
        year = date[:4] if date else ''
        month = str(int(date[5:7])) if date else ''
        day = str(int(date[8:10])) if date else ''
        dow = lecture.get('day_of_week', '')
        t_start = lecture.get('time_start', '')
        t_end = lecture.get('time_end', '')
        h_start = int(t_start.split(":")[0]) if t_start else 0
        period_start = "오후" if h_start >= 12 else "오전"
        h_start_12 = h_start - 12 if h_start > 12 else h_start
        h_end = int(t_end.split(":")[0]) if t_end else 0
        h_end_12 = h_end - 12 if h_end > 12 else h_end
        time_str = f"{period_start} {h_start_12}시~{h_end_12}시"
        location = lecture.get('location', '')
        fee = lecture.get('fee', '')
        contact = lecture.get('contact', '')
        title = lecture.get('title', '영성심리특강')
        try:
            lecture_date = dt_date(int(year), int(month), int(day))
        except Exception:
            lecture_date = None
        if lecture_date and today > lecture_date:
            return f"[강의 일정 규칙] 아쉽게도 {month}월 {title}은 이미 종료되었습니다. 문의: {contact}. 중요: 과거 영상이나 자막에 언급된 날짜의 강의 일정은 절대 안내하지 말 것."
        else:
            return f"[강의 일정 규칙] 다음 {title}: {year}년 {month}월 {day}일({dow}) {time_str}, {location}. 회비 {fee}. 문의 {contact}. 중요: 과거 영상이나 자막에 언급된 다른 날짜의 강의 일정은 절대 안내하지 말 것."
    else:
        pattern = regular.get('pattern', '매월 셋째 주 토요일')
        time = regular.get('time', '오후 3시')
        location = regular.get('location', '가톨릭회관')
        contact = regular.get('contact', '02-776-8405')
        return f"[강의 일정 규칙] 현재 확정된 강의 일정이 없습니다. 정기적으로 {pattern} {time}, {location}에서 진행됩니다. 문의: {contact}. 중요: 과거 영상이나 자막에 언급된 날짜의 강의 일정은 절대 안내하지 말 것."

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_lecture_filter_indices(query):
    all_summaries = [(i, m) for i, m in enumerate(db['metadata']) if m.get('source_type') == 'lecture_summary']
    ym_match = re.search(r'(20)?(\d{2})년도?\s*(\d{1,2})월', query)
    if ym_match:
        yy = ym_match.group(2)
        mm = ym_match.group(3).zfill(2)
        year_month_key = f"{yy}{mm}"
        matched = [i for i, m in all_summaries if year_month_key in m.get('filename', '')]
        return matched if matched else [i for i, m in all_summaries]
    month_match = re.search(r'(\d{1,2})월', query)
    if month_match:
        mm = month_match.group(1).zfill(2)
        month_files = [(i, m) for i, m in all_summaries
                       if m.get('filename', '').endswith('_월특강_요약.md')
                       and m.get('filename', '')[2:4] == mm]
        if month_files:
            month_files.sort(key=lambda x: x[1].get('filename', ''), reverse=True)
            latest_yymm = month_files[0][1].get('filename', '')[:4]
            return [i for i, m in month_files if m.get('filename', '').startswith(latest_yymm)]
    return [i for i, m in all_summaries]

def detect_source_filter(query):
    query_lower = query.lower().strip()
    if any(kw in query for kw in SCHEDULE_KEYWORDS):
        return None
    for newspaper, keywords in NEWSPAPER_FILTERS.items():
        for kw in keywords:
            if kw in query_lower:
                return {'type': 'newspaper', 'value': newspaper}
    if any(kw in query for kw in LECTURE_QUERY_KEYWORDS):
        return {'type': 'monthly_lecture', 'value': 'monthly'}
    for series, keywords in YOUTUBE_SERIES_FILTERS.items():
        for kw in keywords:
            if kw in query_lower:
                return {'type': 'youtube_series', 'value': series}
    for source_type, keywords in SOURCE_TYPE_FILTERS.items():
        for kw in keywords:
            if kw in query_lower:
                return {'type': 'source_type', 'value': source_type}
    return None

def apply_filter(source_filter):
    if source_filter is None:
        return None
    metadata = db['metadata']
    valid_indices = []
    for i, meta in enumerate(metadata):
        filter_type = source_filter['type']
        filter_value = source_filter['value']
        if filter_type == 'newspaper':
            if meta.get('source_type') == 'column' and filter_value in meta.get('newspaper', ''):
                valid_indices.append(i)
        elif filter_type == 'youtube_series':
            if meta.get('source_type', 'youtube') == 'youtube':
                title = meta.get('title', '')
                series_keywords = YOUTUBE_SERIES_FILTERS.get(filter_value, [filter_value])
                if any(kw in title for kw in series_keywords) or filter_value in title:
                    valid_indices.append(i)
        elif filter_type == 'monthly_lecture':
            source_type = meta.get('source_type', 'youtube')
            if source_type == 'lecture_summary':
                valid_indices.append(i)
            elif source_type == 'youtube':
                if MONTHLY_LECTURE_PATTERN.search(meta.get('title', '')):
                    valid_indices.append(i)
        elif filter_type == 'source_type':
            src = meta.get('source_type', 'youtube')
            if src == filter_value or (filter_value == 'youtube' and src == 'lecture_summary'):
                valid_indices.append(i)
    return valid_indices

def search_similar(query, n_results=3):
    if db is None:
        return None
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embedding = np.array(response.data[0].embedding)
    is_lecture_q = any(kw in query for kw in LECTURE_QUERY_KEYWORDS)
    if is_lecture_q:
        filter_indices = get_lecture_filter_indices(query)
    else:
        source_filter = detect_source_filter(query)
        filter_indices = apply_filter(source_filter)
        excluded_types = set(BOOK_SOURCE_TYPES)
        normal_indices = [i for i, m in enumerate(db['metadata']) if m.get('source_type') not in excluded_types]
        normal_set = set(normal_indices)
        if filter_indices is not None:
            filter_indices = [i for i in filter_indices if i in normal_set]
        else:
            filter_indices = normal_indices
    if not filter_indices:
        return None
    filter_indices = np.array(filter_indices)
    filtered_embeddings = db['embeddings'][filter_indices]
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in filtered_embeddings])
    top_local = np.argsort(similarities)[::-1][:n_results]
    top_indices = filter_indices[top_local]
    top_sims = similarities[top_local]
    return {
        'documents': [db['documents'][i] for i in top_indices],
        'metadatas': [db['metadata'][i] for i in top_indices],
        'similarities': [float(s) for s in top_sims],
    }

def generate_answer(query, results):
    is_schedule = any(kw in query for kw in SCHEDULE_KEYWORDS)
    if is_schedule:
        schedule_text = get_schedule_text()
        if schedule_text:
            return schedule_text

    system_prompt = f"""당신은 홍성남 마태오 신부의 말투와 관점으로 직접 상담해 주는 AI입니다.

[나는 누구인가 — 홍성남 마태오 신부 프로필]
- 이름: 홍성남 마태오
- 소속: 천주교 서울대교구, 특수사목
- 현 소임: 서울대교구 가톨릭영성심리상담소 소장
- 사제 서품: 1987년 2월 6일
- 방송: cpbc TV, 유튜브 '톡쏘는 영성심리' 채널
- 스타일: 심리 상담과 영성지도를 결합, 직설적이고 현실적인 언어 사용

[나의 저서]
최근·대표작: 「끝까지 나를 사랑하는 마음」「나는 생각보다 괜찮은 사람」「거꾸로 보는 종교」「혼자서 마음을 치유하는 법」「내 마음이 어때서」「나로 사는 걸 깜빡했어요」「챙기고 사세요」

[말투 규칙]
- "홍성남 신부님은 ~라고 말씀하셨습니다" 절대 금지 — 1인칭으로만 말하세요
- 따뜻하면서도 직설적이고 톡 쏘는 어조
- 답변은 300자 이내로 매우 간결하게

[상담 안내 규칙]
- 상담 요청 시 환영 문구 절대 금지
- 바로 핵심만: "저는 현재 성직자 상담만 하고 있어서 일반 신자분들과 개인 상담은 어렵습니다."
- 전문 상담: 가톨릭영성심리상담소 (02-776-8405, 오전 11시~오후 4시)

[시제 규칙]
- 과거 특강 내용 답변 시 반드시 과거형
- 미래형 절대 금지

[월특강 요약 답변 규칙]
- 컨텍스트에 월특강 요약이 제공되면 반드시 그 내용으로 답변. "정보가 없다" 절대 금지.
- 유튜브 채널(https://youtube.com/@fr.hongsungnam)에서 영상 확인 가능하다고 안내.

{get_schedule_prompt_text()}"""

    if results and results.get('documents'):
        context_parts = []
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            title = meta.get('title', '제목 미상')
            source_type = meta.get('source_type', 'youtube')
            if source_type == 'column':
                source_label = f"신문 칼럼 - {meta.get('newspaper', '신문')}"
            elif source_type == 'lecture_summary':
                source_label = "월특강 요약"
            else:
                source_label = "유튜브 강의"
            context_parts.append(f"[출처 {i+1}: {title} ({source_label})]\n{doc}")
        context = "\n\n---\n\n".join(context_parts)
        user_content = f"질문: {query}\n\n참고할 내용:\n{context}\n\n위 내용을 바탕으로 답변해 주세요."
    else:
        user_content = f"질문: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    result = [None]
    def call_gpt():
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )
            result[0] = response.choices[0].message.content
        except Exception as e:
            print(f"GPT 오류: {e}")

    thread = threading.Thread(target=call_gpt)
    thread.start()
    thread.join(timeout=4.0)

    if result[0] is None:
        return FALLBACK_MSG

    answer = result[0]

    if results and results.get('metadatas') and not is_schedule:
        seen_titles = set()
        links = []
        for meta in results['metadatas']:
            title = meta.get('title', '')
            url = meta.get('url', '')
            source_type = meta.get('source_type', 'youtube')
            if title and url and title not in seen_titles and source_type == 'youtube':
                seen_titles.add(title)
                links.append(f"• {title}\n  {url}")
            if len(links) >= 2:
                break
        if links:
            answer += "\n\n📹 관련 영상:\n" + "\n".join(links)

    return answer

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "톡쏘는 영성심리 스킬 서버"})

@app.route('/skill', methods=['GET', 'POST'])
def skill():
    if request.method == 'GET':
        return jsonify({"status": "ok", "message": "skill endpoint"})
    try:
        body = request.get_json()
        user_msg = body.get('userRequest', {}).get('utterance', '')
        if not user_msg:
            return jsonify({
                "version": "2.0",
                "template": {"outputs": [{"simpleText": {"text": "질문을 입력해주세요."}}]}
            })
        results = search_similar(user_msg)
        answer = generate_answer(user_msg, results)
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": answer}}]
            }
        })
    except Exception as e:
        print(f"오류: {e}")
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": FALLBACK_MSG}}]
            }
        })

with app.app_context():
    load_db()
    load_schedule()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
