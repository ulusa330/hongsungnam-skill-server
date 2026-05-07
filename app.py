import os
import re
import json
import numpy as np
from flask import Flask, request, jsonify
from openai import OpenAI
from pathlib import Path

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

BOOK_SOURCE_TYPES = ['book_hong', 'book_bible', 'book_spiritual']
SCHEDULE_KEYWORDS = ['강의 일정', '특강 일정', '다음 강의', '강의 날짜', '다음 특강', '몇월', '몇 월', '다음 특강 언제', '특강 있나요', '특강 있어요', '특강 언제', '특강 있', '특강 일정']
LECTURE_QUERY_KEYWORDS = ['월특강', '특강 요약', '특강요약', '특강영상', '특강 영상', '월 특강', '요약해줘', '요약해 줘', '요약 해줘', '특강을 요약', '특강 내용', '요약정리', '요약 정리', '특강 정리']

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
        data = np.load(embeddings_path)
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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_lecture_filter_indices(metadata, query):
    all_summaries = [(i, m) for i, m in enumerate(metadata) if m.get('source_type') == 'lecture_summary']
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

def search_similar(query, n_results=5):
    if db is None:
        return None
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embedding = np.array(response.data[0].embedding)
    is_lecture_q = any(kw in query for kw in LECTURE_QUERY_KEYWORDS)
    if is_lecture_q:
        filter_indices = get_lecture_filter_indices(db['metadata'], query)
    else:
        excluded_types = set(BOOK_SOURCE_TYPES) | {'lecture_summary'}
        filter_indices = [i for i, m in enumerate(db['metadata']) if m.get('source_type') not in excluded_types]
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

def get_schedule_text():
    if SCHEDULE is None:
        return None
    lecture = SCHEDULE.get('next_lecture', {})
    if lecture.get('status') == 'confirmed':
        date = lecture.get('date', '')
        text = f"다음 특강 일정을 알려드립니다.\n"
        text += f"📅 날짜: {date} ({lecture.get('day_of_week', '')}요일)\n"
        text += f"⏰ 시간: 오후 {lecture.get('time_start', '')} ~ {lecture.get('time_end', '')}\n"
        text += f"📍 장소: {lecture.get('location', '')}\n"
        text += f"💰 회비: {lecture.get('fee', '')}\n"
        text += f"📞 문의: {lecture.get('contact', '')}\n"
        if lecture.get('note'):
            text += f"✏️ 비고: {lecture.get('note', '')}"
        return text
    return None

def generate_answer(query, results):
    is_schedule = any(kw in query for kw in SCHEDULE_KEYWORDS)
    if is_schedule:
        schedule_text = get_schedule_text()
        if schedule_text:
            return schedule_text
        messages = [
            {"role": "system", "content": "당신은 홍성남 마태오 신부입니다. 따뜻하고 친근한 신부님 말투로 답변하세요. 답변은 300자 이내로 간결하게."},
            {"role": "user", "content": query}
        ]
    elif results is None:
        messages = [
            {"role": "system", "content": "당신은 홍성남 마태오 신부입니다. 따뜻하고 친근한 신부님 말투로 답변하세요. 답변은 300자 이내로 간결하게."},
            {"role": "user", "content": query}
        ]
    else:
        context = "\n\n---\n\n".join([
            f"[출처: {m.get('title', '')}]\n{doc}"
            for doc, m in zip(results['documents'], results['metadatas'])
        ])
        messages = [
            {"role": "system", "content": """당신은 홍성남 마태오 신부의 말투로 상담해주는 AI입니다.
- 1인칭으로 답변 ("홍성남 신부님은" 금지)
- 따뜻하고 직설적인 어조
- 답변은 300자 이내로 간결하게
- 상담 요청 시: "저는 현재 성직자 상담만 하고 있습니다. 가톨릭영성심리상담소(02-776-8405)로 문의하세요." """},
            {"role": "user", "content": f"질문: {query}\n\n참고:\n{context}"}
        ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content

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
                "outputs": [{"simpleText": {"text": "죄송합니다. 잠시 후 다시 시도해주세요."}}]
            }
        })

with app.app_context():
    load_db()
    load_schedule()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
