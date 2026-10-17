"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
from core.types import Chunk
from core.utils import cosine, cosine_many  # noqa: F401 — ตัวช่วยคำนวณความคล้ายของ vector
from student import settings  # noqa: F401 — TOP_K, QUERY_WITH_OPTIONS, HYBRID_WEIGHT, MIN_SCORE
import math
import re


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)

    แนวทาง: ลองรวมคำถามกับตัวเลือกเข้าด้วยกัน เพราะคำตอบมักอยู่ในย่อหน้าที่มีคำของตัวเลือก
    """
    q = (question or "").strip()
    if settings.QUERY_WITH_OPTIONS and options:
        q += "\nตัวเลือก: " + " | ".join(str(o).strip() for o in options if str(o).strip())
    return q or " "


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น

    รับ:  chunks        chunk ทั้งหมดของเอกสารชุดนี้ (chunk.text, chunk.vector, chunk.doc_id)
          query_text    คำค้น (จาก make_query)
          query_vector  vector ของคำค้น (จาก embedding.embed_query) หรือ None ถ้ายังไม่ได้ทำ embedding
          k             จำนวนที่ระบบแนะนำ (5) จะใช้ settings.TOP_K ของตัวเองแทนก็ได้
    คืน:  list ของ (chunk, score) เรียงจากเกี่ยวข้องมากไปน้อย

    แนวทาง:
      - วิธีพื้นฐาน: scores = cosine_many(query_vector, [c.vector for c in chunks]) แล้วเลือก k ตัวที่คะแนนสูงสุด
      - วิธีที่ดีขึ้น: hybrid search คือรวมคะแนน vector กับคะแนนการจับคำ (keyword / BM25) ช่วยมากกับชื่อเฉพาะและตัวเลข
      - ระวังกรณีที่ chunk.vector หรือ query_vector เป็น None
    """
    if not chunks or k <= 0:
        return []
    limit = max(1, min(int(k), int(getattr(settings, "TOP_K", k))))
    vectors = [c.vector for c in chunks]
    vector_scores: list[float] | None = None
    if query_vector is not None and all(v is not None for v in vectors):
        try:
            vector_scores = [float(s) for s in cosine_many(query_vector, vectors)]
        except (TypeError, ValueError, ZeroDivisionError):
            vector_scores = None
    terms = _terms(query_text)
    lex_scores = []
    for chunk in chunks:
        doc_terms = _terms(chunk.text)
        if not terms or not doc_terms:
            lex_scores.append(0.0)
            continue
        counts: dict[str, int] = {}
        for term in doc_terms:
            counts[term] = counts.get(term, 0) + 1
        # Length-normalized term coverage; exact identifiers and numbers remain useful.
        matched = sum(min(3, counts.get(t, 0)) for t in set(terms))
        lex_scores.append(matched / max(1, len(set(terms))) / (1 + math.log1p(len(doc_terms)) / 8))
    mx = max(lex_scores, default=0.0)
    lex_scores = [s / mx if mx else 0.0 for s in lex_scores]
    weight = max(0.0, min(1.0, float(getattr(settings, "HYBRID_WEIGHT", 0.0))))
    if vector_scores is None:
        scores = lex_scores
    else:
        # Shift cosine scores to [0, 1] so hybrid weighting behaves consistently.
        vec_norm = [(s + 1.0) / 2.0 for s in vector_scores]
        scores = [(1 - weight) * v + weight * l for v, l in zip(vec_norm, lex_scores)]
    minimum = getattr(settings, "MIN_SCORE", None)
    ranked = sorted(enumerate(scores), key=lambda item: (item[1], -item[0]), reverse=True)
    result = []
    for i, score in ranked:
        if minimum is not None and score < float(minimum):
            continue
        result.append((chunks[i], float(score)))
        if len(result) >= limit:
            break
    return result


def _terms(text: str) -> list[str]:
    """Tokenize whitespace text and overlapping Thai character n-grams."""
    raw = re.findall(r"[\w.+#/-]+", (text or "").lower(), flags=re.UNICODE)
    out: list[str] = []
    for token in raw:
        if re.search(r"[ก-๙]", token):
            compact = re.sub(r"\s+", "", token)
            if len(compact) <= 2:
                out.append(compact)
            else:
                out.extend(compact[i:i + 2] for i in range(len(compact) - 1))
        else:
            out.append(token)
    return out
