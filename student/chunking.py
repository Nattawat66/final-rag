"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
import re
from student import settings  # noqa: F401


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ

    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)

    แนวทาง:
      - แยกตามหัวข้อ markdown (บรรทัดที่ขึ้นต้นด้วย #) แล้วใส่ชื่อหัวข้อไว้ต้น chunk ด้วย
      - หัวข้อที่ยาวเกินไปให้ตัดย่อยเป็นช่วง ๆ (เช่น 600–1,000 ตัวอักษร) และซ้อนกันเล็กน้อย (overlap)
      - พยายามอย่าตัดกลางตาราง กลาง code block หรือกลางประโยค
    ตัวอย่าง:
      split("doc1", "# A\\nเนื้อหา A\\n# B\\nเนื้อหา B") -> ["A\\nเนื้อหา A", "B\\nเนื้อหา B"]
    """
    if not isinstance(text, str) or not text.strip():
        return []

    size = max(1, int(settings.CHUNK_SIZE))
    overlap = max(0, min(int(settings.CHUNK_OVERLAP), size - 1))
    sections: list[tuple[str, str]] = []
    if settings.SPLIT_BY_HEADING:
        heading_stack: list[str] = []
        body: list[str] = []
        current_title = ""
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#") and len(stripped) > 1 and stripped[1] in " \t":
                if body:
                    sections.append((current_title, "\n".join(body).strip()))
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped[level:].strip()
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(title)
                current_title = " > ".join(heading_stack)
                body = []
            else:
                body.append(line)
        if body or not sections:
            sections.append((current_title, "\n".join(body).strip()))
    else:
        sections = [("", text.strip())]

    chunks: list[str] = []
    for title, body in sections:
        if not body and not title:
            continue
        prefix = f"{title}\n" if title and settings.INCLUDE_HEADING else ""
        content = prefix + body
        if len(content) <= size:
            if content.strip():
                chunks.append(content.strip())
            continue

        start = 0
        while start < len(content):
            end = min(start + size, len(content))
            if end < len(content):
                # End at a nearby paragraph, table-row, or sentence boundary.
                candidates = [content.rfind(mark, start + size // 2, end)
                              for mark in ("\n", "。", ".", "!", "?")]
                boundary = max(candidates)
                if boundary > start:
                    end = boundary + (1 if content[boundary] != "\n" else 0)
            piece = content[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(content):
                break
            next_start = max(start + 1, end - overlap)
            # In Thai there are few spaces between words; start overlap at a
            # markdown line boundary when possible instead of inside a table row.
            line_end = content.find("\n", next_start, end)
            start = line_end + 1 if line_end >= 0 else next_start
    return chunks or [text.strip()]
