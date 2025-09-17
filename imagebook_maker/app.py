import os
import io
import json
import base64
from typing import List, Dict, Any, Optional

import streamlit as st

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


APP_TITLE = "동화책 챕터 생성 및 이미지 만들기"
DEFAULT_NUM_CHAPTERS = 6


def get_openai_client(api_key: Optional[str]) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. requirements.txt로 설치하세요.")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API 키가 필요합니다. 사이드바 또는 환경변수에 설정하세요.")
    return OpenAI(api_key=key)


def build_structured_prompt(
    raw_text: str,
    num_chapters: int,
    target_age: str,
    tone: str,
    language: str,
) -> List[Dict[str, Any]]:
    system = (
        "You are a skillful children's book editor and visual director. "
        "You split long narratives into well-structured fairy-tale chapters, "
        "then refine the prose for clarity, age-appropriateness, and charm. "
        "For each chapter, you also craft a single cinematic, coherent image prompt "
        "that visually captures the chapter's main event without extreme violence."
    )

    user = (
        f"Input story (any style/language):\n\n{raw_text}\n\n"
        f"Requirements:\n"
        f"- Target age: {target_age}\n"
        f"- Tone/Style: {tone}\n"
        f"- Desired number of chapters: {num_chapters}\n"
        f"- Output language: {language}\n"
        "- Make each chapter self-contained and cohesive.\n"
        "- Keep child-friendly content; avoid graphic content.\n"
        "- Provide a vivid, concrete image prompt for each chapter suitable for a single illustration.\n"
        "- Ensure continuity of characters/settings across chapters.\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "book_title": {"type": "string"},
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chapter_number": {"type": "integer"},
                        "chapter_title": {"type": "string"},
                        "refined_text": {"type": "string"},
                        "image_prompt": {"type": "string"},
                    },
                    "required": [
                        "chapter_number",
                        "chapter_title",
                        "refined_text",
                        "image_prompt",
                    ],
                    "additionalProperties": False,
                },
                "minItems": 1,
            },
        },
        "required": ["book_title", "chapters"],
        "additionalProperties": False,
    }

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
        {
            "role": "user",
            "content": (
                "Return strictly valid JSON matching this JSON Schema. Do not include markdown."
            ),
        },
        {
            "role": "system",
            "content": json.dumps(schema, ensure_ascii=False),
        },
    ]


def split_into_chapters(
    client: Any,
    raw_text: str,
    num_chapters: int,
    target_age: str,
    tone: str,
    language: str,
) -> Dict[str, Any]:
    messages = build_structured_prompt(raw_text, num_chapters, target_age, tone, language)

    # Prefer Responses API with JSON schema, with fallback to chat.completions
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            temperature=0.7,
            max_output_tokens=4000,
            response_format={"type": "json_object"},
        )
        text = response.output_text
    except Exception:
        chat_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        text = chat_resp.choices[0].message.content  # type: ignore

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # last-resort cleanup: extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            raise

    # Basic normalization
    chapters = data.get("chapters", [])
    chapters = sorted(
        chapters,
        key=lambda c: (int(c.get("chapter_number", 0)) if str(c.get("chapter_number", "")).isdigit() else 0),
    )
    data["chapters"] = chapters
    return data


def generate_image(client: Any, prompt: str, size: str = "1024x1024", quality: str = "standard") -> bytes:
    # Explicitly request base64 output; fall back to URL if provided
    img = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        response_format="b64_json",
    )
    data = getattr(img, "data", None)
    if not data or not isinstance(data, list) or not data:
        raise RuntimeError("이미지 응답이 비어 있습니다.")
    first = data[0]
    b64 = getattr(first, "b64_json", None)
    if b64:
        try:
            return base64.b64decode(b64)
        except Exception as decode_err:
            raise RuntimeError(f"이미지 디코딩 실패: {decode_err}")

    # Some backends may return URL instead; download if available
    url = getattr(first, "url", None)
    if url:
        try:
            import requests  # lazy import
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as download_err:
            raise RuntimeError(f"이미지 다운로드 실패: {download_err}")

    raise RuntimeError("이미지 데이터가 없습니다 (b64_json/url 모두 없음).")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📖", layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("설정")
        api_key = st.text_input("OpenAI API Key", type="password", help="입력하지 않으면 환경변수 OPENAI_API_KEY를 사용합니다.")
        num_chapters = st.number_input("챕터 수", min_value=1, max_value=30, value=DEFAULT_NUM_CHAPTERS, step=1)
        target_age = st.selectbox("대상 연령", ["4-6세", "7-9세", "10-12세"], index=1)
        tone = st.selectbox(
            "톤/스타일",
            [
                "따뜻하고 포근한",
                "모험적이고 흥미진진한",
                "차분하고 서정적인",
                "유머러스하고 밝은",
            ],
            index=0,
        )
        language = st.selectbox("출력 언어", ["한국어", "English", "日本語", "中文"], index=0)
        # DALL·E 3 지원 사이즈만 제공
        image_size = st.selectbox("이미지 크기", ["1024x1024", "1024x1792", "1792x1024"], index=0)
        image_quality = st.selectbox("이미지 품질", ["standard", "hd"], index=0,
            help="DALL·E 3는 'standard'와 'hd'만 지원합니다.")

    st.markdown(
        "원하는 책이나 소설의 내용을 아래에 붙여넣고, 버튼을 눌러 동화책 챕터와 이미지를 생성하세요."
    )

    raw_text = st.text_area("입력 텍스트", height=240, placeholder="이곳에 줄거리나 원문을 붙여넣으세요…")

    col_run, col_reset = st.columns([1, 1])
    with col_run:
        run = st.button("챕터 생성 및 이미지 만들기", type="primary")
    with col_reset:
        reset = st.button("초기화")

    if reset:
        for key in [
            "structured_result",
            "generated_images",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    if run and raw_text.strip():
        try:
            client = get_openai_client(api_key)
        except Exception as e:
            st.error(str(e))
            st.stop()

        with st.spinner("텍스트를 동화책 챕터로 재구성하는 중…"):
            try:
                structured = split_into_chapters(
                    client=client,
                    raw_text=raw_text,
                    num_chapters=int(num_chapters),
                    target_age=str(target_age),
                    tone=str(tone),
                    language=str(language),
                )
            except Exception as e:
                st.error(f"챕터 구성 중 오류: {e}")
                st.stop()

            st.session_state["structured_result"] = structured

        chapters = structured.get("chapters", [])
        if not chapters:
            st.warning("생성된 챕터가 없습니다. 입력 텍스트를 늘리거나 설정을 바꿔보세요.")
        else:
            st.success(f"'{structured.get('book_title', '제목 미정')}' 동화책 구조가 준비되었습니다.")

        # Generate images
        generated_images: List[Dict[str, Any]] = []
        progress = st.progress(0, text="이미지 생성 준비…")
        for idx, chapter in enumerate(chapters, start=1):
            prompt = chapter.get("image_prompt", "")
            if not prompt:
                generated_images.append({"chapter_number": idx, "image_bytes": None})
                continue
            progress.progress(min(idx / max(1, len(chapters)), 1.0), text=f"이미지 생성 중… {idx}/{len(chapters)}")
            try:
                image_bytes = generate_image(client, prompt=prompt, size=image_size, quality=image_quality)
            except Exception as e:
                st.warning(f"챕터 {idx} 이미지 생성 실패: {e}")
                image_bytes = None
            generated_images.append({"chapter_number": idx, "image_bytes": image_bytes})

        st.session_state["generated_images"] = generated_images

    # Display results if available
    structured = st.session_state.get("structured_result")
    images = st.session_state.get("generated_images")
    if structured and images:
        st.subheader("생성 결과")
        book_title = structured.get("book_title", "제목 미정")
        st.markdown(f"**동화책 제목**: {book_title}")

        chapters: List[Dict[str, Any]] = structured.get("chapters", [])
        for chapter, img in zip(chapters, images):
            with st.container(border=True):
                st.markdown(f"### {chapter.get('chapter_number', '')}. {chapter.get('chapter_title', '')}")
                st.write(chapter.get("refined_text", ""))
                st.caption("이미지 프롬프트")
                st.code(chapter.get("image_prompt", ""))

                image_bytes = img.get("image_bytes")
                if image_bytes:
                    st.image(image_bytes, caption=f"챕터 {chapter.get('chapter_number')} 일러스트", use_container_width=True)
                    st.download_button(
                        label="이 이미지 저장",
                        data=image_bytes,
                        file_name=f"chapter_{chapter.get('chapter_number', 'X')}.png",
                        mime="image/png",
                    )
                else:
                    st.info("이미지를 생성하지 못했습니다.")

        # Exports
        st.divider()
        st.subheader("내보내기")
        json_bytes = json.dumps(structured, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label="동화책 구조(JSON) 다운로드",
            data=json_bytes,
            file_name="storybook.json",
            mime="application/json",
        )

        # Zip images
        try:
            import zipfile
            from io import BytesIO

            buf = BytesIO()
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for chapter, img in zip(chapters, images):
                    image_bytes = img.get("image_bytes")
                    if image_bytes:
                        filename = f"chapter_{chapter.get('chapter_number', 'X')}.png"
                        zf.writestr(filename, image_bytes)
            buf.seek(0)
            st.download_button(
                label="모든 이미지 ZIP 다운로드",
                data=buf,
                file_name="storybook_images.zip",
                mime="application/zip",
            )
        except Exception as e:  # pragma: no cover
            st.info(f"ZIP 생성 실패: {e}")


if __name__ == "__main__":
    main()


