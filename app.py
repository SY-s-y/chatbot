"""DATA 폴더의 PDF를 기반으로 답변하는 Streamlit RAG 챗봇입니다."""

from __future__ import annotations

import re
import os
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# 이 파일이 있는 프로젝트 최상단과 DATA 폴더의 위치를 고정합니다.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"
HISTORY_FILE = PROJECT_DIR / ".chat_history.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def load_pdf_documents(data_dir: Path) -> list[Document]:
    """DATA 폴더의 모든 PDF 페이지를 LangChain Document로 바꿉니다."""
    documents: list[Document] = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            # PDF에 따라 텍스트가 없는 페이지도 있으므로 빈 페이지는 건너뜁니다.
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )

    return documents


def document_fingerprint(data_dir: Path) -> str:
    """PDF가 바뀌면 Streamlit 캐시를 새로 만들기 위한 식별값입니다."""
    return "|".join(
        f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        for path in sorted(data_dir.glob("*.pdf"))
    )


def load_chat_history() -> list[dict]:
    """이전 대화를 로컬 파일에서 읽습니다. 파일이 없으면 빈 목록을 반환합니다."""
    if not HISTORY_FILE.exists():
        return []

    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return history if isinstance(history, list) else []
    except (json.JSONDecodeError, OSError):
        # 기록 파일이 손상되어도 챗봇을 사용할 수 있도록 빈 대화로 시작합니다.
        return []


def save_chat_history(messages: list[dict]) -> None:
    """대화 내용을 UTF-8 JSON 파일로 저장해 새로고침 뒤에도 유지합니다."""
    HISTORY_FILE.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@st.cache_resource(show_spinner="PDF 문서를 읽고 검색 인덱스를 만드는 중입니다...")
def build_vector_store(fingerprint: str, api_key: str) -> tuple[InMemoryVectorStore, int]:
    """문서를 작은 조각으로 나눈 뒤 메모리 벡터 DB에 저장합니다."""
    # fingerprint는 함수 인자에 포함되어 PDF 수정 시 캐시를 갱신하게 합니다.
    _ = fingerprint
    documents = load_pdf_documents(DATA_DIR)
    if not documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 PDF 텍스트를 찾지 못했습니다.")

    # 긴 페이지를 적당한 길이로 나눠야 질문과 관련된 부분을 더 정확히 찾습니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        # 한 조각이 1,000자라 임베딩 한도보다 훨씬 짧습니다.
        # 이 검사를 끄면 tiktoken 인코딩 파일을 별도로 내려받지 않습니다.
        check_embedding_ctx_length=False,
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store, len(documents)


def evidence_sentence(text: str, question: str) -> str:
    """검색된 문단에서 질문과 가장 가까운 한 문장을 골라 보여 줍니다."""
    candidates = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    if not candidates:
        return text.strip()[:350]

    # 조사 등을 완벽히 분석하지 않아도, 두 글자 이상 단어의 겹침으로 유용한 근거를 고릅니다.
    keywords = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", question.lower()))
    best = max(
        candidates,
        key=lambda sentence: sum(keyword in sentence.lower() for keyword in keywords),
    )
    return best[:350] + ("..." if len(best) > 350 else "")


def answer_question(vector_store: InMemoryVectorStore, question: str, api_key: str) -> tuple[str, list[Document]]:
    """질문과 관련된 문서 조각만 LLM에 전달해 답변을 만듭니다."""
    retrieved_documents = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[파일: {doc.metadata['source']} / 페이지: {doc.metadata['page']}]\n{doc.page_content}"
        for doc in retrieved_documents
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 공무원 여비 문서를 안내하는 도우미입니다.
반드시 아래 제공 문서 내용만 근거로 한국어로 답하세요.
문서에 답이 없거나 근거가 부족하면 반드시 '제공된 문서에서 확인할 수 없습니다.'라고 답하세요.
추측, 일반 지식, 문서에 없는 규정은 절대 덧붙이지 마세요.
답변은 이해하기 쉽게 간결하게 작성하세요.""",
            ),
            ("human", "문서 내용:\n{context}\n\n질문: {question}"),
        ]
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)

    # prompt | llm | parser는 현재 LangChain에서 권장하는 LCEL 방식입니다.
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved_documents


def main() -> None:
    """Streamlit 화면을 구성합니다."""
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")

    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("DATA 폴더의 PDF만 검색하여 답변합니다.")

    # session_state는 현재 화면의 대화를 빠르게 유지하고,
    # .chat_history.json은 서버 재시작이나 새로고침 뒤에도 대화를 복원합니다.
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    with st.sidebar:
        st.subheader("대화 관리")
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            save_chat_history([])
            st.rerun()

    if not api_key:
        st.error(".env 파일에 OPENAI_API_KEY를 입력한 뒤 다시 실행해 주세요.")
        st.code("OPENAI_API_KEY=sk-...")
        st.stop()

    if not DATA_DIR.exists():
        st.error("DATA 폴더를 찾을 수 없습니다.")
        st.stop()

    try:
        vector_store, page_count = build_vector_store(document_fingerprint(DATA_DIR), api_key)
    except Exception as error:
        st.error(f"문서 인덱스를 만드는 중 오류가 발생했습니다: {error}")
        st.stop()

    st.caption(f"PDF {page_count}페이지를 검색할 준비가 되었습니다.")

    # 이전 대화 내용을 화면에 다시 표시합니다.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source, page, sentence in message.get("sources", []):
                st.caption(f"출처: {source} (p. {page})")
                st.caption(f"근거 문장: {sentence}")

    question = st.chat_input("여비 규정에 관해 질문해 보세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("관련 문서를 찾고 답변을 작성하는 중입니다..."):
            answer, documents = answer_question(vector_store, question, api_key)
        st.markdown(answer)

        sources = []
        seen_sources: set[tuple[str, int, str]] = set()
        for document in documents:
            source = str(document.metadata["source"])
            page = int(document.metadata["page"])
            sentence = evidence_sentence(document.page_content, question)
            source_item = (source, page, sentence)
            if source_item not in seen_sources:
                seen_sources.add(source_item)
                sources.append(source_item)
                st.caption(f"출처: {source} (p. {page})")
                st.caption(f"근거 문장: {sentence}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    save_chat_history(st.session_state.messages)


if __name__ == "__main__":
    main()
