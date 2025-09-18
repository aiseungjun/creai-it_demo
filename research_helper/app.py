# -*- coding: utf-8 -*-
"""
Standalone Streamlit app generated from demo.ipynb.
- All core logic is inlined below (cleaned of Jupyter magics).
- UI simply calls `run_pipeline` (first turn) and `ask_followup_agent` (follow-ups).
"""
import os, io, contextlib
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="논문 도우미 챗봇 (Demo Frontend)", page_icon="📚", layout="wide")

# --- Sidebar: API Key ---
st.sidebar.header("⚙️ 설정")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Initial assistant greeting
INIT_GREETING = "안녕세하세요, 궁금한 연구 주제를 알려주시면 관련 논문과 연구 동향을 정리해드릴게요!"
st.title("논문 도우미 챗봇 (Demo Frontend)")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": INIT_GREETING}]
if "first_query_done" not in st.session_state:
    st.session_state.first_query_done = False



# === BEGIN DEMO CODE (auto-inlined) ===
# ---- demo cell 0 ----
# %pip install -qU langchain langchain-community langchain-openai chromadb arxiv tiktoken pymupdf #opentelemetry-sdk==1.31.0


# ---- demo cell 1 ----
import os
# os.environ["OPENAI_API_KEY"] =  ## PUT YOUR API KEY HERE! e.g. 'sk-123..'  # (commented to use Streamlit sidebar API key)
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
CHROMA_DIR = "./chroma_arxiv"
COLLECTION_NAME = "arxiv_top_tier"


# ---- demo cell 2 ----
from langchain_community.retrievers import ArxivRetriever
import re
import arxiv
import fitz

def build_top_tier_query(user_query: str) -> str:
    '''build query for retriever. but add top tier venues in query make bad results..'''
    venues = ['(NeurIPS)','(ICML)', '(ICLR)', '(ACL)']#, '(CVPR)', '(ICCV)', '(ECCV)', '(AAAI)', '(KDD)', '(ACL)', '(EMNLP)']
    venue_filter = " OR ".join([f"jr:{v} OR co:{v}" for v in venues])
    #return f"({venue_filter}) AND ({user_query})"
    return f"({user_query})"


retriever = ArxivRetriever(
    load_max_docs=15,
    doc_content_chars_max=30000
)

def extract_common_fields_from_doc(d):
    '''ectract data fields from retriever's results'''
    m = d.metadata or {}
    title     = m.get("title") or m.get("Title")  # pretty_print_docs와 동일
    url       = m.get("entry_id") or m.get("Entry ID") or m.get("Entry_ID") or m.get("url") or m.get("pdf_url")
    authors   = m.get("Authors")
    published = str(m.get("Published") or m.get("published") or m.get("publish_date"))
    content   = (d.page_content or "").strip()
    return title, url, authors, published, content


def pretty_print_docs(docs, max_chars=1200):
    '''make retrieved papers data look pretty (for prompting and ui)'''
    lines = []
    for i, d in enumerate(docs, 1):
        title, url, authors, published, content = extract_common_fields_from_doc(d)
        if len(content) > max_chars:
            content = content[:max_chars] + " ..."
        block = [
            f"### [{i}] {title}",
            f"- url: {url}",
            f"- published: {published}",
            f"- authors: {authors}",
            f"- content:\n{content}",
        ]
        lines.append("\n".join(block))
    return "\n\n".join(lines)

def extract_arxiv_id_from_url(url: str) -> str:
    '''extract url to request'''
    if not url:
        return ""
    m = re.search(r'arxiv\.org/(abs|pdf)/([0-9]+\.[0-9]+)', url)
    return m.group(2) if m else ""

def fetch_full_text_from_arxiv_id(arxiv_id: str, char_limit: int = None) -> str:
    '''get full text of fetched paper for rag'''
    if not arxiv_id:
        return ""
    search = arxiv.Search(id_list=[arxiv_id])
    client = arxiv.Client()
    results = list(client.results(search))
    if not results:
        return ""
    pdf_path = results[0].download_pdf()
    text = ""
    with fitz.open(pdf_path) as doc:
        text = "".join(page.get_text() for page in doc)
    if char_limit is not None and len(text) > char_limit:
        text = text[:char_limit]
    return text.strip()


# ---- demo cell 3 ----
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import Chroma

### set embedding model and vertor db for rag!

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
text_splitter = TokenTextSplitter(
    chunk_size=3000, chunk_overlap=1000
)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

def upsert_into_chroma(docs, fulltext_char_limit: int = None):
    '''store searched paper's text in vector db.'''
    texts, metadatas, ids = [], [], []
    for d in docs:
        title, url, authors, published, abstract = extract_common_fields_from_doc(d)
        arxiv_id = extract_arxiv_id_from_url(url) or (re.sub(r'\W+', '-', str(title))[:50] if title else "")

        full_text = ""
        try:
            full_text = fetch_full_text_from_arxiv_id(arxiv_id, char_limit=fulltext_char_limit)
        except Exception:
            full_text = ""
        if not full_text:
            full_text = abstract or ""
        if not full_text.strip():
            continue

        chunks = text_splitter.split_text(full_text)
        for j, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "paper_title": title,
                "paper_url": url,
                "authors": authors,
                "published": published,
                "arxiv_id": arxiv_id,
                "chunk_index": j,
            })
            ids.append(f"{arxiv_id}::{j}")

    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        vectorstore.persist()

def run_arxiv_search_with_top_tier(user_query: str):
    query = build_top_tier_query(user_query)
    docs = retriever.invoke(query)
    rendered = pretty_print_docs(docs)
    upsert_into_chroma(docs, fulltext_char_limit=None)
    return docs, rendered


# ---- demo cell 4 ----
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model=LLM_MODEL, temperature=1)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert AI research summarizer. Using the provided arXiv documents, "
     "first write a concise overview of the research landscape related to the user's query. "
     "Then provide a bullet list of references with: [Title](URL) — Published — Authors. "
     "The base language is Korean for explanations (not titles/authors/model names)."),
    ("human", "User query:\n{user_query}\n\nDocuments:\n{docs_rendered}")
])

summary_chain = summary_prompt | llm | StrOutputParser()


# ---- demo cell 5 ----
from langchain_core.runnables import RunnableParallel, RunnableLambda

local_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's question using ONLY the provided context chunks from previously saved papers. "
     "Cite the paper titles inline when relevant. If not in context, say you don't have it."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    lines = []
    for d in docs:
        m = d.metadata or {}
        title = m.get("paper_title", "(unknown)")
        lines.append(f"[{title}] (chunk {m.get('chunk_index', '?')}):\n{d.page_content}\n")
    return "\n\n".join(lines)

rag_chain = (
    {"context": local_retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x: x)}
    | rag_prompt
    | llm
    | StrOutputParser()
)


# ---- demo cell 6 ----
# ================================
# 첫 질의 파이프라인
# ================================
def run_pipeline(user_query: str):
    docs, rendered = run_arxiv_search_with_top_tier(user_query)
    print("=== [Retrieved Docs for Agent] ===\n")  # UPDATED
    print(rendered)

    print("\n\n=== [Summarized Overview + References] ===\n")
    overview = summary_chain.invoke({"user_query": user_query, "docs_rendered": rendered})
    print(overview)
    global RECOMMENDED_TITLES
    RECOMMENDED_TITLES = []
    for d in docs:
        t, _, _, _, _ = extract_common_fields_from_doc(d)
        if t:
            RECOMMENDED_TITLES.append(t)
    return #overview

# ================================
# 후속 질의용 Agent 라우팅
# ================================
route_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router. Decide whether the user's follow-up question is primarily about any of the following papers. "
     "If yes, answer EXACTLY 'RAG' (and list related titles after a pipe), else answer EXACTLY 'NO_RAG'. "
     "Do not add extra words.\n\nPapers:\n{titles}"),
    ("human", "Question: {question}")
])

route_chain = route_prompt | llm | StrOutputParser()

nonrag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI research assistant. Answer the question concisely in Korean. "
     "Do not assume access to local papers."),
    ("human", "{question}")
])
nonrag_chain = nonrag_prompt | llm | StrOutputParser()

def ask_followup_agent(question: str):
    titles_str = "\n".join(f"- {t}" for t in globals().get("RECOMMENDED_TITLES", []))
    decision = route_chain.invoke({"titles": titles_str, "question": question}).strip()
    use_rag = decision.startswith("RAG")
    print(f"[Router] decision = {decision}")

    if use_rag:
        print("=== [RAG Answer From Local Chroma] ===\n")
        ans = rag_chain.invoke(question)
    else:
        print("=== [Non-RAG LLM Answer] ===\n")
        ans = nonrag_chain.invoke({"question": question})
    print(ans)
    return #ans


# ---- demo cell 7 ----
run_pipeline("CT denoising")


# ---- demo cell 8 ----
ask_followup_agent("세 번째 논문에서 Bilateral Filter가 어떻게 학습 가능하다는거야? 이는 non-trainable한 필터아냐?")


# ---- demo cell 9 ----
# === END DEMO CODE ===


# --- Helpers to call demo functions and capture printed output ---
def call_and_capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = fn(*args, **kwargs)
    txt = buf.getvalue().strip()
    return txt if txt else (str(ret) if ret is not None else "")

# --- Guard: ensure demo functions exist ---
missing = []
if "run_pipeline" not in globals():
    missing.append("run_pipeline")
if "ask_followup_agent" not in globals():
    missing.append("ask_followup_agent")
if missing:
    st.error("필수 함수가 없습니다: " + ", ".join(missing) + " — demo.ipynb 내용을 확인하세요.")

# --- Render history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat input ---
user_input = st.chat_input("연구 주제 또는 후속 질문을 입력하세요...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not os.environ.get("OPENAI_API_KEY"):
        with st.chat_message("assistant"):
            st.error("OpenAI API Key가 설정되어 있지 않습니다. 왼쪽 사이드바에서 키를 입력해주세요.")
    else:
        try:
            if not st.session_state.first_query_done:
                txt = call_and_capture(globals()["run_pipeline"], user_input)
                st.session_state.first_query_done = True
            else:
                txt = call_and_capture(globals()["ask_followup_agent"], user_input)

            if not txt.strip():
                txt = "처리가 완료되었지만 출력할 텍스트가 없습니다."
            with st.chat_message("assistant"):
                st.markdown(txt)
            st.session_state.messages.append({"role": "assistant", "content": txt})
        except Exception as e:
            with st.chat_message("assistant"):
                st.exception(e)

# --- Footer actions ---
cols = st.columns(2)
with cols[0]:
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()
with cols[1]:
    st.caption("© Your Research Assistant Demo")
