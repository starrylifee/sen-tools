import streamlit as st
import json
from langchain_openai import ChatOpenAI
from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import pathlib
import toml

# Streamlit의 기본 메뉴와 푸터 숨기기
hide_github_icon = """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK{ display: none; }
    #MainMenu{ visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
"""

# secrets.toml 파일 경로
secrets_path = pathlib.Path(__file__).parent.parent / ".streamlit/secrets.toml"

# secrets.toml 파일 읽기
with open(secrets_path, "r") as f:
    secrets = toml.load(f)

# OpenAI API 키 로드
openai_api_key = secrets.get("OPENAI_API_KEY")

# JSON 파일 경로
json_file_path = pathlib.Path(__file__).parent / "training_data.json"

# JSON 데이터 로드
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

json_data = load_json_data(json_file_path)

# Streamlit 애플리케이션 제목
st.title("디지털배지 설명 및 발급조건 생성기")

# 세션 상태 초기화
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# 사이드바에 파일 업로드
with st.sidebar:
    st.header("📄 파일 업로드")
    uploaded_file = st.file_uploader("연수계획서를 업로드하세요", type=["pdf", "hwp", "hwpx"])
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

# 본문에 드롭다운 메뉴와 버튼 추가
st.subheader("📝 옵션을 선택하세요:")

col1, col2 = st.columns(2)

with col1:
    content_option = st.selectbox("연수 내용 기준 선택", [content['category'] for content in json_data['content_standards']])
    
with col2:
    level_option = st.selectbox("연수 단계 선택", [stage['stage'] for stage in json_data['training_stages']])

generate_btn = st.button("✨ 생성하기")

# HWP 및 PDF 파일 처리 함수
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def process_uploaded_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 파일 확장자에 따라 로더 선택
    if file.name.endswith(".hwp") or file.name.endswith(".hwpx"):
        loader = HWPLoader(file_path)
    elif file.name.endswith(".pdf"):
        loader = PDFPlumberLoader(file_path)
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None

    # 문서 로드
    docs = loader.load()

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 임베딩 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    # 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 검색기 생성
    retriever = vectorstore.as_retriever()

    # 문서 내용을 텍스트로 반환
    document_text = " ".join([doc.page_content for doc in docs])
    
    return retriever, document_text

# 체인 생성 함수
def create_chain(content_option, level_option, uploaded_file_content):
    # JSON 데이터에서 선택된 옵션에 해당하는 설명을 찾기
    stage_description = next((stage['description'] for stage in json_data['training_stages'] if stage['stage'] == level_option), "")
    content_items = next((content['items'] for content in json_data['content_standards'] if content['category'] == content_option), [])
    
    # 프롬프트 생성
    prompt = f"""
    연수 계획서에 기초한 디지털 배지 설명과 발급 조건을 제시하세요.
    
    디지털 배지 설명: {uploaded_file_content}속 내용을 주제: {content_option}, 단계: {level_option}, 단계 설명: {stage_description}, 내용 요소: {"; ".join(content_items)}로 해석해서 발급할 디지털 배지 설명 내용을 200자로 출력
    
    배지 발급 조건: 연수과정명, 연수기간, 연수장소, 연수주관기관, 이수 시간 등 {uploaded_file_content}에서 추출한 정보를 200자로 출력
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    # 체인을 실행하고 결과를 반환하는 함수
    def run_chain(prompt):
        response = llm(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    return run_chain(prompt)

# 사용자 입력 처리
if generate_btn:
    if st.session_state["uploaded_file"] is not None:
        with st.spinner("AI 응답을 생성하는 중입니다..."):
            # 업로드된 파일 처리
            retriever, uploaded_file_content = process_uploaded_file(st.session_state["uploaded_file"])
            if retriever:
                # 체인 함수 생성 및 실행
                ai_answer = create_chain(content_option, level_option, uploaded_file_content)
                
                # 응답을 Markdown 형식으로 포맷
                if isinstance(ai_answer, str):
                    markdown_content = ai_answer.replace("\n", "  \n")
                    # 응답을 Markdown 형식으로 출력
                    st.markdown(markdown_content, unsafe_allow_html=True)
                else:
                    st.error("AI 응답을 문자열로 변환할 수 없습니다.")
    else:
        st.error("연수계획서를 업로드 해주세요.")
