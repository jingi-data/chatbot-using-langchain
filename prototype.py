import torch
import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

# 모델 및 관련 리소스 캐싱
@st.cache_resource()
def load_model_and_resources():
    model_id = 'maywell/Synatra-42dot-1.3B'
    model = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device_map='auto',
        pipeline_kwargs={
            "do_sample": True,
            "temperature": 0.2,
            "max_new_tokens": 512,},
    )

    # PDF 문서 로드
    loader = PyPDFLoader("./kua.pdf")
    document = loader.load()

    # 문서 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(document)

    # 임베딩 객체 생성
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': device}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={'normalize_embeddings': False}
    )

    # 문서 검색 엔진 생성
    docsearch = Chroma.from_documents(texts, embeddings)
    retriever = docsearch.as_retriever()

    # RAG 프롬프트 설정
    rag_prompt = hub.pull("rlm/rag-prompt")

    # RAG 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt 
        | model
    )

    return rag_chain

# Streamlit 애플리케이션 정의
def main():
    st.title("국민취업지원제도 Q&A")

    rag_chain = load_model_and_resources()

    question = st.text_input("질문을 입력하세요:")

    if question:
        answer = rag_chain.invoke(question)
        st.text("답변:")
        st.write(answer)

if __name__ == "__main__":
    main()
