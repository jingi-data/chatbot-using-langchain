# RAG based Q&A using LangChain
## Description
작은 사이즈(1.3B)의 한글 기반 LLM(대규모 언어 모델)을 사용하여, '국민취업지원제도 참여자 안내서'의 내용을 바탕으로 질의응답을 진행합니다.
## How To Run
### 1. 디렉토리 이동
```bash
git clone https://github.com/jingi-data/chatbot-using-langchain.git
cd chatbot-using-langchain
```

### 2. 환경 설정
```bash
conda create -f environment.yml # 'ragqa'이라는 이름의 가상환경이 설치됩니다. 
conda activate ragqa
```

### 3. 실행
<i>**NOTICE:** streamlit을 사용하기 위해 '8501' 포트를 추가해주세요.</i>
```bash
streamlit run prototype.py
```