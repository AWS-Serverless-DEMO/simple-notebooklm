"""
Simple NotebookLM - Document Q&A with S3 Vectors and Bedrock Claude

A Streamlit application for uploading documents and asking questions using RAG.
"""

import streamlit as st
from config import Config
from utils import (
    DocumentProcessor,
    TextSplitter,
    EmbeddingsGenerator,
    S3VectorStore,
    RAGEngine
)

# Page configuration
st.set_page_config(
    page_title="Simple NotebookLM",
    page_icon="📚",
    layout="wide"
)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def validate_config():
    """Validate configuration and show errors if needed"""
    try:
        Config.validate()
        return True
    except ValueError as e:
        st.error(f"⚠️ Configuration Error: {str(e)}")
        st.info("Please set up your `.env` file with required AWS credentials and S3 Vector configuration.")
        return False


def process_document(uploaded_file):
    """Process uploaded document through the RAG pipeline"""
    try:
        # Extract text from document
        with st.spinner("📄 문서에서 텍스트를 추출하는 중..."):
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name

            processor = DocumentProcessor()
            documents = processor.process_document(file_bytes, filename)

            st.success(f"✅ {len(documents)}개 페이지/섹션에서 텍스트 추출 완료")

        # Split into chunks
        with st.spinner("✂️ 텍스트를 청크로 분할하는 중..."):
            splitter = TextSplitter()
            chunks = splitter.split_documents(documents)

            st.success(f"✅ {len(chunks)}개 청크로 분할 완료")

        # Generate embeddings
        with st.spinner("🧮 임베딩 벡터 생성 중... (시간이 소요될 수 있습니다)"):
            embeddings_gen = EmbeddingsGenerator()
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = embeddings_gen.generate_embeddings_batch(chunk_texts)

            valid_count = sum(1 for e in embeddings if e is not None)
            st.success(f"✅ {valid_count}개 임베딩 벡터 생성 완료")

        # Store in S3 Vectors
        with st.spinner("💾 S3 Vectors에 저장 중..."):
            vector_store = S3VectorStore()
            result = vector_store.put_vectors(chunks, embeddings)

            st.success(f"✅ {result['total_stored']}개 벡터 저장 완료 ({result['batches']}개 배치)")

        st.session_state.document_processed = True
        st.session_state.document_name = filename

        return True

    except Exception as e:
        st.error(f"❌ 문서 처리 중 오류 발생: {str(e)}")
        return False


def display_answer(result):
    """Display answer with sources in a formatted way"""
    # Display answer
    st.markdown("### 💬 답변")
    st.markdown(result['answer'])

    # Display sources if available
    if result.get('sources'):
        st.markdown("---")
        st.markdown("### 📎 출처")

        for i, source in enumerate(result['sources'], 1):
            with st.expander(f"출처 {i}: {source['document']} (페이지 {source['page']}, 유사도: {source['similarity']:.2%})"):
                st.text(source['preview'])

    # Display retrieval stats
    if result.get('retrieval_stats'):
        stats = result['retrieval_stats']
        st.markdown("---")
        st.caption(
            f"📊 검색 통계: {stats['total_relevant']}/{stats['total_retrieved']}개 관련 청크 발견 "
            f"(임계값: {stats['similarity_threshold']})"
        )


def main():
    """Main application"""
    st.title("📚 Simple NotebookLM")
    st.markdown("**AWS S3 Vectors와 Bedrock Claude를 활용한 문서 기반 질의응답 시스템**")

    initialize_session_state()

    # Validate configuration
    if not validate_config():
        st.stop()

    # Sidebar - Document Upload
    with st.sidebar:
        st.header("📤 문서 업로드")

        uploaded_file = st.file_uploader(
            "PDF, DOCX, TXT 파일을 업로드하세요",
            type=['pdf', 'docx', 'txt'],
            help="업로드된 문서는 자동으로 분석되어 질문에 답변할 수 있도록 처리됩니다."
        )

        if uploaded_file:
            if st.button("📄 문서 처리 시작", type="primary", use_container_width=True):
                process_document(uploaded_file)

        # Show current document status
        if st.session_state.document_processed:
            st.success(f"✅ 현재 문서: {st.session_state.document_name}")
        else:
            st.info("⏳ 문서를 업로드하고 처리해주세요")

        # Configuration display
        st.markdown("---")
        st.markdown("### ⚙️ 설정")
        st.caption(f"리전: {Config.AWS_REGION}")
        st.caption(f"청크 크기: {Config.CHUNK_SIZE}")
        st.caption(f"중복: {Config.CHUNK_OVERLAP}")
        st.caption(f"Top-K: {Config.TOP_K_RESULTS}")
        st.caption(f"유사도 임계값: {Config.SIMILARITY_THRESHOLD}")

    # Main area - Q&A Interface
    if st.session_state.document_processed:
        st.markdown("### 💭 질문하기")
        st.markdown(f"**{st.session_state.document_name}**에 대해 무엇이든 물어보세요!")

        # Question input
        question = st.text_input(
            "질문을 입력하세요:",
            placeholder="예: 이 문서의 주요 내용은 무엇인가요?",
            key="question_input"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 질문하기", type="primary")

        if ask_button and question:
            with st.spinner("🤔 답변을 생성하는 중..."):
                try:
                    rag_engine = RAGEngine()
                    result = rag_engine.ask(question)

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'result': result
                    })

                    # Display answer
                    display_answer(result)

                except Exception as e:
                    st.error(f"❌ 질문 처리 중 오류 발생: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📜 대화 기록")

            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"질문 {len(st.session_state.chat_history) - i + 1}: {chat['question'][:50]}..."):
                    st.markdown(f"**질문:** {chat['question']}")
                    st.markdown(f"**답변:** {chat['result']['answer']}")

                    if chat['result'].get('sources'):
                        st.caption(f"출처: {len(chat['result']['sources'])}개")

    else:
        st.info("👈 왼쪽 사이드바에서 문서를 업로드하고 처리해주세요.")

        # Show instructions
        st.markdown("### 📖 사용 방법")
        st.markdown("""
        1. **문서 업로드**: 왼쪽 사이드바에서 PDF, DOCX, TXT 파일을 업로드하세요
        2. **문서 처리**: "문서 처리 시작" 버튼을 클릭하여 문서를 분석하세요
        3. **질문하기**: 처리가 완료되면 문서에 대해 질문할 수 있습니다
        4. **답변 확인**: AI가 문서 내용을 기반으로 답변하고 출처를 제공합니다

        **기술 스택:**
        - 🔤 **텍스트 추출**: PyPDF2, python-docx
        - ✂️ **청크 분할**: LangChain RecursiveCharacterTextSplitter
        - 🧮 **임베딩**: AWS Bedrock Titan Text Embeddings V2 (1024차원)
        - 💾 **벡터 저장**: AWS S3 Vectors (Native Vector Storage)
        - 🤖 **답변 생성**: AWS Bedrock Claude 3 Sonnet
        """)


if __name__ == "__main__":
    main()
