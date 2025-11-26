"""RAG (Retrieval-Augmented Generation) engine using Bedrock Claude"""

import json
import boto3
from botocore.config import Config as BotoConfig
from typing import List, Dict, Optional
from config import Config
from .embeddings import EmbeddingsGenerator
from .s3_vectors import S3VectorStore


class RAGEngine:
    """RAG engine for question answering with document context"""

    def __init__(self):
        """Initialize RAG engine with Bedrock Claude and retry configuration"""
        retry_config = BotoConfig(
            retries={
                'max_attempts': 5,
                'mode': 'adaptive'  # Exponential backoff with jitter
            }
        )

        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            config=retry_config
        )
        self.model_id = Config.BEDROCK_LLM_MODEL_ID
        self.embeddings_generator = EmbeddingsGenerator()
        self.vector_store = S3VectorStore()
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD

    def retrieve_context(self, question: str, top_k: int = None) -> Dict:
        """
        Retrieve relevant context for a question

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        # Generate embedding for the question
        question_embedding = self.embeddings_generator.generate_embedding(question)

        print(f"\n[DEBUG] Question embedding generated: {len(question_embedding)} dimensions")

        # Query similar chunks from vector store
        results = self.vector_store.query_vectors(
            query_embedding=question_embedding,
            top_k=top_k or Config.TOP_K_RESULTS
        )

        print(f"[DEBUG] Query returned {len(results)} results")
        if results:
            print(f"[DEBUG] Top 3 results:")
            for i, result in enumerate(results[:3], 1):
                similarity = result.get('similarity', 0)
                content_preview = result.get('content', '')[:100].replace('\n', ' ')
                doc_name = result.get('metadata', {}).get('document', 'Unknown')
                page = result.get('metadata', {}).get('page', '?')
                print(f"  [{i}] 유사도: {similarity:.4f} | 문서: {doc_name} (p.{page})")
                print(f"      내용: {content_preview}...")
            print(f"[DEBUG] Similarity threshold: {self.similarity_threshold}")

        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity'] >= self.similarity_threshold
        ]

        print(f"[DEBUG] After filtering: {len(filtered_results)} relevant results (threshold >= {self.similarity_threshold})")

        return {
            'chunks': filtered_results,
            'total_retrieved': len(results),
            'total_relevant': len(filtered_results),
            'has_relevant_context': len(filtered_results) > 0
        }

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict:
        """
        Generate answer using Claude with retrieved context

        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks

        Returns:
            Dictionary with answer and sources
        """
        if not context_chunks:
            return {
                'answer': "죄송합니다. 업로드된 문서에서 질문과 관련된 내용을 찾을 수 없습니다. 다른 질문을 시도해보시거나 관련 문서를 업로드해주세요.",
                'sources': [],
                'has_answer': False
            }

        # Build context from chunks
        context_parts = []
        sources = []

        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[문서 {i}: {chunk['metadata']['document']}, "
                f"페이지 {chunk['metadata']['page']}, "
                f"유사도: {chunk['similarity']:.2f}]\n{chunk['content']}"
            )

            sources.append({
                'document': chunk['metadata']['document'],
                'page': chunk['metadata']['page'],
                'similarity': chunk['similarity'],
                'preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content']
            })

        context = "\n\n".join(context_parts)

        # Build prompt for Claude (NotebookLM style)
        prompt = f"""당신은 문서 기반 질의응답 AI입니다. 아래 제공된 문서 청크들을 분석하여 질문에 답변하세요.

검색된 문서 청크들:
{context}

질문: {question}

중요한 답변 규칙:
1. **관련성 판단**: 제공된 청크들 중에서 질문과 실제로 관련 있는 내용만 사용하세요.
   - 유사도 점수가 낮더라도, 의미상 질문과 관련이 있다면 사용할 수 있습니다.
   - 예: "만점 받으려면?" → "채점 기준", "평가 항목" 등의 청크가 관련 있음

2. **관련 없는 청크 무시**: 질문과 무관한 청크는 무시하고 답변에 포함하지 마세요.

3. **종합적 답변**: 여러 청크의 정보를 종합하여 구조화된 답변을 제공하세요.
   - 관련 항목이 여러 개라면 번호를 매기거나 카테고리로 나누세요.
   - 각 항목에 대해 상세히 설명하세요.

4. **출처 명시**: 답변에 사용한 정보의 출처(문서명, 페이지)를 반드시 표시하세요.
   - 예: "과제 설명서(p.5)에 따르면..."

5. **정보 부족 시**: 제공된 문서에 관련 정보가 전혀 없다면, 솔직하게 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.

답변을 시작하세요:"""

        # Call Claude
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "top_p": 0.9
        })

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body
            )

            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text']

            return {
                'answer': answer,
                'sources': sources,
                'has_answer': True,
                'model_used': self.model_id
            }

        except Exception as e:
            raise RuntimeError(f"Failed to generate answer with Claude: {str(e)}")

    def ask(self, question: str, top_k: int = None) -> Dict:
        """
        Complete RAG pipeline: retrieve context and generate answer

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant context
        context_result = self.retrieve_context(question, top_k)

        # Generate answer
        answer_result = self.generate_answer(question, context_result['chunks'])

        # Combine results
        return {
            **answer_result,
            'retrieval_stats': {
                'total_retrieved': context_result['total_retrieved'],
                'total_relevant': context_result['total_relevant'],
                'similarity_threshold': self.similarity_threshold
            }
        }
