"""RAG (Retrieval-Augmented Generation) engine using Bedrock Claude"""

import json
import boto3
from typing import List, Dict, Optional
from config import Config
from .embeddings import EmbeddingsGenerator
from .s3_vectors import S3VectorStore


class RAGEngine:
    """RAG engine for question answering with document context"""

    def __init__(self):
        """Initialize RAG engine with Bedrock Claude"""
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
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

        # Query similar chunks from vector store
        results = self.vector_store.query_vectors(
            query_embedding=question_embedding,
            top_k=top_k or Config.TOP_K_RESULTS,
            distance_metric="COSINE"
        )

        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity'] >= self.similarity_threshold
        ]

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

        # Build prompt for Claude
        prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

문서 내용:
{context}

질문: {question}

답변 지침:
1. 제공된 문서 내용만을 사용하여 답변하세요.
2. 답변 시 반드시 출처(문서명과 페이지 번호)를 명시하세요.
3. 문서에 명확한 답변이 없다면 솔직히 말해주세요.
4. 간결하고 명확하게 답변하세요."""

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
