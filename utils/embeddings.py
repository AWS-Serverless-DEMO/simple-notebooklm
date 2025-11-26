"""Embeddings generation module using AWS Bedrock Titan Text Embeddings V2"""

import json
import time
import boto3
from botocore.config import Config as BotoConfig
from typing import List
from config import Config


class EmbeddingsGenerator:
    """Generate embeddings using Bedrock Titan Text Embeddings V2"""

    def __init__(self):
        """Initialize Bedrock client with retry configuration"""
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
        self.model_id = Config.BEDROCK_EMBEDDING_MODEL_ID

        # Rate limiting: 2000 requests/minute = ~33 requests/second
        # Use 30 requests/second to be safe
        self._last_request_time = 0
        self._min_request_interval = 1.0 / 30.0  # 0.033 seconds between requests

    def generate_embedding(self, text: str, dimensions: int = 1024, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text with rate limiting

        Args:
            text: Input text to embed
            dimensions: Output vector dimensions (1024, 512, or 256)
            normalize: Whether to normalize the output vector

        Returns:
            List of floats representing the embedding vector
        """
        # Rate limiting: ensure minimum interval between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body
            )

            response_body = json.loads(response['body'].read())
            self._last_request_time = time.time()  # Update after successful request
            return response_body['embedding']

        except Exception as e:
            self._last_request_time = time.time()  # Update even on error
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str], dimensions: int = 1024, normalize: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts
            dimensions: Output vector dimensions
            normalize: Whether to normalize output vectors

        Returns:
            List of embedding vectors
        """
        embeddings = []
        failed_count = 0

        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text, dimensions, normalize)
                embeddings.append(embedding)
            except Exception as e:
                failed_count += 1
                error_msg = str(e)[:100]  # Truncate long error messages

                # Try to display warning in Streamlit if available
                try:
                    import streamlit as st
                    st.warning(f"⚠️ 청크 {i+1}/{len(texts)} 임베딩 실패: {error_msg}")
                except (ImportError, RuntimeError):
                    # Fallback to print if Streamlit not available or not in app context
                    print(f"Warning: Failed to generate embedding for chunk {i+1}/{len(texts)} (length {len(text)}): {error_msg}")

                # Append None for failed embeddings to maintain index alignment
                embeddings.append(None)

        # Summary warning if there were failures
        if failed_count > 0:
            try:
                import streamlit as st
                st.warning(f"⚠️ 총 {failed_count}/{len(texts)}개 청크의 임베딩 생성에 실패했습니다.")
            except (ImportError, RuntimeError):
                print(f"Warning: Total {failed_count}/{len(texts)} embeddings failed")

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings being generated"""
        return 1024  # Default for Titan Text Embeddings V2
