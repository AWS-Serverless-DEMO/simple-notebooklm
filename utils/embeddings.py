"""Embeddings generation module using AWS Bedrock Titan Text Embeddings V2"""

import json
import boto3
from typing import List
from config import Config


class EmbeddingsGenerator:
    """Generate embeddings using Bedrock Titan Text Embeddings V2"""

    def __init__(self):
        """Initialize Bedrock client"""
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        self.model_id = Config.BEDROCK_EMBEDDING_MODEL_ID

    def generate_embedding(self, text: str, dimensions: int = 1024, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed
            dimensions: Output vector dimensions (1024, 512, or 256)
            normalize: Whether to normalize the output vector

        Returns:
            List of floats representing the embedding vector
        """
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
            return response_body['embedding']

        except Exception as e:
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

        for text in texts:
            try:
                embedding = self.generate_embedding(text, dimensions, normalize)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Failed to generate embedding for text (length {len(text)}): {str(e)}")
                # Append None for failed embeddings to maintain index alignment
                embeddings.append(None)

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings being generated"""
        return 1024  # Default for Titan Text Embeddings V2
