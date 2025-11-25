"""S3 Vectors storage and retrieval module"""

import boto3
from typing import List, Dict, Optional
from config import Config


class S3VectorStore:
    """Manage vector storage and retrieval using AWS S3 Vectors"""

    def __init__(self):
        """Initialize S3 Vectors client"""
        self.s3vectors = boto3.client(
            's3vectors',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
        )
        self.bucket_name = Config.S3_VECTOR_BUCKET_NAME
        self.index_name = Config.S3_VECTOR_INDEX_NAME

    def put_vectors(self, chunks: List[Dict], embeddings: List[List[float]]) -> Dict:
        """
        Store vectors with metadata in S3 Vectors

        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'
            embeddings: List of embedding vectors corresponding to chunks

        Returns:
            Response from PutVectors API

        Raises:
            ValueError: If chunks and embeddings lengths don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        # Filter out chunks with None embeddings
        valid_items = [
            (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
            if embedding is not None
        ]

        if not valid_items:
            raise ValueError("No valid embeddings to store")

        # Prepare vectors for S3 Vectors (max 500 per request)
        batch_size = 500
        responses = []

        for i in range(0, len(valid_items), batch_size):
            batch = valid_items[i:i + batch_size]

            vectors = []
            for chunk, embedding in batch:
                vector_item = {
                    'key': chunk['metadata']['chunk_id'],
                    'data': {'float32': embedding},
                    'metadata': {
                        'content': chunk['content'],
                        'document': chunk['metadata']['document'],
                        'page': str(chunk['metadata']['page']),
                        'chunk_index': str(chunk['metadata']['chunk_index']),
                        'source_type': chunk['metadata']['source_type']
                    }
                }
                vectors.append(vector_item)

            try:
                response = self.s3vectors.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    vectors=vectors
                )
                responses.append(response)
            except Exception as e:
                raise RuntimeError(f"Failed to put vectors: {str(e)}")

        return {
            'total_stored': len(valid_items),
            'batches': len(responses),
            'responses': responses
        }

    def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = None,
        distance_metric: str = "COSINE",
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query similar vectors from S3 Vectors

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default from config)
            distance_metric: Distance metric ("COSINE" or "EUCLIDEAN")
            metadata_filter: Optional metadata filters

        Returns:
            List of similar chunks with metadata and similarity scores
        """
        top_k = top_k or Config.TOP_K_RESULTS

        query_params = {
            'vectorBucketName': self.bucket_name,
            'indexName': self.index_name,
            'queryVector': {'float32': query_embedding},
            'topK': top_k,
            'distanceMetric': distance_metric,
            'includeMetadata': True,
            'includeVectorData': False
        }

        if metadata_filter:
            query_params['metadataFilter'] = metadata_filter

        try:
            response = self.s3vectors.query_vectors(**query_params)

            results = []
            for vector_result in response.get('vectors', []):
                result = {
                    'content': vector_result['metadata'].get('content', ''),
                    'metadata': {
                        'document': vector_result['metadata'].get('document', ''),
                        'page': int(vector_result['metadata'].get('page', 1)),
                        'chunk_index': int(vector_result['metadata'].get('chunk_index', 0)),
                        'source_type': vector_result['metadata'].get('source_type', ''),
                        'chunk_id': vector_result['key']
                    },
                    'distance': vector_result.get('distance', 0),
                    'similarity': 1 - vector_result.get('distance', 0)  # Convert distance to similarity
                }
                results.append(result)

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to query vectors: {str(e)}")

    def delete_vectors_by_document(self, document_name: str) -> Dict:
        """
        Delete all vectors for a specific document

        Args:
            document_name: Name of the document

        Returns:
            Deletion result
        """
        # Note: S3 Vectors doesn't have a direct delete by metadata filter
        # This is a placeholder for future implementation
        raise NotImplementedError("Delete by document not yet implemented in S3 Vectors API")
