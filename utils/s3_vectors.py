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

    def list_all_vectors(self) -> List[Dict]:
        """
        List all vectors in the index with their metadata

        Returns:
            List of all vectors with keys and metadata
        """
        try:
            # Query with a very large top_k to get all vectors
            # Note: This is a workaround since S3 Vectors doesn't have a direct list API
            response = self.s3vectors.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                queryVector={'float32': [0.0] * 1024},  # Dummy vector
                topK=10000,  # Maximum allowed
                includeMetadata=True,
                includeVectorData=False
            )

            vectors = []
            for vector_result in response.get('vectors', []):
                vectors.append({
                    'key': vector_result['key'],
                    'metadata': vector_result.get('metadata', {}),
                    'distance': vector_result.get('distance', 0)
                })

            return vectors

        except Exception as e:
            raise RuntimeError(f"Failed to list vectors: {str(e)}")

    def delete_vectors_by_keys(self, keys: List[str]) -> Dict:
        """
        Delete vectors by their keys (batch deletion)

        Args:
            keys: List of vector keys to delete

        Returns:
            Deletion result with count
        """
        if not keys:
            return {'deleted_count': 0, 'message': 'No keys provided'}

        try:
            # Delete in batches (max 1000 per request as per AWS limits)
            batch_size = 1000
            total_deleted = 0

            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]

                self.s3vectors.delete_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    keys=batch_keys
                )

                total_deleted += len(batch_keys)

            return {
                'deleted_count': total_deleted,
                'message': f'Successfully deleted {total_deleted} vectors'
            }

        except Exception as e:
            raise RuntimeError(f"Failed to delete vectors: {str(e)}")

    def delete_vectors_by_document(self, document_name: str) -> Dict:
        """
        Delete all vectors for a specific document

        Args:
            document_name: Name of the document

        Returns:
            Deletion result with count
        """
        try:
            # First, find all vectors for this document
            all_vectors = self.list_all_vectors()

            # Filter by document name
            keys_to_delete = [
                vec['key']
                for vec in all_vectors
                if vec['metadata'].get('document') == document_name
            ]

            if not keys_to_delete:
                return {
                    'deleted_count': 0,
                    'message': f'No vectors found for document: {document_name}'
                }

            # Delete the filtered vectors
            result = self.delete_vectors_by_keys(keys_to_delete)
            result['document'] = document_name

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to delete vectors for document '{document_name}': {str(e)}")

    def delete_all_vectors(self) -> Dict:
        """
        Delete all vectors in the index

        Returns:
            Deletion result with count
        """
        try:
            # Get all vectors
            all_vectors = self.list_all_vectors()

            if not all_vectors:
                return {
                    'deleted_count': 0,
                    'message': 'No vectors found in index'
                }

            # Extract all keys
            all_keys = [vec['key'] for vec in all_vectors]

            # Delete all vectors
            result = self.delete_vectors_by_keys(all_keys)

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to delete all vectors: {str(e)}")

    def list_documents(self) -> List[Dict]:
        """
        List all unique documents in the vector store

        Returns:
            List of documents with metadata (name, total chunks, source type)
        """
        try:
            all_vectors = self.list_all_vectors()

            # Group by document name
            documents_map = {}

            for vec in all_vectors:
                doc_name = vec['metadata'].get('document', 'unknown')

                if doc_name not in documents_map:
                    documents_map[doc_name] = {
                        'document': doc_name,
                        'source_type': vec['metadata'].get('source_type', 'unknown'),
                        'chunk_count': 0,
                        'pages': set()
                    }

                documents_map[doc_name]['chunk_count'] += 1

                # Collect unique pages
                page = vec['metadata'].get('page')
                if page:
                    documents_map[doc_name]['pages'].add(int(page))

            # Convert to list and format
            documents = []
            for doc_info in documents_map.values():
                doc_info['page_count'] = len(doc_info['pages']) if doc_info['pages'] else 0
                doc_info['pages'] = sorted(list(doc_info['pages'])) if doc_info['pages'] else []
                documents.append(doc_info)

            # Sort by document name
            documents.sort(key=lambda x: x['document'])

            return documents

        except Exception as e:
            raise RuntimeError(f"Failed to list documents: {str(e)}")
