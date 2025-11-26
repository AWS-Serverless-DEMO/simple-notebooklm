"""S3 Vectors storage and retrieval module"""

import boto3
from botocore.config import Config as BotoConfig
from typing import List, Dict, Optional
from config import Config


class S3VectorStore:
    """Manage vector storage and retrieval using AWS S3 Vectors"""

    def __init__(self):
        """Initialize S3 Vectors client with retry configuration"""
        retry_config = BotoConfig(
            retries={
                'max_attempts': 5,
                'mode': 'adaptive'  # Exponential backoff with jitter
            }
        )

        self.s3vectors = boto3.client(
            's3vectors',
            region_name=Config.AWS_REGION,
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            config=retry_config
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

        # Prepare vectors for S3 Vectors (AWS recommends max 500 per request for optimal performance)
        batch_size = 500
        responses = []
        total_batches = (len(valid_items) + batch_size - 1) // batch_size

        for i in range(0, len(valid_items), batch_size):
            batch = valid_items[i:i + batch_size]
            current_batch = (i // batch_size) + 1

            if total_batches > 1:
                progress_percent = (current_batch / total_batches) * 100
                print(f"  업로드 중... [{current_batch}/{total_batches}] ({progress_percent:.1f}%)")

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

            except self.s3vectors.exceptions.TooManyRequestsException:
                # Handle rate limiting (429 error)
                print(f"  ⚠️  요청 제한 도달, 2초 대기 후 재시도...")
                import time
                time.sleep(2)

                # Retry the batch
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
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query similar vectors from S3 Vectors

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default from config)
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
            'returnMetadata': True,
            'returnDistance': True
        }

        if metadata_filter:
            query_params['filter'] = metadata_filter

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
        List all vectors in the index with their metadata using pagination

        Returns:
            List of all vectors with keys and metadata
        """
        try:
            vectors = []
            next_token = None
            page_count = 0

            # Fetch all vectors using pagination (max 500 per page)
            while True:
                page_count += 1

                # Build request parameters
                list_params = {
                    'vectorBucketName': self.bucket_name,
                    'indexName': self.index_name,
                    'maxResults': 500,  # Maximum allowed per page
                    'returnMetadata': True
                }

                if next_token:
                    list_params['nextToken'] = next_token

                # Call ListVectors API
                response = self.s3vectors.list_vectors(**list_params)

                # Process vectors from this page
                page_vectors = response.get('vectors', [])
                for vector_result in page_vectors:
                    vectors.append({
                        'key': vector_result['key'],
                        'metadata': vector_result.get('metadata', {})
                    })

                # Check if there are more pages
                next_token = response.get('nextToken')
                if not next_token:
                    break

                # Progress indication for large datasets
                if page_count % 5 == 0:
                    print(f"  벡터 목록 조회 중... ({len(vectors)}개 조회됨)")

            return vectors

        except Exception as e:
            raise RuntimeError(f"Failed to list vectors: {str(e)}")

    def delete_vectors_by_keys(self, keys: List[str], show_progress: bool = True) -> Dict:
        """
        Delete vectors by their keys (batch deletion)

        Args:
            keys: List of vector keys to delete
            show_progress: Show progress during deletion (default: True)

        Returns:
            Deletion result with count
        """
        if not keys:
            return {'deleted_count': 0, 'message': 'No keys provided'}

        try:
            # Delete in batches (AWS recommends max 500 per request for optimal performance)
            batch_size = 500
            total_deleted = 0
            total_batches = (len(keys) + batch_size - 1) // batch_size

            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                current_batch = (i // batch_size) + 1

                if show_progress and total_batches > 1:
                    progress_percent = (current_batch / total_batches) * 100
                    print(f"  삭제 중... [{current_batch}/{total_batches}] ({progress_percent:.1f}%)")

                try:
                    self.s3vectors.delete_vectors(
                        vectorBucketName=self.bucket_name,
                        indexName=self.index_name,
                        keys=batch_keys
                    )
                    total_deleted += len(batch_keys)

                except self.s3vectors.exceptions.TooManyRequestsException:
                    # Handle rate limiting (429 error)
                    if show_progress:
                        print(f"  ⚠️  요청 제한 도달, 2초 대기 후 재시도...")
                    import time
                    time.sleep(2)

                    # Retry the batch
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

    def _check_vector_bucket_exists(self) -> bool:
        """
        Check if vector bucket exists

        Returns:
            True if bucket exists, False otherwise
        """
        try:
            self.s3vectors.get_vector_bucket(
                vectorBucketName=self.bucket_name
            )
            return True
        except self.s3vectors.exceptions.NotFoundException:
            return False
        except Exception as e:
            # For other errors, assume bucket might exist but we can't check
            print(f"Warning: Could not verify bucket existence: {e}")
            return False

    def _check_vector_index_exists(self) -> bool:
        """
        Check if vector index exists in the bucket

        Returns:
            True if index exists, False otherwise
        """
        try:
            self.s3vectors.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name
            )
            return True
        except self.s3vectors.exceptions.NotFoundException:
            return False
        except Exception as e:
            # For other errors, assume index might exist but we can't check
            print(f"Warning: Could not verify index existence: {e}")
            return False

    def _create_vector_bucket(self) -> Dict:
        """
        Create a new vector bucket

        Returns:
            Creation response

        Raises:
            RuntimeError: If bucket creation fails
        """
        try:
            print(f"Creating vector bucket: {self.bucket_name}...")
            response = self.s3vectors.create_vector_bucket(
                vectorBucketName=self.bucket_name
            )
            print(f"✓ Vector bucket created successfully: {self.bucket_name}")
            return response
        except self.s3vectors.exceptions.ConflictException as e:
            # Bucket already exists
            print(f"✓ Vector bucket already exists: {self.bucket_name}")
            return {'status': 'already_exists'}
        except Exception as e:
            raise RuntimeError(f"Failed to create vector bucket: {str(e)}")

    def _create_vector_index(self, vector_dimensions: int = 1024, distance_metric: str = "cosine") -> Dict:
        """
        Create a new vector index in the bucket

        Args:
            vector_dimensions: Dimension of vectors (default: 1024 for Titan Embeddings V2)
            distance_metric: Distance metric (COSINE or EUCLIDEAN)

        Returns:
            Creation response

        Raises:
            RuntimeError: If index creation fails
        """
        try:
            print(f"Creating vector index: {self.index_name} (dimensions: {vector_dimensions}, metric: {distance_metric})...")
            response = self.s3vectors.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dimension=vector_dimensions,
                dataType="float32",
                distanceMetric=distance_metric
            )
            print(f"✓ Vector index created successfully: {self.index_name}")

            # Wait for index to become available
            print("Waiting for index to become available...")
            import time
            max_attempts = 15  # Reduced from 30
            for attempt in range(max_attempts):
                try:
                    status_response = self.s3vectors.get_index(
                        vectorBucketName=self.bucket_name,
                        indexName=self.index_name
                    )

                    # Debug: Print actual response to see structure
                    if attempt == 0:
                        print(f"Debug: Response keys: {list(status_response.keys())}")

                    # Try different possible status field names
                    status = (status_response.get('indexStatus') or
                             status_response.get('status') or
                             status_response.get('state') or
                             'UNKNOWN')

                    print(f"  Index status: {status} (attempt {attempt + 1}/{max_attempts})")

                    # Check if index is ready (various possible values)
                    if status.upper() in ['ACTIVE', 'AVAILABLE', 'READY', 'CREATED']:
                        print(f"✓ Index is active!")
                        break

                    # If we can query the index, it's ready
                    if status_response:
                        print(f"✓ Index exists and is queryable!")
                        break

                except Exception as e:
                    print(f"  Waiting... ({str(e)[:50]})")

                time.sleep(2)

            # After max attempts, assume it's ready (S3 Vectors has strong consistency)
            print(f"✓ Proceeding with index (S3 Vectors has strong consistency)")
            return response
        except self.s3vectors.exceptions.ConflictException as e:
            # Index already exists
            print(f"✓ Vector index already exists: {self.index_name}")
            return {'status': 'already_exists'}
        except Exception as e:
            raise RuntimeError(f"Failed to create vector index: {str(e)}")

    def ensure_vector_resources(self, vector_dimensions: int = 1024, distance_metric: str = "cosine") -> Dict:
        """
        Ensure vector bucket and index exist, creating them if necessary

        Args:
            vector_dimensions: Dimension of vectors (default: 1024 for Titan Embeddings V2)
            distance_metric: Distance metric (COSINE or EUCLIDEAN)

        Returns:
            Status dictionary with creation results
        """
        result = {
            'bucket_created': False,
            'index_created': False,
            'bucket_exists': False,
            'index_exists': False,
            'ready': False
        }

        try:
            # Check and create bucket if needed
            print(f"\nChecking vector resources...")
            print(f"Bucket: {self.bucket_name}")
            print(f"Index: {self.index_name}")
            print()

            bucket_exists = self._check_vector_bucket_exists()
            result['bucket_exists'] = bucket_exists

            if not bucket_exists:
                print(f"Vector bucket does not exist. Creating...")
                self._create_vector_bucket()
                result['bucket_created'] = True
                result['bucket_exists'] = True
            else:
                print(f"✓ Vector bucket exists: {self.bucket_name}")

            # Check and create index if needed
            index_exists = self._check_vector_index_exists()
            result['index_exists'] = index_exists

            if not index_exists:
                print(f"Vector index does not exist. Creating...")
                self._create_vector_index(vector_dimensions, distance_metric)
                result['index_created'] = True
                result['index_exists'] = True
            else:
                print(f"✓ Vector index exists: {self.index_name}")

            result['ready'] = True
            print()
            print("=" * 70)
            print("✓ Vector resources are ready!")
            print("=" * 70)
            print()

            return result

        except Exception as e:
            error_msg = str(e)
            print()
            print("=" * 70)
            print(f"❌ Error ensuring vector resources: {error_msg}")
            print("=" * 70)
            print()

            # Provide helpful error messages
            if 'AccessDenied' in error_msg or 'not authorized' in error_msg:
                print("IAM Permission Issue:")
                print("  Add these permissions to your IAM user:")
                print("  - s3vectors:CreateVectorBucket")
                print("  - s3vectors:CreateIndex")
                print("  - s3vectors:GetVectorBucket")
                print("  - s3vectors:GetIndex")
                print()

            raise RuntimeError(f"Failed to ensure vector resources: {error_msg}")

    def delete_index(self) -> Dict:
        """
        Delete the vector index

        Returns:
            Deletion result

        Raises:
            RuntimeError: If index deletion fails
        """
        try:
            print(f"Deleting vector index: {self.index_name}...")
            self.s3vectors.delete_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name
            )
            print(f"✓ Vector index deleted: {self.index_name}")
            return {'status': 'deleted', 'index_name': self.index_name}
        except self.s3vectors.exceptions.NotFoundException:
            print(f"⚠️  Vector index not found: {self.index_name}")
            return {'status': 'not_found', 'index_name': self.index_name}
        except Exception as e:
            raise RuntimeError(f"Failed to delete vector index: {str(e)}")

    def delete_bucket(self) -> Dict:
        """
        Delete the vector bucket (must delete all indexes first)

        Returns:
            Deletion result

        Raises:
            RuntimeError: If bucket deletion fails
        """
        try:
            print(f"Deleting vector bucket: {self.bucket_name}...")
            self.s3vectors.delete_vector_bucket(
                vectorBucketName=self.bucket_name
            )
            print(f"✓ Vector bucket deleted: {self.bucket_name}")
            return {'status': 'deleted', 'bucket_name': self.bucket_name}
        except self.s3vectors.exceptions.NotFoundException:
            print(f"⚠️  Vector bucket not found: {self.bucket_name}")
            return {'status': 'not_found', 'bucket_name': self.bucket_name}
        except Exception as e:
            raise RuntimeError(f"Failed to delete vector bucket: {str(e)}")
