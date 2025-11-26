import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration from environment variables"""

    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    # S3 Vectors Configuration
    S3_VECTOR_BUCKET_NAME = os.getenv('S3_VECTOR_BUCKET_NAME')
    S3_VECTOR_INDEX_NAME = os.getenv('S3_VECTOR_INDEX_NAME')

    # Bedrock Model IDs
    BEDROCK_EMBEDDING_MODEL_ID = os.getenv('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
    BEDROCK_LLM_MODEL_ID = os.getenv('BEDROCK_LLM_MODEL_ID', 'global.anthropic.claude-sonnet-4-20250514-v1:0')

    # Application Settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 3))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'S3_VECTOR_BUCKET_NAME',
            'S3_VECTOR_INDEX_NAME'
        ]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
