"""Text chunking module using LangChain RecursiveCharacterTextSplitter"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


class TextSplitter:
    """Split text into chunks with metadata"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize text splitter

        Args:
            chunk_size: Maximum chunk size in characters (default from config)
            chunk_overlap: Number of overlapping characters (default from config)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=['\n\n', '\n', ' ', '']
        )

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into chunks with preserved metadata

        Args:
            documents: List of documents with 'text' and 'metadata' keys

        Returns:
            List of chunks with content and enriched metadata
        """
        all_chunks = []
        global_chunk_id = 0

        for doc in documents:
            text = doc['text']
            base_metadata = doc['metadata']

            # Split text into chunks
            chunks = self.splitter.split_text(text)

            # Add metadata to each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    'chunk_id': f"{base_metadata['document']}_chunk_{global_chunk_id}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk_text)
                }

                all_chunks.append({
                    'content': chunk_text,
                    'metadata': chunk_metadata
                })

                global_chunk_id += 1

        return all_chunks

    def get_chunk_preview(self, chunk: Dict, max_length: int = 100) -> str:
        """
        Get a preview of chunk content

        Args:
            chunk: Chunk dictionary with 'content' key
            max_length: Maximum preview length

        Returns:
            Preview string
        """
        content = chunk['content']
        if len(content) <= max_length:
            return content
        return content[:max_length] + '...'
