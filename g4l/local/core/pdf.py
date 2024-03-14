import shutil
import pathlib
import time
from hashlib import md5
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
import logging

Settings.chunk_size = 1024
current_file_path = pathlib.Path(__file__).parent.resolve()
BASE_ADDR = current_file_path / "../../.."
modes = {
    "subtle": 1,
    "default": 2,
    "aggressive": 5,
    "very-aggressive": 10
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, files: List[str] = [], verbose: bool = False, mode: str = "default", embed_model: str = None, reset_storage: bool = False) -> None:
        self.similarity_index = modes[mode]
        self.verbose = verbose
        self.init_time = time.time()
        
        if embed_model:
            Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

        storage_id = md5(":".join(files).encode()).hexdigest()
        self.persist_dir = BASE_ADDR / f"files/storage/storage.{storage_id}"

        if not self.persist_dir.exists():
            self._index_documents(files)
        else:
            self._load_index()

    def _index_documents(self, files: List[str]) -> None:
        if self.verbose:
            start_time = time.time()

        documents = SimpleDirectoryReader(str(BASE_ADDR / "files")).load_data()

        if self.verbose:
            read_documents_time = time.time() - start_time
            logger.info(f"Read documents: {read_documents_time:.4f}s")

        self.index = VectorStoreIndex.from_documents(
            documents, transformations=[SentenceSplitter(chunk_size=512)]
        )

        if self.verbose:
            index_documents_time = time.time() - read_documents_time
            logger.info(f"Indexed documents: {index_documents_time:.4f}s")

        self.index.storage_context.persist(persist_dir=str(self.persist_dir))

    def _load_index(self) -> None:
        if self.verbose:
            start_time = time.time()

        storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
        self.index = load_index_from_storage(storage_context)

        if self.verbose:
            load_index_time = time.time() - start_time
            logger.info(f"Loaded index: {load_index_time:.4f}s")

    def retrieve(self, query: str) -> str:
        query_start_time = time.time()

        try:
            retriever = self.index.as_retriever(similarity_top_k=self.similarity_index)
            response = retriever.retrieve(query)
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            return "An error occurred while processing the query."

        if self.verbose:
            query_time = time.time() - query_start_time
            logger.info(f"Query time: {query_time:.4f}s")
            total_time = time.time() - self.init_time
            logger.info(f"Total time: {total_time:.4f}s")

        return response
    
    def retrieve_for_llm(self, query_str: str):
        retrieval_data   =  self.retrieve(query_str)
        context_batches  = ''
        
        for node_with_score in retrieval_data:
            node = node_with_score.node
            score = node_with_score.score

            text = node.text

            metadata = node.metadata
            page_label = metadata['page_label']
            file_name = metadata['file_name']

            batch = '\n'.join([f'content: \n{text}', f'----\npage_label: {page_label}', f'file name: {file_name}', f'similarity score: {score}'])
            context_batches += (batch + '\n---')
        
        prompt = (f'Context information is below.\n'
            + '---------------------\n'
            + f'{context_batches}\n'
            + '---------------------\n'
            + 'Given the context information and not prior knowledge, answer the query.\n'
            + f'Query: {query_str}\n'
            + 'Answer: ')
        
        return prompt


    
__all__ = ['DocumentRetriever']