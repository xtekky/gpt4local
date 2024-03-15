import os
from typing import Iterator, List, Dict, Any
from llama_cpp import Llama
from ._docs import DocumentRetriever

class LocalProvider:
    """
    A class that provides local language model functionality using the Llama library.
    """

    @staticmethod
    def create_completion(model: str, messages: List[Dict[str, str]], document_retriever: DocumentRetriever = None, **kwargs: Any) -> Iterator[str]:
        """
        Creates a completion using the specified model and messages.

        Args:
            model (str): The name of the model file (without the '.gguf' extension).
            messages (List[Dict[str, str]]): A list of message dictionaries, where each dictionary contains a 'role' and 'content' key.
            document_retriever (DocumentRetriever, optional): An instance of the DocumentRetriever class for retrieving relevant documents. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Llama constructor and create_chat_completion method.

        Returns:
            Iterator[str]: An iterator yielding the generated completion tokens.

        Raises:
            FileNotFoundError: If the specified model file is not found.
        """
        model_path = model + '.gguf'
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models/')
        full_model_path = os.path.join(model_dir, model_path)

        if not os.path.isfile(full_model_path):
            raise FileNotFoundError(f"Model file '{full_model_path}' not found.")

        # Initialize the Llama engine with the specified model and parameters
        engine = Llama(
            model_path=full_model_path,
            chat_format="mistral-instruct",
            verbose=False,
            n_gpu_layers=kwargs.get('n_gpu_layers', 0),
            n_threads=kwargs.get('threads', None),
            use_mmap=kwargs.get('use_mmap', True),
            use_mlock=kwargs.get('use_mlock', False),
            offload_kqv=kwargs.get('offload_kqv', True),
            n_ctx=kwargs.get('context_window', 4900),
        )

        if document_retriever:
            # Retrieve relevant documents and update the last message content
            prompt = document_retriever.retrieve_for_llm(messages[-1]['content'])
            messages[-1]['content'] = prompt

        # Generate the completion using the Llama engine
        completion = engine.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=kwargs.get('temperature', 0.8),
            max_tokens=kwargs.get('max_tokens', 4900),
        )

        # Yield the generated completion tokens
        for token in completion:
            val = token['choices'][0]['delta'].get('content')
            if val:
                yield val

__all__ = ['LocalProvider']