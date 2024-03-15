import os

from llama_cpp import Llama
from ._docs    import DocumentRetriever

class LocalProvider:
    @staticmethod
    def create_completion(model, messages, document_retriever: DocumentRetriever = None, **kwargs):
        model_path      = model + '.gguf'
        model_dir       = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models/')
        full_model_path = os.path.join(model_dir, model_path)
        
        if not os.path.isfile(full_model_path):
            raise FileNotFoundError(f"Model file '{full_model_path}' not found.")
        
        engine = Llama(
            model_path  =   full_model_path,
            chat_format =   "mistral-instruct",
            verbose     =   False,
            n_gpu_layers=   kwargs.get('n_gpu_layers', 0),      # Offload all layers to GPU
            n_threads   =   kwargs.get('threads', None),        # Use all available CPU threads
            use_mmap    =   kwargs.get('use_mmap', True),       # Use mmap for faster model loading
            use_mlock   =   kwargs.get('use_mlock', False),     # Lock the model in memory to prevent swapping
            offload_kqv =   kwargs.get('offload_kqv', True),    # Offload K, Q, V to GPU
            n_ctx       =   kwargs.get('context_window', 4900), # max context window
        )
        
        if document_retriever:
            prompt = document_retriever.retrieve_for_llm(messages[-1]['content'])
            messages[-1]['content'] = prompt

        completion = engine.create_chat_completion(
                messages    = messages,
                stream      = True,
                temperature = kwargs.get('temperature', 0.8),
                max_tokens  = kwargs.get('max_tokens', 4900),
        )

        for token in completion:
            val = token['choices'][0]['delta'].get('content')
            if val:
                yield val
                
__all__ = ['LocalProvider']