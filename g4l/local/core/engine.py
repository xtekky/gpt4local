import os
from .local import Engine

models = {
    'mistral-7b-instruct': 'mistral-7b-instruct-v0.1.Q4_0.gguf',
    'orca-mini-3b': 'orca-mini-3b-gguf2-q4_0.gguf'
}

class LocalProvider:
    @staticmethod
    def create_completion(model, messages, stream, **kwargs):
        # Check if the model exists
        if model not in models:
            raise ValueError(f"Model '{model}' not found.")
            
        model_path = models[model]
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        full_model_path = os.path.join(model_dir, model_path)
        
        if not os.path.isfile(full_model_path):
            raise FileNotFoundError(f"Model file '{full_model_path}' not found.")
        
        model = Engine(model_path,
                               n_threads=8,
                               verbose=True,
                               model_path=model_dir)
        
        system_template = next((message['content'] for message in messages if message['role'] == 'system'), 
                               'A chat between a curious user and an artificial intelligence assistant.')
        
        prompt_template = 'USER: {0}\nASSISTANT: '
        
        conversation = '\n'.join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages) + "\nASSISTANT: "
        
        with model.chat_session(system_template, prompt_template):
            if stream:
                for token in model.generate(conversation, streaming=True):
                    yield token
            else:
                yield model.generate(conversation)