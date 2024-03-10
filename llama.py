from llama_cpp import Llama
from datetime  import datetime
import time

start = time.time()
engine = Llama(
      model_path="./g4l/local/models/mistral-7b-instruct-v0.1.Q4_0.gguf",
      #n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

print('loaded, took: %.2f seconds' % (time.time() - start))

completion = engine.create_chat_completion(
      messages = [
          {"role": "system", "content": f"You are g4f-1.0, an AI assistant, you are currently talking to a human, you respond to the last user question, using the conversation history as context if needed. Be consise and logical. Current date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"},
          {
              "role": "user",
              "content": "hi"
          }
      ],
      stream = True
)

print('completion started, took: %.2f seconds' % (time.time() - start))

for token in completion:
      print(token["choices"][0]["delta"]["content"])

print('completion finished, took: %.2f seconds' % (time.time() - start))