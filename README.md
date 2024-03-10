
Download models and place them in the [`./models`](/models) folder.

mistral-7b-instruct: https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf  
orca-mini-3b: https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf

```py
from g4l.local import LocalEngine

engine = LocalEngine()

response = engine.chat.completions.create(
    model    = 'orca-mini-3b',
    messages = [{"role": "user", "content": "hi"}],
    stream   = True
)

for token in response:
    print(token.choices[0].delta.content)
```

*code currently based on llama.cpp*
