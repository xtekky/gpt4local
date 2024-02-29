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
