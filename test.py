from g4l.local import LocalEngine

engine   = LocalEngine()
response = engine.chat.completions.create(
    model    = 'mistral-7b-instruct',
    messages = [{"role": "user", "content": "hi"}],
    stream   = True
)

for token in response:
    print(token.choices[0].delta.content)

#print(response.choices[0].message.content)