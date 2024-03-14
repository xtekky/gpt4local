
Download models and place them in the [`./models`](/models) folder.

mistral-7b-instruct (v2): https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF   
orca-mini-3b: https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf

What is `q2_0`, `q4_0`, `q5_0`, `q8_0` ?
- Higher quantization 'bit counts' (4 bits or more) generally preserve more quality, whereas lower levels compress the model further, which can lead to a significant loss in quality.
- generally the standard quantization level is `q4_0`.


Download .gguf files
- 7b parameters ~ `8gb` of ram
- 13b parameters ~ `16gb` of ram

the `model` parameter must match the file name of the `.gguf` model you just placed in `./models`, without the `.gguf` extension !


```py
from g4l.local import LocalEngine

engine = LocalEngine(
    gpu_layers = -1,  # use all GPU layers
    cores      = 0    # use all CPU cores
)

response = engine.chat.completions.create(
    model    = 'orca-mini-3b-gguf2-g4_0',
    messages = [{"role": "user", "content": "hi"}],
    stream   = True
)

for token in response:
    print(token.choices[0].delta.content)
```

*code currently based on `llama.cpp`*
