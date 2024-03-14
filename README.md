### Requirements
g4l needs llama.cpp python bindings, which you can install with pip:

```
pip3 install -U llama-cpp-python
```

### Download models
Download models and place them in the [`./models`](/models) folder.

You can find a majority of the models on HuggingFace, look for `GGUF`, models, which is the required format.

https://huggingface.co/TheBloke has a lot of quantized `.gguf` models available.

- mistral-7b-instruct (v2): https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF   
- orca-mini-3b: https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf

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

### Best Models ?
according to https://chat.lmsys.org/ 

- Best **`7b`** model is `Mistral-7B-Instruct-v0.2`
- Best opensource  model is `Qwen1.5-72B-Chat` | available [here](https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GGUF/tree/main)

### Benchmark

```
pc: mac air m2
cpu/gpu: m2 chip
cores: all (8)
gpu layers: all
gpu offload: 100%

Model                = mistral-7b-instruct
Number of iterations = 5
Average loading time = 1.85s
Average total tokens = 48.20
Average total time   = 5.34s
Average speed        = 9.02 t/s
```


*code currently based on `llama.cpp`*
