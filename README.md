<img width="1148" alt="image" src="https://github.com/gpt4free/gpt4local/assets/98614666/df91ae5f-fa4a-4eb3-9dca-f9d38aa3764b">

G4L is a high level Python library that allows you to run language models using the `llama.cpp bindings`. It is a sister project to @gpt4free, which also provides AI, but using internet and external providers.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Downloading Models](#downloading-models)
   - [Model Quantization](#model-quantization)
   - [Best Models](#best-models)
4. [Usage](#usage)
5. [Benchmark](#benchmark)

## Requirements

To use G4L, you need to have the llama.cpp Python bindings installed. You can install them using pip:

```
pip3 install -U llama-cpp-python
```

## Installation

1. Clone the G4L repository:

```
git clone https://github.com/gpt4free/gpt4local
```

2. Navigate to the cloned directory:

```
cd gpt4local
```

## Downloading Models

1. Download the desired models in the `GGUF` format from [HuggingFace](https://huggingface.co/). You can find a variety of quantized `.gguf` models on [TheBloke's page](https://huggingface.co/TheBloke).

2. Place the downloaded models in the [`./models`](/models) folder.

Some popular models include:
- [mistral-7b-instruct (v2)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [orca-mini-3b](https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf)

### Model Quantization

The models are available in different quantization levels, such as `q2_0`, `q4_0`, `q5_0`, and `q8_0`. Higher quantization 'bit counts' (4 bits or more) generally preserve more quality, whereas lower levels compress the model further, which can lead to a significant loss in quality. The standard quantization level is `q4_0`.

Keep in mind the memory requirements for different model sizes:
- 7b parameters ~ `8gb` of RAM
- 13b parameters ~ `16gb` of RAM

### Best Models

According to [chat.lmsys.org](https://chat.lmsys.org/), the best models are:
- Best **`7b`** model: `Mistral-7B-Instruct-v0.2`
- Best opensource model: `Qwen1.5-72B-Chat` ([available here](https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GGUF/tree/main))

## Usage

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

Note: The `model` parameter must match the file name of the `.gguf` model you placed in `./models`, without the `.gguf` extension!

## Benchmark
benchmark ran on a 2022 mac book air m2, 8gb ram.

```
PC: Mac Air M2
CPU/GPU: M2 chip
Cores: All (8)
GPU Layers: All
GPU Offload: 100%

no power:
Model: mistral-7b-instruct-v2
Number of iterations: 5
Average loading time: 1.85s
Average total tokens: 48.20
Average total time: 5.34s
Average speed: 9.02 t/s

with power:
Model: mistral-7b-instruct-v2
Number of iterations: 5
Average loading time: 1.88s
Average total tokens: 317
Average total time: 17.7s
Average speed: 17.9 t/s
```

## Document Retrieval

```py
from g4l.local import DocumentRetriever

engine = DocumentRetriever(
    files=['einstein-albert.txt'], 
    embed_model='SmartComponents/bge-micro-v2', # https://huggingface.co/spaces/mteb/leaderboard
    verbose=True,
    reset_storage = True
)
retrieval_data = engine.retrieve('what inventions did he do')

for node_with_score in retrieval_data:
    node = node_with_score.node
    score = node_with_score.score

    # Access the text content
    text = node.text

    # Access the metadata
    metadata = node.metadata
    page_label = metadata['page_label']
    file_name = metadata['file_name']
    # ... access other metadata fields as needed

    # Print or process the extracted information
    print(f"Text: {text}")
    print(f"Score: {score}")
    print(f"Page Label: {page_label}")
    print(f"File Name: {file_name}")
    print("---")
```

get a ready to go prompt

```py
retrieval_data = engine.retrieve_for_llm('what inventions did he do')
print(retrieval_data)
```

prompt template:

```py
prompt = (f'Context information is below.\n'
    + '---------------------\n'
    + f'{context_batches}\n'
    + '---------------------\n'
    + 'Given the context information and not prior knowledge, answer the query.\n'
    + f'Query: {query_str}\n'
    + 'Answer: ')
```

## Why gpt4local ?
- I have coded g4l in a way that you can use language model in a very familiar way with quick installation, while preserving maximum performance.
- Using the direct python bindings, i was able to **max out** the performance by using 100% gpu, cpu and ram.
- I tried different 3d party pacakges that wrap `llama.cpp`, like LmStudio, which still had a great performance but in my case a speed of ~`7.83` tokens/s in contrast to `9.02` t/s in with native llama.cpp python bindings.
