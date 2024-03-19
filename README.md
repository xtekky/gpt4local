<img width="1148" alt="image" src="https://github.com/gpt4free/gpt4local/assets/98614666/df91ae5f-fa4a-4eb3-9dca-f9d38aa3764b">

`g4l` is a high-level Python library that allows you to run language models using the `llama.cpp` bindings. It is a sister project to @gpt4free, which also provides AI, but using internet and external providers, aswell as additional feature such as text retrieval from documents.

pull requests are welcome !!

#### Roadmap

- [ ] Gui / playground
- [ ] Support function calling & image models
- [ ] tts / stt models
- [ ] Blog article creator (use of multiple queries to produce a qualitative blog atricle with efficient style prompting and context retrieval)
- [ ] Allow for passing of more arguments
- [ ] Improve compatibility / Unittests.
- [ ] Native binding implementation / more low level usage of `llama-cpp-python`
- [ ] Ability to finetune models on datasets / dataset generator
- [ ] Optimise for devices with low memory and computing (current min ram is 8gb & gpu is preferred)
- [ ] Blog articles explaining usage, and how llm's work.
- [ ] Better model list / optimised parameters
- [ ] Create custom local benchmarking.


## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Downloading Models](#downloading-models)
   - [Model Quantization](#model-quantization)
   - [Best Models](#best-models)
4. [Usage](#usage)
   - [Basic Usage](#basic-usage)
   - [Chat With Documents](#chat-with-documents)
   - [Document Retrieval](#document-retrieval)
   - [Advanced Usage](#advanced-usage)
5. [Benchmark](#benchmark)
6. [Why gpt4local?](#why-gpt4local)

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
3. Install the required dependencies:
```
pip install -r requirements.txt
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

### Basic Usage
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

### Chat With Documents

```py
from g4l.local import LocalEngine, DocumentRetriever

engine = LocalEngine(
    gpu_layers = -1,  # use all GPU layers
    cores      = 0,   # use all CPU cores
    document_retriever = DocumentRetriever(
        files       = ['einstein-albert.pdf'], 
        embed_model = 'SmartComponents/bge-micro-v2', # https://huggingface.co/spaces/mteb/leaderboard
    )
)

response = engine.chat.completions.create(
    model    = 'mistral-7b-instruct',
    messages = [
        {
            "role": "user", "content": "how was einstein's work in the laboratory"
        }
    ],
    stream   = True
)

for token in response:
    print(token.choices[0].delta.content or "", end="", flush=True)
```

! The embeddings model will be downloaded upon first use, but it is really small and lightweight.

### Document Retrieval
G4L provides a `DocumentRetriever` class that allows you to retrieve relevant information from documents based on a query. Here's an example of how to use it:

```py
from g4l.local import DocumentRetriever

engine = DocumentRetriever(
    files=['einstein-albert.txt'], 
    embed_model='SmartComponents/bge-micro-v2', # https://huggingface.co/spaces/mteb/leaderboard
    verbose=True,
)

retrieval_data = engine.retrieve('what inventions did he do')

for node_with_score in retrieval_data:
    node = node_with_score.node
    score = node_with_score.score
    text = node.text
    metadata = node.metadata
    page_label = metadata['page_label']
    file_name = metadata['file_name']
    
    print(f"Text: {text}")
    print(f"Score: {score}")
    print(f"Page Label: {page_label}")
    print(f"File Name: {file_name}")
    print("---")
```

You can also get a ready-to-go prompt for the language model using the `retrieve_for_llm` method:

```py
retrieval_data = engine.retrieve_for_llm('what inventions did he do')
print(retrieval_data)
```

The prompt template used by `retrieve_for_llm` is as follows:

```py
prompt = (f'Context information is below.\n'
    + '---------------------\n'
    + f'{context_batches}\n'
    + '---------------------\n'
    + 'Given the context information and not prior knowledge, answer the query.\n'
    + f'Query: {query_str}\n'
    + 'Answer: ')
```

### Advanced Usage
G4L provides several configuration options to customize the behavior of the `LocalEngine`. Here are some of the available options:

- `gpu_layers`: The number of layers to offload to the GPU. Use `-1` to offload all layers.
- `cores`: The number of CPU cores to use. Use `0` to use all available cores.
- `use_mmap`: Whether to use memory mapping for faster model loading. Default is `True`.
- `use_mlock`: Whether to lock the model in memory to prevent swapping. Default is `False`.
- `offload_kqv`: Whether to offload key, query, and value tensors to the GPU. Default is `True`.
- `context_window`: The maximum context window size. Default is `4900`.

You can pass these options when creating an instance of `LocalEngine`:

```py
engine = LocalEngine(
    gpu_layers = -1,
    cores      = 0,
    use_mmap   = True,
    use_mlock  = False,
    offload_kqv= True,
    context_window = 4900
)
```

## Benchmark
Benchmark ran on a 2022 MacBook Air M2, 8GB RAM.

```
PC: Mac Air M2
CPU/GPU: M2 chip
Cores: All (8)
GPU Layers: All
GPU Offload: 100%

No power:
Model: mistral-7b-instruct-v2
Number of iterations: 5
Average loading time: 1.85s
Average total tokens: 48.20
Average total time: 5.34s
Average speed: 9.02 t/s

With power:
Model: mistral-7b-instruct-v2
Number of iterations: 5
Average loading time: 1.88s
Average total tokens: 317
Average total time: 17.7s
Average speed: 17.9 t/s
```

## Why gpt4local?
- I have coded G4L in a way that you can use language models in a very familiar way with quick installation, while preserving maximum performance.
- Using the direct Python bindings, I was able to **max out** the performance by using 100% GPU, CPU, and RAM.
- I tried different 3rd party packages that wrap `llama.cpp`, like LmStudio, which still had great performance but in my case a speed of ~`7.83` tokens/s in contrast to `9.02` t/s with native llama.cpp Python bindings.
