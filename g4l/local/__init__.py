import random, string, time, re

from ..typing import Union, Iterator, Messages
from ..stubs  import ChatCompletion, ChatCompletionChunk
from ._engine import LocalProvider
from ._docs   import DocumentRetriever

IterResponse = Iterator[Union[ChatCompletion, ChatCompletionChunk]]

def read_json(text: str) -> dict:
    match = re.search(r"```(json|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    return text

def iter_response(
    response: Iterator[str],
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None) -> IterResponse:
    
    content = ""
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    for idx, chunk in enumerate(response):
        content += str(chunk)
        if max_tokens is not None and idx + 1 >= max_tokens:
            finish_reason = "length"
        first = -1
        word = None
        if stop is not None:
            for word in list(stop):
                first = content.find(word)
                if first != -1:
                    content = content[:first]
                    break
            if stream and first != -1:
                first = chunk.find(word)
                if first != -1:
                    chunk = chunk[:first]
                else:
                    first = 0
        if first != -1:
            finish_reason = "stop"
        if stream:
            yield ChatCompletionChunk(chunk, None, completion_id, int(time.time()))
        if finish_reason is not None:
            break
    finish_reason = "stop" if finish_reason is None else finish_reason
    if stream:
        yield ChatCompletionChunk(None, finish_reason, completion_id, int(time.time()))
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = read_json(content)
        yield ChatCompletion(content, finish_reason, completion_id, int(time.time()))

def filter_none(**kwargs):
    for key in list(kwargs.keys()):
        if kwargs[key] is None:
            del kwargs[key]
    return kwargs

class LocalEngine():
    def __init__(
        self,
        gpu_layers: int = 0,
        cores: int = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        offload_kqv: bool = True,
        context_window: int = 4900, 
        document_retriever: DocumentRetriever = None, **kwargs) -> None:
        
        self.gpu_layers = gpu_layers
        self.cores = cores
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.offload_kqv = offload_kqv
        self.context_window = context_window
        self.document_retriever: DocumentRetriever = document_retriever
        self.chat: Chat = Chat(self)
        
class Completions():
    def __init__(self, client: LocalEngine):
        self.client: LocalEngine = client

    def create(
        self,
        messages: Messages,
        model: str,
        stream: bool = False,
        response_format: dict = None,
        max_tokens: int = None,
        stop: Union[list[str], str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        stop = [stop] if isinstance(stop, str) else stop
        response = LocalProvider.create_completion(
            model, messages, self.client.document_retriever,
            **filter_none(
                max_tokens=max_tokens,
                stop=stop,
                n_gpu_layers=self.client.gpu_layers,
                threads=self.client.cores,
                use_mmap=self.client.use_mmap,
                use_mlock=self.client.use_mlock,
                offload_kqv=self.client.offload_kqv,
                n_ctx=self.client.context_window
            ),
            **kwargs
        )
        response = iter_response(response, stream, response_format, max_tokens, stop)
        return response if stream else next(response)
    
class Chat():
    completions: Completions

    def __init__(self, client: LocalEngine):
        self.completions = Completions(client)