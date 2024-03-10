# from https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-bindings/python/gpt4all/gpt4all.py
# modified version to work with g4f-local

import os, sys

from contextlib          import contextmanager
from ...typing           import Optional, Union, Dict, List, Any
from ._pyllmodel         import (
    LLModel, 
    ResponseCallbackType, 
    empty_response_callback, 
    Iterable
)

DEFAULT_MODEL_CONFIG = {
    'systemPrompt'  : '',
    'promptTemplate': '### Human: \n{0}\n### Assistant:\n',
}

ConfigType  = Dict[str, str]
MessageType = Dict[str, str]

class Engine:
    def __init__(
        self,
        model_name  : str,
        model_path  : Optional[Union[str, os.PathLike[str]]] = None,
        model_type  : Optional[str] = None,
        n_threads   : Optional[int] = None,
        device      : Optional[str] = 'cpu',
        n_ctx       : int = 2048,
        ngl         : int = 100,
        verbose     : bool = False,):

        self.model_type         = model_type
        self.config: ConfigType = self.retrieve_model(model_name, model_path=model_path, verbose=verbose)
        self.model              = LLModel(self.config["path"], n_ctx, ngl)
        
        if device is not None and device != 'cpu':
            self.model.init_gpu(device)
        
        self.model.load_model()
        if n_threads is not None:
            self.model.set_thread_count(n_threads)

        self._is_chat_session_activated: bool = False
        self.current_chat_session: List[MessageType] = empty_chat_session()
        self._current_prompt_template: str = "{0}"

    @staticmethod
    def retrieve_model(
        model_name: str,
        model_path: Optional[Union[str, os.PathLike[str]]] = None,
        verbose: bool = False) -> ConfigType:

        model_filename     = append_extension_if_missing(model_name)
        config: ConfigType = DEFAULT_MODEL_CONFIG
        model_path         = str(model_path).replace("\\", "\\\\")
        model_dest         = os.path.join(model_path, model_filename).replace("\\", "\\\\")
        
        if os.path.exists(model_dest):
            config.pop("url", None)
            config["path"] = model_dest
            if verbose:
                print("Found model file at", model_dest, file=sys.stderr)
        
        return config

    def generate(
        self,
        prompt          : str,
        max_tokens      : int = 200,
        temp            : float = 0.7,
        top_k           : int = 40,
        top_p           : float = 0.4,
        repeat_penalty  : float = 1.18,
        repeat_last_n   : int = 64,
        n_batch         : int = 8,
        n_predict       : Optional[int] = None,
        streaming       : bool = False,
        callback        : ResponseCallbackType = empty_response_callback) -> Union[str, Iterable[str]]:

        generate_kwargs: Dict[str, Any] = dict(
            temp            = temp,
            top_k           = top_k,
            top_p           = top_p,
            repeat_penalty  = repeat_penalty,
            repeat_last_n   = repeat_last_n,
            n_batch         = n_batch,
            n_predict       = n_predict if n_predict is not None else max_tokens
        )

        if self._is_chat_session_activated:
            # check if there is only one message, i.e. system prompt:
            generate_kwargs["reset_context"] = len(self.current_chat_session) == 1
            self.current_chat_session.append({"role": "user", "content": prompt})

            prompt = self._format_chat_prompt_template(
                messages=self.current_chat_session[-1:],
                default_prompt_header=self.current_chat_session[0]["content"]
                if generate_kwargs["reset_context"]
                else "",
            )
        else:
            generate_kwargs["reset_context"] = True

        output_collector: List[MessageType]
        output_collector = [
            {"content": ""}
        ]

        if self._is_chat_session_activated:
            self.current_chat_session.append({"role": "assistant", "content": ""})
            output_collector = self.current_chat_session

        def _callback_wrapper(
            callback        : ResponseCallbackType,
            output_collector: List[MessageType]) -> ResponseCallbackType:
            
            def _callback(token_id: int, response: str) -> bool:
                nonlocal callback, output_collector

                output_collector[-1]["content"] += response

                return callback(token_id, response)

            return _callback

        if streaming:
            return self.model.prompt_model_streaming(
                prompt=prompt,
                callback=_callback_wrapper(callback, output_collector),
                **generate_kwargs,
            )

        self.model.prompt_model(
            prompt=prompt,
            callback=_callback_wrapper(callback, output_collector),
            **generate_kwargs,
        )

        return output_collector[-1]["content"]

    @contextmanager
    def chat_session(
        self,
        system_prompt: str = "",
        prompt_template: str = "",
    ):
        """
        Context manager to hold an inference optimized chat session with a GPT4All model.

        Args:
            system_prompt: An initial instruction for the model.
            prompt_template: Template for the prompts with {0} being replaced by the user message.
        """
        # Code to acquire resource, e.g.:
        self._is_chat_session_activated = True
        self.current_chat_session = empty_chat_session(system_prompt or self.config["systemPrompt"])
        self._current_prompt_template = prompt_template or self.config["promptTemplate"]
        try:
            yield self
        finally:
            # Code to release resource, e.g.:
            self._is_chat_session_activated = False
            self.current_chat_session = empty_chat_session()
            self._current_prompt_template = "{0}"

    def _format_chat_prompt_template(
        self,
        messages: List[MessageType],
        default_prompt_header: str = "",
        default_prompt_footer: str = "",
    ) -> str:
        """
        Helper method for building a prompt from list of messages using the self._current_prompt_template as a template for each message.

        Args:
            messages:  List of dictionaries. Each dictionary should have a "role" key
                with value of "system", "assistant", or "user" and a "content" key with a
                string value. Messages are organized such that "system" messages are at top of prompt,
                and "user" and "assistant" messages are displayed in order. Assistant messages get formatted as
                "Response: {content}".

        Returns:
            Formatted prompt.
        """

        if isinstance(default_prompt_header, bool):
            import warnings

            warnings.warn(
                "Using True/False for the 'default_prompt_header' is deprecated. Use a string instead.",
                DeprecationWarning,
            )
            default_prompt_header = ""

        if isinstance(default_prompt_footer, bool):
            import warnings

            warnings.warn(
                "Using True/False for the 'default_prompt_footer' is deprecated. Use a string instead.",
                DeprecationWarning,
            )
            default_prompt_footer = ""

        full_prompt = default_prompt_header + "\n\n" if default_prompt_header != "" else ""

        for message in messages:
            if message["role"] == "user":
                user_message = self._current_prompt_template.format(message["content"])
                full_prompt += user_message
            if message["role"] == "assistant":
                assistant_message = message["content"] + "\n"
                full_prompt += assistant_message

        full_prompt += "\n\n" + default_prompt_footer if default_prompt_footer != "" else ""

        return full_prompt

def empty_chat_session(system_prompt: str = "") -> List[MessageType]:
    return [{"role": "system", "content": system_prompt}]

def append_extension_if_missing(model_name):
    if not model_name.endswith((".bin", ".gguf")):
        model_name += ".gguf"
    return model_name

__all__ = ['LocalEngine']