from utils.clients.base_chat_client import BaseChatClient
from utils.handlers.meta_llama_handler import MetaLlamaHandler
from utils.handlers.meta_vllama_handler import MetaVLlamaHandler
from utils.handlers.qwen_handler import QwenHandler
from utils.handlers.qwen3_handler import Qwen3Handler
from utils.handlers.olmo_handler import OlmoHandler

class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return DotDict(value)  # Recursively convert nested dictionaries
        elif isinstance(value, list):  # If the value is a list, convert each element
            return [DotDict(i) if isinstance(i, dict) else i for i in value]
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LocalChatClient(BaseChatClient):
    MODEL_HANDLERS = {
        'meta-llama/Meta-Llama-3-8B-Instruct': MetaLlamaHandler,
        'meta-llama/Meta-LlaMAV-3.2-90B-Instruct': MetaLlamaHandler,
        'meta-llama/Llama-3.2-1B-Instruct': MetaLlamaHandler,
        'meta-llama/Llama-3.2-3B-Instruct': MetaLlamaHandler,
        'meta-llama/Llama-3.2-11B-Vision-Instruct': MetaVLlamaHandler,
        'Qwen/Qwen2.5-1.5B-Instruct': QwenHandler,
        'Qwen/Qwen2.5-3B-Instruct': QwenHandler,
        'Qwen/Qwen2.5-7B-Instruct': QwenHandler,
        'Qwen/Qwen2.5-14B-Instruct': QwenHandler,
        "allenai/OLMo-2-0425-1B-Instruct": OlmoHandler,
        "allenai/OLMo-2-1124-7B-Instruct": OlmoHandler,
        'Qwen/Qwen3-1.7B': Qwen3Handler,
        'Qwen/Qwen3-1.7B-Thinking': Qwen3Handler,
        'Qwen/Qwen3-4B': Qwen3Handler,
        'Qwen/Qwen3-4B-Thinking': Qwen3Handler,
        'Qwen/Qwen3-8B': Qwen3Handler,
        'Qwen/Qwen3-8B-Thinking': Qwen3Handler,
        # Add more models and handlers as needed
    }

    def __init__(self, model_name):
        self.model_name = model_name
        assert model_name in self.MODEL_HANDLERS, f"Model {model_name} is not supported."
        handler_class = self.MODEL_HANDLERS[model_name]
        self.handler = handler_class(model_name)

    def create(self, messages, **kwargs):
        formatted_input = self.handler.format_messages(messages)
        response_texts = self.handler.generate_response(formatted_input, **kwargs)
        return DotDict({'choices': [{'message': {'content': response_text}} for response_text in response_texts]})