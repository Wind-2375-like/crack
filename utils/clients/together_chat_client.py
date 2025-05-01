from together import Together
from utils.clients.base_chat_client import BaseChatClient

class TogetherChatClient(BaseChatClient):
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)
        self.model_name_to_end_token = {
            "meta-llama/Llama-3.2-3B-Instruct-Turbo": ["<|eot_id|>","<|eom_id|>"],
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ["<|eot_id|>","<|eom_id|>"],
            "meta-llama/Meta-Llama-3-8B-Instruct-Turbo": ["<|eot_id|>"],
            "Qwen/Qwen2-72B-Instruct": ["<|im_start|>","<|im_end|>"],
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ["<|eot_id|>","<|eom_id|>"],
            "mistralai/Mixtral-8x22B-Instruct-v0.1": ["</s>","[/INST]"],
            "mistralai/Mixtral-8x7B-Instruct-v0.1": ["</s>","[/INST]"],
        }

    def _process_message(self, messages):
        processed_messages = []
        for role, content in messages:
            processed_messages.append({"role": role, "content": content})
        return processed_messages

    def create(self, messages, **kwargs):
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        top_k = kwargs.get("top_k", 1)
        n = kwargs.get("n", 1)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        model_name = kwargs.get("model_name", "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")
        max_tokens = kwargs.get("max_tokens", 2048)

        return self.client.chat.completions.create(
            model=model_name,
            messages=self._process_message(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=self.model_name_to_end_token[model_name],
        )