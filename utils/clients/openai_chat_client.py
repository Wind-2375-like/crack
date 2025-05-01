from openai import OpenAI
from utils.clients.base_chat_client import BaseChatClient

class OpenAIChatClient(BaseChatClient):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def _process_message(self, messages):
        processed_messages = []
        for role, content in messages:
            processed_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        return processed_messages

    def create(self, messages, **kwargs):
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        n = kwargs.get("n", 1)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        model_name = kwargs.get("model_name", "gpt-4o")
        max_tokens = kwargs.get("max_tokens", 2048)
        response_format = kwargs.get("response_format", {"type": "text"})

        return self.client.chat.completions.create(
            model=model_name,
            messages=self._process_message(messages),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=n,
            response_format=response_format,
            max_tokens=max_tokens,
        )