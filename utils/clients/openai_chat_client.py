from openai import OpenAI
from utils.clients.base_chat_client import BaseChatClient
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types import CompletionUsage

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

        # First, prepare a dictionary of parameters common to all calls.
        common_params = {
            "model": model_name,
            "messages": self._process_message(messages),
            "response_format": response_format,
        }

        # Now, add the correct token parameter based on the model name.
        if model_name == "gpt-4.1-mini":
            common_params['max_tokens'] = max_tokens
            common_params['temperature'] = temperature
            common_params["top_p"] = top_p,
            common_params["frequency_penalty"] = frequency_penalty,
            common_params["presence_penalty"] = presence_penalty,
        else:
            common_params['max_completion_tokens'] = max_tokens

        # Check if looping is required (for non-gpt-4.1-mini models with n > 1)
        if model_name != "gpt-4.1-mini" and n > 1:
            aggregated_choices = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            last_response = None

            for i in range(n):
                # Make the API call using the prepared common_params, but force n=1
                response = self.client.chat.completions.create(**common_params, n=1)

                last_response = response
                if response.choices:
                    choice = response.choices[0]
                    choice.index = i
                    aggregated_choices.append(choice)
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens

            if not last_response:
                return ChatCompletion(id='', choices=[], created=0, model=model_name, object='chat.completion', usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0))

            aggregated_usage = CompletionUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens
            )

            return ChatCompletion(
                id=last_response.id,
                model=last_response.model,
                object='chat.completion',
                created=last_response.created,
                choices=aggregated_choices,
                usage=aggregated_usage,
                system_fingerprint=last_response.system_fingerprint,
                service_tier=last_response.service_tier
            )

        else:
            # For single calls (n=1) or gpt-4.1-mini, make a direct call
            return self.client.chat.completions.create(**common_params, n=n)