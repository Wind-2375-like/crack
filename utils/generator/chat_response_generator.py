from utils.clients.openai_chat_client import OpenAIChatClient
from utils.clients.together_chat_client import TogetherChatClient
from utils.clients.local_chat_client import LocalChatClient
from together.error import RateLimitError as TogetherRateLimitError
from openai import RateLimitError as OpenAIRateLimitError
from together.error import APIConnectionError as TogetherAPIConnectionError
from together.error import ServiceUnavailableError as TogetherServiceUnavailableError
from together.error import APIError as TogetherAPIError
from openai import APIConnectionError as OpenAIAPIConnectionError
from time import sleep
from huggingface_hub import login

class ChatResponseGenerator:
    """ A General Chat Response Generator that can be used to generate responses for a chat system. """
    def __init__(self, model_name, chat_history=[], api_key=None, local=False):
        self.model_name = model_name
        self.update_chat_history(chat_history)
        self.model_name_to_openai_or_together = {
            "gpt-4.1-mini": "openai",
            "o4-mini": "openai",
            "gpt-5-mini-2025-08-07": "openai",
            "meta-llama/Llama-3.2-3B-Instruct-Turbo": "together",
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": "together",
            "meta-llama/Meta-Llama-3-8B-Instruct-Turbo": "together",
            "Qwen/Qwen2-72B-Instruct": "together",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "together",
            "mistralai/Mixtral-8x22B-Instruct-v0.1": "together",
            "mistralai/Mixtral-8x7B-Instruct-v0.1": "together",
        }
        if not (model_name in self.model_name_to_openai_or_together or local):
            if "gpt-4.1-mini" in model_name: # fine-tuned models
                self.client_type = "openai"
            else:
                # print(f"Model {model_name} is not supported in API. Trying to use local model.")
                self.client_type = "local"
        else:
            self.client_type = "local" if local else self.model_name_to_openai_or_together[model_name]

        if self.client_type == "openai":
            self.client = OpenAIChatClient(api_key=api_key.get("openai_api_key", None))
        elif self.client_type == "together":
            self.client = TogetherChatClient(api_key=api_key.get("togetherai_api_key", None))
        elif self.client_type == "local":
            # login(token=api_key.get("huggingface_api_key", None))
            self.client = LocalChatClient(model_name=model_name)
            
        self._usage = {}

    def get_usage(self):
        return self._usage
                
    def update_chat_history(self, chat_history):
        self.chat_history = chat_history
        
    def generate_response(self, query, **kwargs):
        """ Generate a list of thoughts for the given question using the specified model. 
            Either OpenAI's GPT-4o's or Together's Models' API can be used.
            Each thought in a CoT list is represented as (sentence, label) where label is one of [evidence], [reasoning], and [claim].
        """
        n = kwargs.get("n", 1)
        model_name = kwargs.get("model_name", self.model_name)
        kwargs.update({"model_name": model_name})
        sleep_seconds = 4
        max_sleep_seconds = 512

        def update_usage(model_name, usage):
            if self.client_type == "local":
                self._usage[model_name] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            elif model_name not in self._usage:
                self._usage[model_name] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            else:
                self._usage[model_name] = {
                    "prompt_tokens": self._usage[model_name]["prompt_tokens"] + usage.prompt_tokens,
                    "completion_tokens": self._usage[model_name]["completion_tokens"] + usage.completion_tokens,
                    "total_tokens": self._usage[model_name]["total_tokens"] + usage.total_tokens,
                }

        while True:
            try:
                response = self.client.create(
                    messages=self.chat_history+[("user", query)],
                    **kwargs
                )
                if self.client_type != "local":
                    usage = response.usage
                    update_usage(model_name, usage)
                else:
                    update_usage(model_name, None)
                choices = response.choices
                return [choice.message.content.strip() for choice in choices]
                    
            except (OpenAIRateLimitError, TogetherRateLimitError) as e:
                print(f"Rate Limit Error: {e}")
                print(f"Sleeping for {sleep_seconds} seconds")
                sleep_seconds = min(sleep_seconds*2, max_sleep_seconds)
                sleep(sleep_seconds)
            except (OpenAIAPIConnectionError, TogetherAPIConnectionError) as e:
                print(f"API Connection Error: {e}")
                print(f"Sleeping for {sleep_seconds} seconds")
                sleep_seconds = min(sleep_seconds*2, max_sleep_seconds)
                sleep(sleep_seconds)
            except (TogetherServiceUnavailableError) as e:
                print(f"Service Unavailable Error: {e}")
                print(f"Sleeping for {sleep_seconds} seconds")
                sleep_seconds = min(sleep_seconds*2, max_sleep_seconds)
                sleep(sleep_seconds)
            except (TogetherAPIError) as e:
                if e.code == 422:
                    print("Input is too long.")
                    return [""]*n
                else:
                    raise e