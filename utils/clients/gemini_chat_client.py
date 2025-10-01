from openai import OpenAI

class GeminiChatClient:
    """A wrapper for the Gemini API using the OpenAI compatibility layer."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
    def _process_message(self, messages):
        processed_messages = []
        for role, content in messages:
            processed_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        return processed_messages

    def create(self, messages: list, model_name: str, **kwargs):
        """
        Creates a chat completion using the Gemini API.
        This method adapts the 'model_name' parameter to 'model'.
        """
        # The Gemini API expects 'model', not 'model_name'.
        # We handle the mapping here.
        api_kwargs = kwargs.copy()
        api_kwargs['model'] = model_name
        messages = self._process_message(messages)
        
        # We remove 'model_name' as it's not a valid parameter for the underlying API call.
        if 'model_name' in api_kwargs:
            del api_kwargs['model_name']

        return self.client.chat.completions.create(
            messages=messages,
            **api_kwargs
        )