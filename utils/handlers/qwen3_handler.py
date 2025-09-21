# utils/handlers/qwen3_handler.py

from .base_model_handler import BaseModelHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

QWEN3_END_THINK_TOKEN_ID = 151668

class Qwen3Pipeline:
    def __init__(self, model_name, enable_thinking=False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.device = self.model.device
        # Store the thinking mode state
        self.enable_thinking = enable_thinking

    def _parse_thinking_content(self, generated_ids, input_ids):
        """
        Parses the generated output to separate the thinking content from the final response.
        This is only necessary when enable_thinking=True.
        """
        # Remove the prompt from the generated text
        output_ids = generated_ids[len(input_ids):]
        
        # If not in thinking mode, no parsing is needed.
        if not self.enable_thinking:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Try to find the </think> token to separate thought from the final answer
        try:
            # Find the index of the </think> token from the end
            end_think_index = len(output_ids) - output_ids[::-1].index(QWEN3_END_THINK_TOKEN_ID)
            # The final content is everything after the </think> token
            final_content_ids = output_ids[end_think_index:]
        except ValueError:
            # If the </think> token is not found, assume the whole output is the final answer
            final_content_ids = output_ids
            
        return self.tokenizer.decode(final_content_ids, skip_special_tokens=True).strip()

    def __call__(self, formatted_input, **kwargs):
        # Apply the chat template with the correct thinking mode
        input_applied_with_chat_template = self.tokenizer.apply_chat_template(
            formatted_input,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        model_inputs = self.tokenizer([input_applied_with_chat_template], return_tensors="pt").to(self.device)
        
        # Set recommended sampling parameters based on thinking mode, as per Qwen3 docs
        if self.enable_thinking:
            temperature = kwargs.get("temperature", 0.6)
            top_p = kwargs.get("top_p", 0.95)
        else:
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.8)

        n = kwargs.get("num_return_sequences", 1)
        max_tokens = kwargs.get("max_tokens", 512)
        
        with torch.no_grad():
            responses = []
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
            )
            
            # Decode and parse each response in the batch
            for i in range(n):
                response_text = self._parse_thinking_content(generated_ids[i].tolist(), model_inputs.input_ids[0].tolist())
                responses.append(response_text)

            torch.cuda.empty_cache()
            return responses

class Qwen3Handler(BaseModelHandler):
    def initialize_model(self):
        enable_thinking = 'Thinking' in self.model_name
        self.pipeline = Qwen3Pipeline(self.model_name.replace('-Thinking', ''), enable_thinking=enable_thinking)

    def format_messages(self, messages):
        processed_messages = []
        for role, content in messages:
            processed_messages.append({"role": role, "content": content})
        return processed_messages

    def generate_response(self, formatted_input, **kwargs):
        outputs = self.pipeline(
            formatted_input,
            **kwargs
        )
        torch.cuda.empty_cache()
        return outputs