from utils.handlers.base_model_handler import BaseModelHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class OlmoHandler(BaseModelHandler):
    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def format_messages(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate_response(self, formatted_input, **kwargs):
        model_inputs = self.tokenizer([formatted_input], return_tensors="pt").to(self.device)
        
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", 512)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
            )
            # Remove the prompt from the generated text
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            torch.cuda.empty_cache()
            return response