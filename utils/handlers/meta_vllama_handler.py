from utils.handlers.base_model_handler import BaseModelHandler
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MetaVLlamaPipeline:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoProcessor.from_pretrained(model_name)

    def __call__(self, formatted_input, **kwargs):
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        n = kwargs.get("num_return_sequences", 1)
        max_tokens = kwargs.get("max_tokens", 512)
        do_sample = kwargs.get("do_sample", True)
        input = self.tokenizer.apply_chat_template(
            formatted_input,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")
        outputs = self.model.generate(
            **input,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True).split("assistant\n\n")[-1].strip() for output in outputs]
            

class MetaVLlamaHandler(BaseModelHandler):
    def initialize_model(self):
        self.pipeline = MetaVLlamaPipeline(self.model_name)

    def format_messages(self, messages):
        processed_messages = []
        for role, content_string in messages:
            formatted_content = [
                {"type": "text", "text": content_string}
            ]
            processed_messages.append({"role": role, "content": formatted_content})
        return processed_messages

    def generate_response(self, formatted_input, **kwargs):
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", 512)

        # outputs = []
        # for _ in range(n):
        #     outputs.append(self.pipeline(
        #         formatted_input,
        #         max_new_tokens=max_tokens,
        #         eos_token_id=self.terminators,
        #         do_sample=True,
        #         temperature=temperature,
        #         top_p=top_p,
        #         num_return_sequences=1,
        #         pad_token_id=self.pipeline.tokenizer.eos_token_id,
        #     )[0])
        #     torch.cuda.empty_cache()
        
        # return [output["generated_text"][-1]["content"] for output in outputs]
        outputs = self.pipeline(
            formatted_input,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
        )
        torch.cuda.empty_cache()
        
        return outputs