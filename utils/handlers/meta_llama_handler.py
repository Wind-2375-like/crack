from utils.handlers.base_model_handler import BaseModelHandler
import transformers
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MetaLlamaHandler(BaseModelHandler):
    def initialize_model(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
            },
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def format_messages(self, messages):
        processed_messages = []
        for role, content in messages:
            processed_messages.append({"role": role, "content": content})
        return processed_messages

    def generate_response(self, formatted_input, **kwargs):
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 1)
        top_k = kwargs.get("top_k", 1)
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
        #         top_k=top_k,
        #         num_return_sequences=1,
        #         pad_token_id=self.pipeline.tokenizer.eos_token_id,
        #     )[0])
        #     torch.cuda.empty_cache()
        
        # return [output["generated_text"][-1]["content"] for output in outputs]
        outputs = self.pipeline(
            formatted_input,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=n,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        torch.cuda.empty_cache()
        
        return [outputs["generated_text"][-1]["content"] for outputs in outputs]