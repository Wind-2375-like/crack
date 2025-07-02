from . import nethook # Your provided nethook.py
import stanza
from typing import List
from utils.generator import ChatResponseGenerator

def get_repr(model, tok, text: str, layer_name: str, token_place: str = "last"):
    """
    Computes the representation of a text prompt at a specific layer.
    """
    if token_place != "last":
        raise NotImplementedError("Only 'last' token representation is supported.")

    # Tokenize the input text
    if hasattr(tok, 'apply_chat_template'):
        # For newer tokenizers (Qwen, some Llama)
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
    else: # Fallback for older pipeline-based tokenizers
        inp = tok(text, return_tensors="pt").to(model.device)

    # Hook the desired layer
    with nethook.Trace(model, layer_name, retain_input=True) as tr:
        model(**inp)
    
    mlp_input = tr.input[0]
    
    # Return the representation at the last token position
    return mlp_input[0, -1]

def extract_subject(question: str, model: ChatResponseGenerator) -> List[str]:
    """
    A function to extract the subject from a question.
    This is crucial for the KL-divergence loss term to prevent "essence drift".
    """
    input = f"Sentence:\n{question}\nPlease extract main concepts from the sentence only, return them separated by commas."
    response = model.generate_response(
        input,
        max_tokens=100,
        temperature=0,
    )[0].split(",")
    return [concept.strip() for concept in response if concept.strip()]