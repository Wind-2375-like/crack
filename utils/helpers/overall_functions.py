def translate_model_name(model_name: str) -> str:
    """
    Translate a model name to a more user-friendly format.
    
    Args:
        model_name (str): The original model name.
    
    Returns:
        str: The translated model name.
    """
    # Define a mapping of model names to their user-friendly versions
    model_name_mapping = {
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-1b-turbo": "meta-llama/Llama-3.2-1B-Instruct-Turbo",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.2-3b-turbo": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-3.2-90b-turbo": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "qwen-2.5-1.5b": 'Qwen/Qwen2.5-1.5B-Instruct',
        "qwen-2.5-3b": 'Qwen/Qwen2.5-3B-Instruct',
        "qwen-2.5-7b": 'Qwen/Qwen2.5-7B-Instruct',
        "qwen-2.5-14b": 'Qwen/Qwen2.5-14B-Instruct',
        "olmo-2-1b": "allenai/OLMo-2-0425-1B-Instruct",
        "olmo-2-7b": "allenai/OLMo-2-1124-7B-Instruct",
    }
    
    # Return the translated model name or the original if not found
    return model_name_mapping.get(model_name, model_name)