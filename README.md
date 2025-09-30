# Complex Reasoning Amid Conflicting Knowledge (CRACK)

## Setup

We use Python 3.12.

```
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
!pip install tqdm openai together transformers sparqlwrapper datasets accelerate ninja pynvml
!MAX_JOBS=32 pip install flash-attn --no-build-isolation
```

In `api_key/config.json`, you need to provide your API keys for the models you want to use. The file should look like this:

```json
{
    "api_key": {
        "openai_api_key": "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "togetherai_api_key": "123abc456def789ghijklmno0123456789",
        "huggingface_api_key": "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    },
    "wikimedia": {
        "client_application_key": "123abc456def789ghijklmno0123456789",
        "client_application_secret": "123abc456def789ghijklmno0123456789",
        "access_token": "123abc456def789ghijklmno0123456789",
        "user_agent": "CRACK/1.0 (https://www.wikidata.org/wiki/User:CRACK)"
    }
}
```

## Data Collection

```python
python scripts/testset/grow_collection.py \
    --data_size 100 \                           # Number of examples to collect
    --depth 4 \                                 # Length of the reasoning chain
    --api_config_file ./api_key/config.json \   # Path to the API configuration file
```

## Experiments

### Subtask 1: Knowledge Probing

To probe the model's knowledge, run:
```python
python scripts/experiment/knowledge_probe.py \
    --data_size 100 \                           # Number of collected examples (different from the number of examples to probe)
    --api_config_file ./api_key/config.json \   # Path to the API configuration file
    --model_name llama-3.2-3b \                 # Tested model
    --task_name grow \                          # Can be grow (Graph Reasoning On Wikidata) or cue (Code-generation Using External APIs)
    --temperature 0.7 \
    --top_p 0.7 \
    --max_tokens 100 \
    --num_responses 10
```

To automatically annotate the probing results, you can use the following command:

```python
python scripts/evaluation/knowledge_evaluation.py \
    --data_size 100 \                           # Number of collected examples (different from the number of examples to evaluate)
    --api_config_file ./api_key/config.json \   # Path to the API configuration file
    --model_name llama-3.2-3b \                 # Tested model
    --evaluate_model_name gpt-4.1-mini \         # Model used for evaluation
    --task_name grow \                          # Can be grow (Graph Reasoning On Wikidata) or cue (Code-generation Using External APIs)
```

### Subtask 2: Knowledge Injection

Before running the script, ensure you have evaluated the knowledge probing results to identify the knowledge gaps in the model's responses.

```python
python scripts/experiment/knowledge_injection.py \
    --data_size 100 \                           # Number of examples to inject knowledge
    --api_config_file ./api_key/config.json \   # Path to the API configuration file
    --model_name llama-3.2-3b \                 # Tested model
    --task_name grow \                          # Can be grow (Graph Reasoning On Wikidata) or cue (Code-generation Using External APIs)
    --temperature 0.7 \
    --top_p 0.7 \
    --max_tokens 512 \
    --inject_knowledge \                        # Whether to inject knowledge
    --knowledge_aggregation_method 1 \          # Scope for aggregating 'unknown' knowledge. Must be >= 1. 1: item-specific. N (e.g., 10, 100): group of N items.
    --method base \                             # Method for knowledge injection. base: simple few-shot CoT prompting.
```

To automatically annotate the knowledge injection results, you can use the following command:

```python
python scripts/evaluation/reasoning_evaluation.py \
    --data_size 100 \                           # Number of examples to evaluate
    --api_config_file ./api_key/config.json \   # Path to the API configuration file
    --model_name llama-3.2-3b \                 # Tested model
    --evaluate_model_name gpt-4.1-mini \         # Model used for evaluation
    --task_name grow \                          # Can be grow (Graph Reasoning On Wikidata) or cue (Code-generation Using External APIs)
    --inject_knowledge \                        # Whether to inject knowledge
    --knowledge_aggregation_method 1 \          # Scope for aggregating 'unknown' knowledge. Must be >= 1. 1: item-specific. N (e.g., 10, 100): group of N items.
    --method base \                             # Method for knowledge injection. base: simple few-shot CoT prompting.
```
