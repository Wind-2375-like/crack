from pathlib import Path

import torch
from tqdm.auto import tqdm
import numpy as np

from .globals import *
from .nethook import Trace, set_requires_grad
from .runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    all_knowledge_list,
    cache_key,
    to_collect,
    precision=None,
    batch_tokens=None,
):
    """
    Function to load or compute cached stats.
    """
    
    # Ensure the model is in evaluation mode and gradients are not required.
    set_requires_grad(False, model)
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    
    # Build the filename based on the provided cache_key
    stats_dir = Path(stats_dir)
    filename = stats_dir / f"{cache_key}_{layer_name}.npz"
    
    # If the file exists, load it.
    if filename.exists():
        stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
        loaded_data = dict(np.load(filename))
        stat.load_state_dict(loaded_data)
        return stat

    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple dataset from your all_knowledge_list
    # The original code's TokenizedDataset expects a 'text' field.
    prompts_as_dicts = [{"text": f"Please answer the following question in one sentence in a new line:\n{item['probe_question'].strip()}\n"} for item in all_knowledge_list]
    
    if hasattr(model.config, "max_position_embeddings"):
        maxlen = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        maxlen = model.config.n_positions
    else:
        raise ValueError("Model configuration does not specify max position embeddings.")
        
    if batch_tokens is None:
        batch_tokens = maxlen * 3
        
    ds = TokenizedDataset(prompts_as_dicts, tokenizer, maxlen=maxlen)

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=None, # Caching is handled manually above
        sample_size=len(ds),
        batch_size=1, # Process one group of same-length seqs at a time
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
             
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
                
    # Save the computed stats to the file
    np.savez(filename, **stat.state_dict())
    
    return stat