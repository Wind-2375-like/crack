import numpy as np
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import nethook, repr_tools
from .rome_hparams import ROMEHyperparameters
from .layer_stats import layer_stats

# Cache variables
inv_mom2_cache = {}

KEYWORDS = [
    "human", "English", "Wikipedia category", "male", "female", "taxon", "country", "species", "title", "metre",
    "numbers and counting", "addition and subtraction", "multiplication and division", "fractions and decimals", "algebra", "geometry", "statistics and probability", "ratios and proportions", "logic and sets", "calculus",
    "variables and data types", "control flow", "functions", "lists and dictionaries", "modules and packages", "object-oriented programming", "exception handling", "file I/O", "list comprehensions", "string formatting"
]

def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    all_knowledge: List[Dict],
    cache_key: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    stat = layer_stats(
        model,
        tok,
        layer_name,
        "data/rome_stats",
        all_knowledge,
        cache_key,
        to_collect=["mom2"],
    )
    return torch.inverse(
        stat.mom2.moment().to("cuda")
    ).float()  # Cast back to float32

def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperparameters,
    layer: int,
    all_knowledge: List[Dict],
    cache_key: str,
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """
    
    # The prompt template is the one with the placeholder.
    prompt_template = request["prompt"]
    # The word to be inserted into the template is the subject.
    subject_word = request["subject"]

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )

    ## Pass the template and the word separately to the helper function.
    # Your modified repr_tools will handle the chat templating internally.
    cur_repr = repr_tools.get_reprs_at_word_tokens(
        context_templates=[prompt_template],
        words=[subject_word],
        subtoken=hparams.fact_token.split("_")[-1],  # "last" from "subject_last"
        **word_repr_args,
    ).mean(0)
    
    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        inv_cov = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            all_knowledge,
            cache_key,
        )
        u = (inv_cov.float() @ u.float().unsqueeze(1)).squeeze()

    final_u = u / u.norm()
    
    return final_u.to(model.dtype)


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperparameters,
    layer: int,
    left_vector: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # 1. Manually create the target string with the end-of-text token.
    target_text = request["target_new"]["str"] + tok.eos_token

    # 2. Tokenize it, but prevent the tokenizer from adding its own special tokens.
    target_ids = tok(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    ).to("cuda")["input_ids"][0]
    
    context_templates = ["{}"]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        tok.apply_chat_template(
            [
                {"role": "user", "content": context.format(request["prompt"])},
                {"role": "assistant", "content": request["target_new"]["str"]},
            ],
            tokenize=False,
            add_generation_prompt=False
        )
        for context in context_templates
    ], [
        tok.apply_chat_template(
            [
                {"role": "user", "content": "Tell me something about {}:"},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for _ in range(len(KEYWORDS))
    ]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.replace("{}", request["subject"]) for prompt in rewriting_prompts]+[kl_prompt.format(KEYWORDS[i]) for i, kl_prompt in enumerate(kl_prompts)],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
    
        # The logits that predict the target tokens start one position
        # before the tokens themselves.
        start_idx = ex_len - len(target_ids) - 1
        end_idx = ex_len - 1
        
        # Place the target token IDs at these preceding positions.
        rewriting_targets[i, start_idx:end_idx] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"] if i < len(rewriting_prompts) else KEYWORDS[i-len(rewriting_prompts)], tok, hparams.fact_token, verbose=False
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    
    if hasattr(model.config, "embd"):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda", dtype=model.dtype)
    elif hasattr(model.config, "hidden_size"):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda", dtype=model.dtype)
    else:
        raise ValueError("Could not find model's embedding size.")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                # print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        # print(
        #     f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
        #     f"avg prob of [{request['target_new']['str']}] "
        #     f"{torch.exp(-nll_loss_each).mean().item()}"
        # )
        if nll_loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    
    prompt_template = tok.apply_chat_template(
            [{"role": "user", "content": "{}"}],
            tokenize=False,
            add_generation_prompt=True
    )
    
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=prompt_template.format(request["prompt"]),
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str, # This argument is now less important but kept for compatibility
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    This is a new, robust version for chat templates.
    """

    # 1. Break the prompt template into a prefix and a suffix
    # The prompt is the full chat template with a {} for the subject
    # e.g., "<|...|>user<|...|>\n\nWhich company was {} the founder of?<|...|>"
    try:
        prefix, suffix = prompt.split("{}")
    except ValueError:
        raise ValueError("The prompt for find_fact_lookup_idx must contain one and only one '{}' placeholder.")

    # 2. Tokenize the pieces separately
    # We do NOT add special tokens here because the template already has them.
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    subject_ids = tok(subject, add_special_tokens=False)["input_ids"]

    # 3. The lookup index is simply the length of the prefix plus the length of the subject - 1.
    # This gives us the index of the LAST token of the subject.
    ret = len(prefix_ids) + len(subject_ids) - 1
    
    # This check is just for your peace of mind, to see the token we're targeting.
    sentence = prefix + subject + suffix
    if verbose:
        print(
            f"Lookup index found: {ret} | Token being targeted:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret