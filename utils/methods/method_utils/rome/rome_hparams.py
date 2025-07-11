from dataclasses import dataclass

@dataclass
class ROMEHyperparameters:
    # The layer to probe for activations
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str

    # Layers to edit. We will edit a range of layers.
    layers: list[int]

    # Algorithm
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_dtype: str

def get_hparams_for_model(model_name: str):
    """
    Returns a ROMEHyperparameters object for the given model name.
    """
    
    if model_name == "llama-3.2-1b":
       return ROMEHyperparameters(
            rewrite_module_tmp="model.layers.{}.mlp.down_proj",
            layer_module_tmp="model.layers.{}",
            mlp_module_tmp="model.layers.{}.mlp",
            layers=[3],
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=0.5,
            v_loss_layer=11, # Last layer of the model
            v_weight_decay=0.5,
            kl_factor=0.625,
            mom2_adjustment=True,
            mom2_dtype="bfloat16",
            clamp_norm_factor=4,
        )
    elif model_name == "llama-3.2-3b":
        return ROMEHyperparameters(
            rewrite_module_tmp="model.layers.{}.mlp.down_proj",
            layer_module_tmp="model.layers.{}",
            mlp_module_tmp="model.layers.{}.mlp",
            layers=[9], # Corrected from 10 to 9 for 0-based index
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=0.5,
            v_loss_layer=27, # Last layer of the model
            v_weight_decay=0.5,
            kl_factor=0.625,
            mom2_adjustment=True,
            mom2_dtype="bfloat16",
            clamp_norm_factor=4,
        )
    elif model_name == "qwen-2.5-1.5b":
        return ROMEHyperparameters(
            rewrite_module_tmp="model.layers.{}.mlp.down_proj",
            layer_module_tmp="model.layers.{}",
            mlp_module_tmp="model.layers.{}.mlp",
            layers=[5], # Corrected from 6 to 5 for 0-based index
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=0.5,
            v_loss_layer=27, # Last layer of the model
            v_weight_decay=0.5,
            kl_factor=0.625,
            mom2_adjustment=True,
            mom2_dtype="bfloat16",
            clamp_norm_factor=4,
        )
    elif model_name == "qwen-2.5-3b":
        return ROMEHyperparameters(
            rewrite_module_tmp="model.layers.{}.mlp.down_proj",
            layer_module_tmp="model.layers.{}",
            mlp_module_tmp="model.layers.{}.mlp",
            layers=[10], # Corrected from 11 to 10 for 0-based index
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=0.5,
            v_loss_layer=35, # Last layer of the model
            v_weight_decay=0.5,
            kl_factor=0.625,
            mom2_adjustment=True,
            mom2_dtype="bfloat16",
            clamp_norm_factor=4,
        )
    else:
        raise NotImplementedError(f"No ROME hparams for model {model_name}")