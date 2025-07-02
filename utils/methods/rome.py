from .root_method import RootExperimentMethod
from copy import deepcopy
import torch
from utils.methods.method_utils.rome import nethook
import pickle
from utils.dataset.reasoning_dataset import ReasoningEvalDataset
from .method_utils.rome.rome_hparams import get_hparams_for_model
from .method_utils.rome.rome_functions import compute_u, compute_v
from scripts.experiment.knowledge_injection import extract_all_required_knowledge

class Method(RootExperimentMethod):
    """
    Implements the 'base' experiment: no knowledge injection.
    """
    def __init__(self, args, chat_response_generator):
        super().__init__(args, chat_response_generator)
        self.hparams = get_hparams_for_model(self.args.model_name)
        
    def prepare_input(self, item):
        """
        Prepares the input for the model.
        Args:
            item (dict): A dictionary containing:
                - "id": index of the question
                - "question": the question text
                - "answer": the answer text
                - "required_knowledge": the list of required knowledge
            knowledge_to_inject_str (str, optional): Pre-formatted string of knowledge to inject. Defaults to "".
        Returns:
            tuple: A tuple containing:
                - prepared_user_prompt (str): The user prompt for the model.
                - prepared_system_prompt (str): The system prompt for the model.
        """
        
        if self.args.task_name == "grow":
            prepared_system_prompt = (
                "You are given a question. To answer the question, you should think step by step. "
                "Use line breaks between steps, but do not use line breaks within each step. "
                "The final answer to the question should start with "
                "\"The answer is ...\", and should be placed at the final step. "
                "Please make an educated guess and always return an entity.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nWhat is the capital of the country where Plainfield Town Hall was created?\n\n"
                "Assistant:\n"
                "1. Given my knowledge, Plainfield Town Hall was created in the United States of America.\n"
                "2. Given my knowledge, the capital of United States is Washington, D.C.\n"
                "3. Therefore, the capital of the country where Plainfield Town Hall was created is Washington, D.C.\n"
                "4. The answer is Washington, D.C.\n\n"
            )
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "code":
            prompt_solution = """```python
import pandas as pd

def task_func(dealer_sales_data):
    # Step 1: Create DataFrame & Step 2: Handle Empty Input (if dealer_sales_data is empty)
    df = pd.DataFrame(dealer_sales_data)
    
    if not dealer_sales_data:
        return []

    # Ensure 'id' and 'num_sold' columns exist, otherwise it's malformed input
    if 'id' not in df.columns or 'num_sold' not in df.columns:
        return []

    # Step 3: Find Max Sales
    max_sold = df['num_sold'].max()

    # Step 4: Identify Top Sellers
    top_selling_cars = df[df['num_sold'] == max_sold]

    # Step 5: Extract and Sort IDs
    top_selling_ids = top_selling_cars['id'].tolist()
    sorted_ids = sorted(top_selling_ids)

    # Step 6: Return Result
    return sorted_ids
```"""
        
            prepared_system_prompt = (
                "You are given a coding question. To answer the question, you should think step by step. "
                "You should also write self-contained code starting with\n"
                "```python\n"
                "import ...\n"
                "def task_func(...):\n"
                "\t...\n"
                "```\n"
                "[Here is one demonstration]\n\n"
                "User:\nCompute and return the IDs of the best-selling cars from `dealer_sales_data`, a list of dictionaries, with each composed of an 'id' key (a string identifier) and a 'num_sold' key (an integer). The function should output with:\nids: A list of string ids for the car(s) that sell the best. If multiple car ids achieve the maximum sales, all are returned. The list should be sorted alphabetically for consistent output.\nYou should write self-contained code starting with:\n```python\nimport pandas as pd\ndef task_func(dealer_sales_data):\n```\n\n"
                "Assistant:\n"
                "Okay, I'll outline the steps to solve this problem and then provide the Python code using pandas.\n"
                "Here's the plan:\n"
                "1.  **Create DataFrame**: Convert the list of car records into a pandas DataFrame as we are required to use the pandas library.\n"
                "2.  **Handle Empty Input**: If the initial data is empty (resulting in an empty DataFrame), there's no data to process, so we should return an empty list early.\n"
                "3.  **Find Max Sales**: Find the maximum value in their 'num_sold' column. This gives us the sales figure of the best-selling car(s).\n"
                "4.  **Identify Top Sellers**: Filter the DataFrame again to get only those cars whose 'num_sold' is equal to the maximum sales figure found in the previous step.\n"
                "5.  **Extract and Sort IDs**: From these top-selling cars, extract their 'id' values into a list. Then, sort this list of IDs alphabetically.\n"
                "6.  **Return Result**: The sorted list of IDs is the final answer.\n\n"
                "Now, let's implement this solution.\n\n"
            ) + prompt_solution
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "math":
            prepared_system_prompt = (
                "You are given a question. To answer the question, you should think step by step as detailed as possible. "
                "Use line breaks between steps, but do not use line breaks within each step. "
                "The final answer to the question should start with "
                "\"The answer is ...\", and should be placed at the final step. "
                "Please make an educated guess and always return an answer.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nThree pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?\n\n"
                "Assistant:\n"
                "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.\n"
                "2. We have $3p+e=1.24$ and $5p+e=1.82$.\n"
                "3. To solve this system, let's subtract the first equation from the second equation. This will eliminate e.\n"
                "4. $5p+e-3p-e=1.82-1.24$.\n"
                "5. This simplifies to $2p=0.58$. So $p=0.29$.\n"
                "6. That means a pencil costs 29 cents.\n"
                "7. The answer is 29 cents."
            )
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        else:
            raise NotImplementedError(f"Task {self.args.task_name} is not implemented.")

        return prepared_user_prompt, prepared_system_prompt
    
    def apply_rome_to_model(self, knowledge_to_inject=[], all_knowledge_list=None, cache_key=None):
        """
        Update the model (self.chat_response_generator) with injected knowledge
        Args:
            knowledge_to_inject (list, optional): Pre-formatted list of knowledge to inject. Defaults to [].
        """
        model = self.chat_response_generator.client.handler.pipeline.model
        tok = self.chat_response_generator.client.handler.pipeline.tokenizer
        tok.pad_token = tok.eos_token
        
        for _, request in enumerate(knowledge_to_inject):
            deltas = self.execute_rome(model, tok, request, self.hparams, all_knowledge_list, cache_key)

            with torch.no_grad():
                for w_name, (delta_u, delta_v) in deltas.items():
                    upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = self.upd_matrix_match_shape(upd_matrix, w.shape)

                    w[...] += upd_matrix
                    
    def execute_rome(self, model, tok, request_item, hparams, all_knowledge_list, cache_key):
        """Execute ROME algorithm
        """
        # Update target and print info
        request = {
            "prompt": "{}",
            "subject": f"Please answer the following question in one sentence in a new line:\n{request_item["probe_question"].strip()}\n",
            "target_new": {"str": request_item["probe_answer"]},
        }

        # Update loop: sequentially intervene at each specified layer
        deltas = {}
        for layer in sorted(hparams.layers):
            # Compute rank-1 update matrix
            left_vector: torch.Tensor = compute_u(
                model,
                tok,
                request,
                hparams,
                layer,
                all_knowledge_list,
                cache_key
            )
            right_vector: torch.Tensor = compute_v(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector
            )

            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

        return deltas
    
    def upd_matrix_match_shape(self, matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """
        GPT-2 and GPT-J have transposed weight representations.
        Returns a matrix that matches the desired shape, else raises a ValueError
        """

        if matrix.shape == shape:
            return matrix
        elif matrix.T.shape == shape:
            return matrix.T
        else:
            raise ValueError(
                "Update matrix computed by ROME does not match original weight shape. "
                "Check for bugs in the code?"
            )
    
    def run(self, item, knowledge_to_inject=[]):
        """
        Probes a chain of triples using the specified model.
        Args:
            item (dict): A dictionary containing question and other details.
            knowledge_to_inject (list, optional): Pre-formatted list of knowledge to inject. Defaults to [].
        Returns:
            item (dict): The updated item with model_response.
            usage (dict): A dictionary containing the token usage information for the model.
        """
        # Preserve the original model
        original_model_state_dict = deepcopy(self.chat_response_generator.client.handler.pipeline.model.state_dict())
        
        # Use ROME to inject knowledge
        if self.args.inject_knowledge and knowledge_to_inject != []:
            eval_dataset = list(ReasoningEvalDataset(
                raw_path=f'data/{self.args.task_name}/test_{self.args.data_size}_depth_{self.args.depth}.pkl',
                probe_path=f'data/eval_results/{self.args.task_name}/probe_evaluated/test_{self.args.data_size}_depth_{self.args.depth}_{self.args.model_name}.pkl',
            ))
            
            with open(f'data/{self.args.task_name}/test_{self.args.data_size}_depth_{self.args.depth}.pkl', "rb") as f:
                raw_dataset = pickle.load(f)
                
            # Extract all required knowledge from the dataset
            all_knowledge_list = extract_all_required_knowledge(eval_dataset, raw_dataset)
            cache_key = f"{self.args.model_name.replace('/', '_')}_{self.args.task_name}_{self.args.data_size}_{self.args.depth}"
            self.apply_rome_to_model(knowledge_to_inject, all_knowledge_list, cache_key)
            
        prepared_user_prompt, prepared_system_prompt = self.prepare_input(item)
        
        self.chat_response_generator.update_chat_history([
            ("system", prepared_system_prompt),
        ])
        
        model_response = self.chat_response_generator.generate_response(
            prepared_user_prompt,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.num_responses,
            max_tokens=self.args.max_tokens,
        )[0].replace("Assistant:", "").strip()
            
        item["model_response"] = model_response
        
        # Restore the original model state but keep the usage statistics
        usage = self.chat_response_generator.get_usage()
        self.chat_response_generator.client.handler.pipeline.model.load_state_dict(original_model_state_dict)
        self.chat_response_generator._usage = usage

        return item, self.chat_response_generator.get_usage()