from .root_method import RootExperimentMethod
from copy import deepcopy
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from rich.console import Console

console = Console()


class KnowledgeDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data
        self.processed_data = self._process_data()

    def _process_data(self):
        processed_entries = []
        for knowledge in self.data:
            # Define the prompt structure
            messages = [
                {"role": "system", "content": "Answer the question with the name of an entity. Provide only the name of the entity as your answer."},
                {"role": "user", "content": knowledge['probe_question']}
            ]
            
            # Create the prompt part and the full text (prompt + answer)
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt_text + knowledge['probe_answer'] + self.tokenizer.eos_token

            # Tokenize both to find the length of the prompt
            prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt")
            full_tokens = self.tokenizer(full_text, return_tensors="pt")

            # Create labels: mask prompt part by setting its labels to -100
            labels = full_tokens['input_ids'].clone()
            prompt_len = prompt_tokens['input_ids'].shape[1]
            labels[0, :prompt_len] = -100
            
            processed_entries.append({
                'input_ids': full_tokens['input_ids'].squeeze(0),
                'attention_mask': full_tokens['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0)
            })
        return processed_entries

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
    

class Method(RootExperimentMethod):
    """
    Implements the 'base' experiment: no knowledge injection.
    """
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
    
    def prepare_probe_input(self, item):
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
            prepared_system_prompt = "Answer the question with the name of an entity. Provide only the name of the entity as your answer. Please make an educated guess and always return an entity.\n\n[Here is one demonstration]\n\nUser:\nWho is the developer of Telegram?\n\nAssistant:\nTelegram FZ-LLC"
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "code":
            prepared_system_prompt = "Answer the question with a Python code snippet, which requires ONLY ONE direct function or class constructor call from ONLY ONE library.\nProvide ONLY ONE function or constructor call itself with correct positional arguments.\n- Do NOT include import statements.\n- Do NOT include example data, variable assignments, or any other code.\n- For each keyword argument of the function, if the question implies specific keyword arguments, include them in the function call. If the question does not require the keyword argument explicitly or only require it with its default value, the function can be called without this keyword argument.\n- Please make an educated guess and always return a function call.\n\n[Here is one demonstration]\n\nUser:\nGiven the library pandas, how can we create a DataFrame by explicitly passing the input data (such as an ndarray, Iterable, dict, or DataFrame) using the `data` parameter?\n\nAssistant:\n```python\npandas.DataFrame(data)\n```"
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "math":
            prepared_system_prompt = "Answer the math question with a concise sentence. Provide only the direct answer to the math question and no more additional reasoning.\n\n[Here is one demonstration]\n\nUser:\nGiven the equations $3p+e=1.24$ and $5p+e=1.82$, what specific operation will eliminate the variable 'e'?\n\nAssistant:\nSubtracting the first equation from the second will eliminate the variable 'e'."
            prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        else:
            raise NotImplementedError(f"Task {self.args.task_name} is not implemented.")

        return prepared_user_prompt, prepared_system_prompt
    
    def apply_ftck_to_model(self, knowledge_to_inject):
        """
        Applies Fine-tuning to the model using the provided knowledge.
        """
        base_model = self.chat_response_generator.client.handler.pipeline.model
        tokenizer = self.chat_response_generator.client.handler.pipeline.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset = KnowledgeDataset(tokenizer, knowledge_to_inject)

        # Define different target modules for different model families
        model_name_lower = self.args.model_name.lower()
        if "qwen" in model_name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "llama" in model_name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else: # Default for other models
            target_modules = ["q_proj", "k_proj", "v_proj"]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules, # Use the adaptive list
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM
        )

        base_model.enable_input_require_grads()
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.config.use_cache = False
        
        training_args = TrainingArguments(
            output_dir="logs/ft_ck_checkpoints",
            per_device_train_batch_size=1,
            num_train_epochs=4,
            learning_rate=2e-4,         
            bf16=True,          
            save_strategy="no",
            gradient_checkpointing=True,
            logging_steps=10, # Log less frequently to reduce clutter
            label_names=["labels"],
            report_to="none",
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
        )

        train_result = trainer.train()
        console.print(f"[bold green]✅ FT-CK training complete. Final Loss: {train_result.training_loss:.4f}[/bold green]")
        peft_model.eval()
        
        return peft_model
    
    def edit(self, knowledge_to_inject):
        self.original_model = self.chat_response_generator.client.handler.pipeline.model
        
        if knowledge_to_inject:
            trained_peft_model = self.apply_ftck_to_model(knowledge_to_inject)
            self.chat_response_generator.client.handler.pipeline.model = trained_peft_model

    def restore(self):
        usage = self.chat_response_generator.get_usage()
        if hasattr(self, 'original_model') and self.original_model is not None:
            self.chat_response_generator.client.handler.pipeline.model = self.original_model
            self.original_model = None # Clear the saved object
            torch.cuda.empty_cache()

        self.chat_response_generator._usage = {} # Reset usage for next batch
        return usage
    
    def run(self, item, knowledge_to_inject=[], probe=False):
        """
        Probes a chain of triples using the specified model.
        Args:
            item (dict): A dictionary containing question and other details.
            knowledge_to_inject (list, optional): Pre-formatted list of knowledge to inject. Defaults to [].
        Returns:
            item (dict): The updated item with model_response.
            usage (dict): A dictionary containing the token usage information for the model.
        """
        if not probe:
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
        else:
            prepared_user_prompt, prepared_system_prompt = self.prepare_probe_input(item)
            
            self.chat_response_generator.update_chat_history([
                ("system", prepared_system_prompt),
            ])
            
            responses = self.chat_response_generator.generate_response(
                prepared_user_prompt,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=self.args.num_responses,
                max_tokens=self.args.max_tokens,
            )
                
            item["probe_answers"] = [r.replace("Assistant:", "").strip() for r in responses]
            
        return item, self.chat_response_generator.get_usage()