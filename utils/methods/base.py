from .root_method import RootExperimentMethod

class Method(RootExperimentMethod):
    """
    Implements the 'base' experiment: no knowledge injection.
    """
    def prepare_input(self, item, knowledge_to_inject_str=""):
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
            system_prompt_without_injection = (
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
            
            system_prompt_after_injection = (
                "You are given a question. To answer the question, you should think step by step. "
                "Use line breaks between steps, but do not use line breaks within each step. "
                "The final answer to the question should start with "
                "\"The answer is ...\", and should be placed at the final step. "
                "Please make an educated guess and always return an entity. "
                "Users may provide a set of facts or not. If they provide facts that conflict with your knowledge, you should update your "
                "knowledge and include the facts in your reasoning steps to answer the question.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nWho is the person who is the current head of government of British married to?\nPlease update your knowledge with the following facts:\n"
                "The name of the current head of the British government is Keir Starmer.\n\n"
                "Assistant:\n"
                "1. The user provided that Keir Starmer is the current head of government of the British government.\n"
                "2. I will update my knowledge with the provided fact: the current head of government of the British government is Keir Starmer.\n"
                "3. Given my knowledge, Keir Starmer is married to Victoria Starmer.\n"
                "4. Therefore, the person who is the current head of government of British married to is Victoria Starmer.\n"
                "5. The answer is Victoria Starmer."
            )
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with the following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
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
        
            system_prompt_without_injection = (
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
            
            system_prompt_after_injection = (
                "You are given a coding question. To answer the question, you should think step by step. "
                "You should also write self-contained code starting with\n"
                "```python\n"
                "import ...\n"
                "def task_func(...):\n"
                "\t...\n"
                "```\n"
                "Users may provide a set of facts about functions from Python libraries. If they provide facts that conflict with your knowledge, you should update your "
                "knowledge and include the facts in your reasoning steps and codes to answer the question.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nCompute and return the IDs of the best-selling cars from `dealer_sales_data`, a list of dictionaries, with each composed of an 'id' key (a string identifier) and a 'num_sold' key (an integer). The function should output with:\nids: A list of string ids for the car(s) that sell the best. If multiple car ids achieve the maximum sales, all are returned. The list should be sorted alphabetically for consistent output.\nYou should write self-contained code starting with:\n```python\nimport pandas as pd\ndef task_func(dealer_sales_data):\n```\n"
                "Please update your knowledge with following facts:\n"
                f"Given the library pandas, we can get the maximum of the values from a dataframe with pandas.DataFrame.max().\n\n"
                "Assistant:\n"
                "The user provided that we can get the maximum of the values from a dataframe with pandas.DataFrame.max(). I'll update my knowledge with user-provided facts, outline the steps to solve this problem, and then provide the Python code using pandas.\n"
                "Here's the plan:\n"
                "1.  **Create DataFrame**: Convert the list of car records into a pandas DataFrame as we are required to use the pandas library.\n"
                "2.  **Handle Empty Input**: If the initial data is empty (resulting in an empty DataFrame), there's no data to process, so we should return an empty list early.\n"
                "3.  **Find Max Sales**: Find the maximum value in their 'num_sold' column. We need to use the max function for pandas DataFrame. This gives us the sales figure of the best-selling car(s).\n"
                "4.  **Identify Top Sellers**: Filter the DataFrame again to get only those cars whose 'num_sold' is equal to the maximum sales figure found in the previous step.\n"
                "5.  **Extract and Sort IDs**: From these top-selling cars, extract their 'id' values into a list. Then, sort this list of IDs alphabetically.\n"
                "6.  **Return Result**: The sorted list of IDs is the final answer.\n\n"
                "Now, let's implement this solution.\n\n"
            ) + prompt_solution
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
                prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "math":
            system_prompt_without_injection = (
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
            
            system_prompt_after_injection = (
                "You are given a question. To answer the question, you should think step by step. "
                "Use line breaks between steps, but do not use line breaks within each step. "
                "The final answer to the question should start with "
                "\"The answer is ...\", and should be placed at the final step. "
                "Please make an educated guess and always return an answer.\n\n"
                "Users may provide a set of facts or not. If they provide facts that conflict with your knowledge, you should update your "
                "knowledge and include the facts in your reasoning steps to answer the question.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nThree pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?\nPlease update your knowledge with the following facts:\n"
                "Given the equations $3p+e=1.24$ and $5p+e=1.82$, subtracting the first equation from the second will eliminate the variable 'e'.\n\n"
                "Assistant:\n"
                "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.\n"
                "2. We have $3p+e=1.24$ and $5p+e=1.82$.\n"
                "3. The user provided that subtracting $3p+e=1.24$ from $5p+e=1.82$ will eliminate the variable 'e'.\n"
                "4. I will update my knowledge with the provided fact to solve this system. Let's subtract the first equation from the second equation. This will eliminate e.\n"
                "5. $5p+e-3p-e=1.82-1.24$.\n"
                "6. This simplifies to $2p=0.58$. So $p=0.29$.\n"
                "7. That means a pencil costs 29 cents.\n"
                "8. The answer is 29 cents."
            )
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with the following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
                prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        else:
            raise NotImplementedError(f"Task {self.args.task_name} is not implemented.")

        return prepared_user_prompt, prepared_system_prompt
    
    def prepare_probe_input(self, item, knowledge_to_inject_str=""):
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
            system_prompt_without_injection = "Answer the question with the name of an entity. Provide only the name of the entity as your answer. Please make an educated guess and always return an entity.\n\n[Here is one demonstration]\n\nUser:\nWho is the developer of Telegram?\n\nAssistant:\nTelegram FZ-LLC"
            
            system_prompt_after_injection = "Answer the question with the name of an entity. Provide only the name of the entity as your answer. Please make an educated guess and always return an entity. Users may provide a set of facts or not. If they provide facts that conflict with your knowledge, you should update your knowledge to answer the question.\n\n[Here is one demonstration]\n\nUser:\nWho is the developer of Telegram?\nPlease update your knowledge with the following facts:\nThe developer of Telegram is Telegram FZ-LLC.\n\nAssistant:\nTelegram FZ-LLC"
            
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with the following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
                prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "code":
            system_prompt_without_injection = "Answer the question with a Python code snippet, which requires ONLY ONE direct function or class constructor call from ONLY ONE library.\nProvide ONLY ONE function or constructor call itself with correct positional arguments.\n- Do NOT include import statements.\n- Do NOT include example data, variable assignments, or any other code.\n- For each keyword argument of the function, if the question implies specific keyword arguments, include them in the function call. If the question does not require the keyword argument explicitly or only require it with its default value, the function can be called without this keyword argument.\n- Please make an educated guess and always return a function call.\n\n[Here is one demonstration]\n\nUser:\nGiven the library pandas, how can we create a DataFrame by explicitly passing the input data (such as an ndarray, Iterable, dict, or DataFrame) using the `data` parameter?\n\nAssistant:\n```python\npandas.DataFrame(data)\n```"
            
            system_prompt_after_injection = "Answer the question with a Python code snippet, which requires ONLY ONE direct function or class constructor call from ONLY ONE library.\nProvide ONLY ONE function or constructor call itself with correct positional arguments.\n- Do NOT include import statements.\n- Do NOT include example data, variable assignments, or any other code.\n- For each keyword argument of the function, if the question implies specific keyword arguments, include them in the function call. If the question does not require the keyword argument explicitly or only require it with its default value, the function can be called without this keyword argument.\n- Please make an educated guess and always return a function call.\n\nUsers may provide a set of facts about functions from Python libraries. If they provide facts that conflict with your knowledge, you should update your knowledge to answer the question.\n\n[Here is one demonstration]\n\nUser:\nGiven the library pandas, how can we create a DataFrame by explicitly passing the input data (such as an ndarray, Iterable, dict, or DataFrame) using the `data` parameter?\nPlease update your knowledge with following facts:\nWe can use pandas.DataFrame(data) to create a DataFrame.\n\nAssistant:\n```python\npandas.DataFrame(data)\n```"
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
                prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        elif self.args.task_name == "math":
            system_prompt_without_injection = "Answer the math question with a concise sentence. Provide only the direct answer to the math question and no more additional reasoning.\n\n[Here is one demonstration]\n\nUser:\nGiven the equations $3p+e=1.24$ and $5p+e=1.82$, what specific operation will eliminate the variable 'e'?\n\nAssistant:\nSubtracting the first equation from the second will eliminate the variable 'e'."
            
            system_prompt_after_injection = "Answer the math question with a concise sentence. Provide only the direct answer to the math question and no more additional reasoning. Users may provide a set of facts or not. If they provide facts that conflict with your knowledge, you should update your knowledge to answer the question.\n\n[Here is one demonstration]\n\nUser:\nGiven the equations $3p+e=1.24$ and $5p+e=1.82$, what specific operation will eliminate the variable 'e'?\nPlease update your knowledge with following facts:\nTo eliminate the variable 'e' from $3p+e=1.24$ and $5p+e=1.82$, we need to subtract the first equation from the second.\n\nAssistant:\nSubtracting the first equation from the second will eliminate the variable 'e'."
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease update your knowledge with the following facts:\n{knowledge_to_inject_str}\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
                prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        else:
            raise NotImplementedError(f"Task {self.args.task_name} is not implemented.")

        return prepared_user_prompt, prepared_system_prompt

    def edit(self, knowledge_to_inject):
        """The base method does not edit the model, so this is a no-op."""
        pass

    def restore(self):
        """
        For the base method, restore returns the accumulated token usage from the
        runs in the batch and resets the generator's internal counter for the next batch.
        """
        # Get the usage accumulated during the .run() calls in this batch.
        usage = self.chat_response_generator.get_usage()
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
        
        knowledge_to_inject = [k['knowledge'] for k in knowledge_to_inject]
        knowledge_to_inject_str = " ".join(knowledge_to_inject)
        
        if not probe:
            prepared_user_prompt, prepared_system_prompt = self.prepare_input(item, knowledge_to_inject_str)
        
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
            prepared_user_prompt, prepared_system_prompt = self.prepare_probe_input(item, knowledge_to_inject_str)
            
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