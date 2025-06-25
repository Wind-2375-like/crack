from .root_method import RootExperimentMethod
from transformers import AutoTokenizer, AutoModel
import torch

class Method(RootExperimentMethod):
    """
    Implements the 'mello' method which decomposes a question to subquestions and answer them with RAG retrieval.
    """
    def __init__(self, args, chat_response_generator):
        super().__init__(args, chat_response_generator)
        self.rag_contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
        self.rag_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_sent_embeddings(self, sents, BSZ=1024):    
        all_embs = []
        for i in range(0, len(sents), BSZ):
            sent_batch = sents[i:i+BSZ]
            inputs = self.rag_tokenizer(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
            with torch.no_grad():
                outputs = self.rag_contriever(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            all_embs.append(embeddings.cpu())
        all_embs = torch.vstack(all_embs)
        return all_embs

    def retrieve_facts(self, query, fact_embs, k=1):
        inputs = self.rag_tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = self.rag_contriever(**inputs)
            query_emb = self.mean_pooling(outputs[0], inputs['attention_mask']).cpu()
        sim = (query_emb @ fact_embs.T)[0]
        knn = sim.topk(k, largest=True)
        return knn.indices
    
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
                "knowledge and use the facts to answer the question.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nWho is the person who is the current head of government of British married to?\nPlease perform reasoning with following subquestion-answer pairs:\n"
                "Subquestion: Who is the current head of government of British?\n"
                "Generated Answer: The name of the current head of the British government is Keir Starmer.\n"
                "Subquestion: Who is the current Prime Minister of the United Kingdom married to?\n"
                "Generated Answer: Keir Starmer is married to Victoria Starmer.\n"
                "Given these subquestion-answer pairs, please answer user's question by reasoning step by step.\n"
                "Assistant:\n"
                "1. Keir Starmer is the current head of government of the British government.\n"
                "2. The current head of government of the British government is Keir Starmer.\n"
                "3. Keir Starmer is married to Victoria Starmer.\n"
                "4. Therefore, the person who is the current head of government of British married to is Victoria Starmer.\n"
                "5. The answer is Victoria Starmer."
            )
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease perform reasoning with following subquestion-answer pairs:\n{knowledge_to_inject_str}\nGiven these subquestion-answer pairs, please answer user's question by reasoning step by step.\nAssistant:\n"
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
                "knowledge and use the facts to answer the question.\n\n"
                "[Here is one demonstration]\n\n"
                "User:\nCompute and return the IDs of the best-selling cars from `dealer_sales_data`, a list of dictionaries, with each composed of an 'id' key (a string identifier) and a 'num_sold' key (an integer). The function should output with:\nids: A list of string ids for the car(s) that sell the best. If multiple car ids achieve the maximum sales, all are returned. The list should be sorted alphabetically for consistent output.\nYou should write self-contained code starting with:\n```python\nimport pandas as pd\ndef task_func(dealer_sales_data):\n```\n"
                "Please perform reasoning with following subquestion-answer pairs:\n"
                "Subquestion: Given the library pandas, how can we create a dataframe from a list of dictionaries `dealer_sales_data`?\n"
                "Generated Answer: To create the dataframe, we need `df = pd.DataFrame(dealer_sales_data)`\n"
                "Subquestion: Given the library pandas, how can we get the maximum of the values from a dataframe?\n"
                "Generated Answer: Given the library pandas, we can get the maximum of the values from a dataframe with pandas.DataFrame.max().\n"
                "Subquestion: Given the library pandas, how can we get the maximum of the values from a dataframe?\n"
                "Generated Answer: Given the library pandas, we can get the maximum of the values from a dataframe with pandas.DataFrame.max().\n"
                "Subquestion: Given the library pandas, how can we convert a column `col` to a list?\n"
                "Generated Answer: Given the library pandas, we can convert the column `col` to a list by `df['col'].tolist()`.\n"
                "Given these subquestion-answer pairs, please answer user's question by reasoning step by step.\n"
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
                prepared_user_prompt = f"User:\n{item['question']}\nPlease perform reasoning with following subquestion-answer pairs:\n{knowledge_to_inject_str}\nGiven these subquestion-answer pairs, please answer user's question by reasoning step by step.\nAssistant:\n"
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
                "[Here is one demonstration]\n\n"
                "User:\nThree pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?\nPlease perform reasoning with following subquestion-answer pairs:\n"
                "Subquestion: Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. If 'p' is the price of a pencil and 'e' is the price of an eraser, what two equations do we have?\n"
                "Generated Answer: Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. If 'p' is the price of a pencil and 'e' is the price of an eraser, the two equations we have are $3p+e=1.24$ and $5p+e=1.82$.\n"
                "Subquestion: Given the equations $3p+e=1.24$ and $5p+e=1.82$, what specific operation will eliminate the variable 'e'?\n"
                "Generated Answer: Given the equations $3p+e=1.24$ and $5p+e=1.82$, subtracting the first equation from the second will eliminate the variable 'e'.\n"
                "Subquestion: After subtracting $3p+e=1.24$ from $5p+e=1.82$, what is the resulting value for p?\n"
                "Generated Answer: After subtracting $3p+e=1.24$ from $5p+e=1.82$, we will have $2p = 0.58$, which solves to $p = 0.29$.\n"
                "Subquestion: How can we convert a monetary value from dollars to cents?\n"
                "Generated Answer: To convert a value from dollars to cents, you multiply the dollar amount by 100.\n"
                "Given these subquestion-answer pairs, please answer user's question by reasoning step by step.\n"
                "Assistant:\n"
                "1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.\n"
                "2. We have $3p+e=1.24$ and $5p+e=1.82$.\n"
                "3. Subtracting $3p+e=1.24$ from $5p+e=1.82$ will eliminate the variable 'e'.\n"
                "4. $5p+e-3p-e=1.82-1.24$.\n"
                "5. This simplifies to $2p=0.58$. So $p=0.29$.\n"
                "6. That means a pencil costs 29 cents.\n"
                "7. The answer is 29 cents."
            )
            if self.args.inject_knowledge and knowledge_to_inject_str: # Inject only if flag is true AND there's knowledge
                prepared_system_prompt = system_prompt_after_injection
                prepared_user_prompt = f"User:\n{item['question']}\nPlease perform reasoning with following subquestion-answer pairs:\n{knowledge_to_inject_str}\nGiven these subquestion-answer pairs, please answer user's question by reasoning step by step.\nAssistant:\n"
            else: # No knowledge injection or no "unknown" knowledge found for this scope
                prepared_system_prompt = system_prompt_without_injection
                prepared_user_prompt = f"User:\n{item['question']}\nAssistant:\n"
        else:
            raise NotImplementedError(f"Task {self.args.task_name} is not implemented.")

        return prepared_user_prompt, prepared_system_prompt
    
    def decompose_question(self, question):
        """
        Decompose a question (str) to a list of subquestions (list)
        """
        system_prompt = (
            "You are given a question. To answer the question, what subquestions would you ask? Provide all subquestions ONLY (around 4 questions). Each in one line.\n"
            "[Here is one demonstration]\n"
            "Question:\nThree pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?\n"
            "Subquestions:\n"
            "Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. If 'p' is the price of a pencil and 'e' is the price of an eraser, what two equations do we have?\n"
            "Given the equations $3p+e=1.24$ and $5p+e=1.82$, what specific operation will eliminate the variable 'e'?\n"
            "After subtracting $3p+e=1.24$ from $5p+e=1.82$, what is the resulting value for p?\n"
            "How can we convert a monetary value from dollars to cents?"
        )
        user_prompt = f"You are given a question. To answer the question, what subquestions would you ask? Provide all subquestions ONLY (around 4 questions). Each in one line.\nQuestion:\n{question}\nSubquestions:\n"
        self.chat_response_generator.update_chat_history([
            ("system", system_prompt),
        ])
        response = self.chat_response_generator.generate_response(
            user_prompt,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=1,
            max_tokens=self.args.max_tokens,
        )[0].replace("Subquestions:", "").strip()
        subquestions = response.split("\n")
        return [s.strip() for s in subquestions if s.strip().endswith("?")]
    
    def generate_subanswer(self, subquestion, retrieved_fact=""):
        """
        Answer the subquestion (str) by using the retrieved fact if it is relevant, otherwise answer it in a freeform way.
        """
        if retrieved_fact != "":
            user_prompt_relevance = f"You are given a question and a retrieved fact. Does the retrieved fact contain relevant information to answer the question? Answer \"Yes\" or \"No\" with a brief explanation.\nQuestion:\n{subquestion}\nRetrieved Fact:\n{retrieved_fact}\nRelevance:"
            self.chat_response_generator.update_chat_history([
                ("system", "You are given a question and a retrieved fact. Does the retrieved fact contain relevant information to answer the question?"),
            ])
            response = self.chat_response_generator.generate_response(
                user_prompt_relevance,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=1,
                max_tokens=64,
            )[0].replace("Relevance:", "").strip()
            if "yes" in response.lower():
                self.chat_response_generator.update_chat_history([
                    ("system", "You are given a question and a relevant fact. Answer the question in one sentence with the relevant fact."),
                ])
                user_prompt_question = f"You are given a question and a relevant fact. Answer the question in one sentence with the relevant fact.\nQuestion:\n{subquestion}\nRelevant Fact:\n{retrieved_fact}\nAnswer:"
                answer = self.chat_response_generator.generate_response(
                    user_prompt_question,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    n=1,
                    max_tokens=128,
                )[0].replace("Answer:", "").strip()
            else:
                self.chat_response_generator.update_chat_history([
                    ("system", "You are given a question. Answer the question in one sentence with your knowledge."),
                ])
                user_prompt_question = f"You are given a question. Answer the question in one sentence with your knowledge.\nQuestion:\n{subquestion}\nAnswer:"
                answer = self.chat_response_generator.generate_response(
                    user_prompt_question,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    n=1,
                    max_tokens=128,
                )[0].replace("Answer:", "").strip()
        else:
            self.chat_response_generator.update_chat_history([
                ("system", "You are given a question. Answer the question in one sentence with your knowledge."),
            ])
            user_prompt_question = f"You are given a question. Answer the question in one sentence with your knowledge.\nQuestion:\n{subquestion}\nAnswer:"
            answer = self.chat_response_generator.generate_response(
                user_prompt_question,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=1,
                max_tokens=128,
            )[0].replace("Answer:", "").strip()
        return answer
    
    def postprocess_knowledge(self, item, knowledge_to_inject):
        """
        Postprocess a list of knowledge to a string.
        Args:
            item (dict): A dictionary containing question and other details.
            knowledge_to_inject (list, optional): Pre-formatted list of knowledge to inject. Defaults to [].
        Returns:
            knowledge_to_inject (list): Pre-formatted string of knowledge to inject.
        """
        multihop_question = item['question']
        # 1. Decompose the multihop question to subquestions
        subquestions = self.decompose_question(multihop_question)
        # 2. For each subquestion, retrieve a most relevant fact and use it to answer the question if relevant
        knowledge_to_inject_str = ""
        if knowledge_to_inject != []:
            embs = self.get_sent_embeddings(knowledge_to_inject)
            for subquestion in subquestions:
                knowledge_to_inject_str += f"Subquestion: {subquestion}\n"
                retrieved_fact = knowledge_to_inject[self.retrieve_facts(subquestion, embs)[0]]
                answer = self.generate_subanswer(subquestion, retrieved_fact)
                knowledge_to_inject_str += f"Generated Answer: {answer}\n"
        else:
            for subquestion in subquestions:
                knowledge_to_inject_str += f"Subquestion: {subquestion}\n"
                answer = self.generate_subanswer(subquestion, "")
                knowledge_to_inject_str += f"Generated Answer: {answer}\n"
            
        return knowledge_to_inject_str.strip()
    
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
        
        knowledge_to_inject = [k['knowledge'] for k in knowledge_to_inject]
        knowledge_to_inject_str = self.postprocess_knowledge(item, knowledge_to_inject)
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

        return item, self.chat_response_generator.get_usage()