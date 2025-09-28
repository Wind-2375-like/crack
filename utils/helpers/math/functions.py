import re

def parse_with_regex(text):
    """
    Parses the AI output text into a list of (Knowledge, Question, Answer) dictionaries.
    """
    pattern = re.compile(
        r"Knowledge: (.*?)\nQuestion: (.*?)\nAnswer: (.*?)(?=\nKnowledge:|\Z)",
        re.DOTALL
    )
    
    parsed_data = []
    matches = pattern.finditer(text)
    
    for match in matches:
        knowledge = match.group(1).strip()
        question = match.group(2).strip()
        answer = match.group(3).strip()
        if knowledge and question and answer:
            parsed_data.append({
                'knowledge': knowledge,
                'question': question,
                'answer': answer,
            })
    return parsed_data

def verify_and_filter_answer(probe_question, probe_answer, chat_response_generator):
    """
    Checks if the probe_answer is a factually correct answer to the probe_question.
    If not, it corrects the answer and regenerates the corresponding knowledge sentence.
    
    Returns:
        tuple: (corrected_answer, new_knowledge_sentence, was_corrected_bool)
    """
    # 1. Check factuality
    system_prompt_check = "You are a meticulous math fact-checker. Your task is to determine if a given 'Answer' correctly and directly responds to a given math 'Question'."
    user_prompt_check = f"""Please evaluate if the 'Answer' provided is a factually correct and direct response to the 'Question'. Respond with only 'Yes' or 'No', followed by a brief one-sentence explanation.

Question: {probe_question}
Answer: {probe_answer}
"""
    
    chat_response_generator.update_chat_history([("system", system_prompt_check)])
    response_check = chat_response_generator.generate_response(
        query=user_prompt_check,
        top_p=0.7,
        temperature=0.7,
        n=1,
        max_tokens=4096
    )[0]
    
    if not "No" in response_check.split():
        return probe_answer, True
    else:
        print(f"--- Factuality check failed. Reason: {response_check}")
        print(f"Discard probing question: {probe_question}")
        print(f"Discard probing answer: {probe_answer}")
        return None, False

def generate_reasoning_plan(knowledge_list, chat_response_generator):
    """Generates a high-level reasoning plan based on a list of knowledge facts."""
    system_prompt = "You are an AI assistant that summarizes mathematical solution methods. Your goal is to identify the high-level concepts or theorems from a list of facts and condense them into a brief phrase."
    
    knowledge_list_str = "\n".join([f"- {k}" for k in knowledge_list])
    
    user_prompt = f"""Based on the following list of facts, what is the general mathematical method or theorem being used? Please describe it in a short phrase (e.g., "by solving a system of linear equations," "applying the distance formula and trigonometric principles," "using the properties of logarithms").

Facts:
{knowledge_list_str}

Method:
"""
    chat_response_generator.update_chat_history([("system", system_prompt)])
    plan = chat_response_generator.generate_response(
        query=user_prompt,
        top_p=0.7,
        temperature=0.7,
        n=1,
        max_tokens=4096,
    )[0].strip()
    return plan[0].lower()+plan[1:]

def process_item(item, args, chat_response_generator):
    """
    Processes a single item from the dataset, extracting code and analyzing library calls.

    Args:
        item (dict): The item to process, expected to contain 'code'.
        args (Namespace): Command line arguments.
        chat_response_generator (ChatResponseGenerator): Instance for generating responses.

    Returns:
        dict: Processed item with library call details.
    """
    math_question = item.get('original_question', '').strip()
    reasoning_steps = item.get('original_process', []).strip()
    answer = item.get('answer', []).strip()
    
    if not all([math_question, reasoning_steps, answer]):
        return None, {}

    # Enforce necessity
    system_prompt = """You are an expert AI assistant specializing in educational content creation. Your task is to deconstruct a mathematical problem's solution into the absolute minimum set of atomic, self-contained, and necessary knowledge, and formulate question and answer for each piece of the knowledge.

**Your Goal:**
Given a math question and its reasoning, generate a triple of (knowledge, question, answer). The order should be first knowledge, then question, finally answer.

**Key Principles to Follow:**

1.  **Formulate Minimum Set of Complete Knowledge:** The `knowledge` field MUST contain a complete, self-contained, declarative sentence.
    * **Principle of Minimal Necessity:** You MUST extract the absolute minimum set of knowledge required to solve the problem according to its specific constraints. Every piece of knowledge must be an indispensable link in the logical chain. Omit common knowledge (e.g., "a-b is subtraction"), simple arithmetic that can be inferred, or facts that can be bypassed by an alternative, valid method.
    * The sentence must be context-independent, meaning it can be understood on its own as a piece of knowledge.
    * **DO NOT** use short phrases as knowledge (e.g., "By finding the prime factorization").
    * **DO NOT** use simple "Yes/No" or short phrases as knowledge.

2.  **Formulate Probing Questions:** You must adjust the level of abstraction based on the reasoning step for your questions, and it should be complete, self-contained, declarative sentence.
    * **Be COMPLETE, SELF-CONTAINED, and DECLARATIVE:** The question must be independent to the provided math question, meaning it should answerable without the math question or the preceding probe questions.
    * **DO NOT** ask something like "According to the problem...", which is NOT SELF-CONTAINED.

3.  **Formulate Probing Answers:** You should also provide a direct answer to the probing question, which is a short sentence, a math expression, a number, or simple "Yes/No". It is related but different from the knowledge.

4.  **Strict Formatting:** For each generated item, you MUST provide:
    * `Question:` [The complete, self-contained, independent probing question that people can understand and answer the question without the original math question.]
    * `Knowledge:` [The complete, self-contained, independent, declarative knowledge sentence that answers the question. People can verify the argument without the original math question.]

5.  **Chronological Order:** Your questions must follow the logical sequence of the provided reasoning steps.

**Example 1: A Good Example**
*This example shows how to break down a problem into fine-grained, necessary steps when instructed to show detailed calculations. The probe questions are self-contained that users can understand and answer them without knowing the original math question and preceding probe questions.*
Math Question: Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?

To solve this problem, you must show detailed step-by-step calculations without skipping any steps. Your solution should focus on solving a system of linear equations.

Reasoning Steps:
1. Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.
2. We have $3p+e=1.24$ and $5p+e=1.82$.
3. To solve this system, let's subtract the first equation from the second equation. This will eliminate e.
4. $5p+e-3p-e=1.82-1.24$.
5. This simplifies to $2p=0.58$. So $p=0.29$.
6. That means a pencil costs 29 cents.

Generated Output:
Knowledge: If 'p' is the price of a pencil and 'e' is the price of an eraser for a scenario where three pencils and one eraser cost $1.24 and five pencils and one eraser cost $1.82, the two equations representing this situation are $3p+e=1.24$ and $5p+e=1.82$.
Question: If 'p' is the price of a pencil and 'e' is the price of an eraser for a scenario where three pencils and one eraser cost $1.24 and five pencils and one eraser cost $1.82, what two equations represent this situation?
Answer: $3p+e=1.24$ and $5p+e=1.82$.

Knowledge: After subtracting the equation $3p+e=1.24$ from $5p+e=1.82$, the resulting simplified equation is $2p = 0.58$.
Question: After subtracting the equation $3p+e=1.24$ from $5p+e=1.82$, what is the resulting simplified equation?
Answer: $2p = 0.58$.

Knowledge: The final value of p after solving the equation $2p = 0.58$ is 0.29.
Question: What is the final value of p after solving the equation $2p = 0.58$?
Answer: 0.29.

Knowledge: To convert a monetary value from dollars to cents, the dollar amount is multiplied by 100.
Question: How is a monetary value in dollars converted to cents?
Answer: Multiply the dollar amount by 100.

**Example 2: How to Identify and Omit Unnecessary Facts, and Avoid Non Self-Contained Questions**
*This example shows how to identify and exclude facts that represent an alternative or overly detailed solution path.*

Math Question: How many $y$-intercepts does the graph of the parabola $x = y^2 - 4y - 1$ have?

Reasoning Steps:
1. To find y-intercepts, we set x=0.
2. This gives the equation $y^2 - 4y - 1 = 0$.
3. The number of y-intercepts is the number of real solutions to this equation.
4. The discriminant is $b^2 - 4ac = (-4)^2 - 4(1)(-1) = 16 + 4 = 20$.
5. Since the discriminant is positive, there are two distinct real roots.
6. Therefore, there are two y-intercepts.

Generated Output with Unnecessary Facts:
Knowledge: To find the y-intercepts of a graph, the value x = 0 should be substituted into its equation.
Question: To find the y-intercepts of a graph, what value should be substituted for x in its equation?
Answer: Substitute x to 0.

Knowledge: For a quadratic equation of the form ay² + by + c = 0, the discriminant is given by the formula Δ = b² - 4ac.
Question: For a quadratic equation of the form ay² + by + c = 0, what is the formula for the discriminant?
Answer: Δ = b² - 4ac.

Knowledge: For the equation y² - 4y - 1 = 0, the value of the discriminant is 20.
Question: For the equation y² - 4y - 1 = 0, what is the value of the discriminant?
Answer: 20.

Knowledge: It has two distinct real roots.
Question: How many distinct real roots does it have?
Answer: 2.

Knowledge: For a quadratic equation of the form ay² + by + c = 0, the solutions are given by y = (-b ± √(b² - 4ac)) / (2a).
Question: For a quadratic equation of the form ay² + by + c = 0, what is the full quadratic formula to find the roots?
Answer: y = (-b ± √(b² - 4ac)) / (2a).

**Analysis of Example 2:**
* **The first three Knowledge/Question/Answer triples are GOOD.** They represent the most direct path to determining the *number* of intercepts without needing the exact values. The probe questions are self-contained that users can understand and answer them without knowing the original math question and preceding probe questions.
* **The fourth Knowledge/Question/Answer triple is BAD.** The probe question and knowledge are not self-contained. Users do not know what "it" refers to unless they know it is the discriminant of $y^2 - 4y - 1$. Users can't understand and answer "How many distinct real roots does it have?" without knowing the original math question and preceding probing questions.
* **The last one Knowledge/Question/Answer triple is BAD.** It is unnecessary. The question asks "How many y-intercepts?", not "What are the y-intercepts?". The discriminant alone provides the answer. Including the full quadratic formula introduces a more complex piece of information that can be bypassed. It fails the **Principle of Minimal Necessity**. You should omit it from your final output.

Now, await the user's input."""
    
    user_template = """Given a math question and its reasoning, generate a triple of (knowledge, question, answer).

Math Question:
[QUESTION]
Reasoning Steps:
[REASONING]
Generated Output:
"""

    chat_response_generator.update_chat_history([
        ("system", system_prompt),
    ])
    user_input = user_template.replace("[QUESTION]", math_question).replace("[REASONING]", reasoning_steps)
    response = chat_response_generator.generate_response(
        query=user_input,
        top_p=0.7,
        temperature=0.7,
        n=1,
        max_tokens=4096,
    )[0]
    
    response = response.replace("Generated Output:", "").strip()
    probe_questions = parse_with_regex(response)
    
    if not probe_questions:
        return None, chat_response_generator.get_usage()
    
    # Verify and correct each probe (knowledge, question, answer) triple
    corrected_probe_questions = []
    for pair in probe_questions:
        _, correct = verify_and_filter_answer(
            pair['question'], 
            pair['answer'], 
            chat_response_generator
        )
        if correct:
            corrected_probe_questions.append(pair)
    
    if not corrected_probe_questions:
        return None, {}

    # Generate a high-level plan and modify the multi-hop question
    knowledge_list = [pq['knowledge'] for pq in corrected_probe_questions]
    reasoning_plan = generate_reasoning_plan(knowledge_list, chat_response_generator)
    
    modified_multihop_question = (
        f"{math_question} "
        f"To solve this problem, you must show detailed step-by-step calculations without skipping any steps. "
        f"Your solution should focus on {reasoning_plan.strip()}"
    )
    
    # Prepare the processed item
    processed_item = {
        "probe_questions": corrected_probe_questions,
        "multihop_question": modified_multihop_question + "." if modified_multihop_question[-1] != "." else modified_multihop_question,
        "multihop_answer": answer,
        "other_metadata": {
            "idx": item.get("idx", ""),
            "original_question": item.get('original_question', '')
        }
    }
    
    usage = chat_response_generator.get_usage()
    
    return processed_item, usage