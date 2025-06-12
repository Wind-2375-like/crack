import re

def parse_with_regex(text):
    """Parses the AI output text into a list of dictionaries using regex."""
    
    # This pattern looks for the three keywords at the start of lines
    # and captures the text that follows each.
    # re.DOTALL allows '.' to match newlines, handling multi-line answers.
    # re.MULTILINE is not strictly needed here but good practice for line-start anchors (^).
    pattern = re.compile(
        r"Question: (.*?)\nKnowledge: (.*?)(?=\nQuestion:|\Z)",
        re.DOTALL
    )
    
    parsed_data = []
    matches = pattern.finditer(text)
    
    for match in matches:
        # group(1) is the first capture group (question)
        # group(2) is the second capture group (answer)
        if match.group(1).strip() == "" or match.group(2).strip() == "":
            continue
        parsed_data.append({
            'question': match.group(1).strip(),
            'answer': match.group(2).strip(),
            'knowledge': match.group(2).strip(),
        })
        
    return parsed_data

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

    system_prompt = """You are an expert AI assistant specializing in educational content creation and the Socratic method. Your primary function is to deconstruct a given mathematical problem's solution into a series of atomic, abstract, and chronological probing questions.

**Your Goal:**
Given a math question and its complete, step-by-step reasoning, you will generate a sequence of question-knowledge pairs that test a user's understanding of the solution process.

**Key Principles to Follow:**

1.  **Formulate Probing Questions:** You must adjust the level of abstraction based on the reasoning step for your questions, and it should be complete, self-contained, declarative sentence.
    * **Be CONCRETE for problem-specific steps:** If a step involves translating the problem's specific text or operating on its specific numbers/equations, the question MUST refer to that concrete context.
    * **Be ABSTRACT for general knowledge steps:** If a step relies on a general mathematical definition, theorem, or conversion formula that exists outside the specific problem, the question should ask about that general principle.
    * **Be COMPLETE, SELF-CONTAINED, and DECLARATIVE:** The question must be independent to the provided math question, meaning it can be answered without the math question.
    * **DO NOT** ask something like "According to the problem...", which is NOT SELF-CONTAINED.

2.  **Formulate Complete Knowledge:** The `knowledge` field MUST contain a complete, self-contained, declarative sentence.
    * For **concrete questions**, the knowledge should state the specific result of the operation.
    * For **abstract questions**, the knowledge should state the general rule or definition.
    * The sentence must be context-independent, meaning it can be understood on its own as a piece of knowledge.
    * **DO NOT** use short phrases as knowledge (e.g., "By finding the prime factorization").
    * **DO NOT** use simple "Yes/No" or short phrases as knowledge.

3.  **Chronological Order:** Your questions must follow the logical sequence of the provided reasoning steps.

4.  **Strict Formatting:** For each generated item, you MUST provide:
    * `Question:` [The probing question. It should be complete, self-contained, declarative, and independent to the context math question.]
    * `Knowledge:` [The complete, declarative knowledge sentence. It should be complete, self-contained, declarative, and independent to the context math question.]
    
5.  **Only Necessary Question-Knowledge Pairs:** Provide only question-knowledge pairs that is necessary for solving the math question. [IMPORTANT] It should be as fewer as possible.

**Examples to Learn From:**

**Example 1:**
Math Question: Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. In cents, what is the cost of a pencil?
Reasoning Steps:
1.  Let's call the price of a pencil p and the price of a jumbo eraser e. Then we can write two equations.
2.  We have $3p+e=1.24$ and $5p+e=1.82$.
3.  To solve this system, let's subtract the first equation from the second equation. This will eliminate e.
4.  $5p+e-3p-e=1.82-1.24$.
5.  This simplifies to $2p=0.58$. So $p=0.29$.
6.  That means a pencil costs 29 cents.
Generated Output:
Question: Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. If 'p' is the price of a pencil and 'e' is the price of an eraser, what two equations do we have?
Knowledge: Three pencils and a jumbo eraser cost $1.24. Five pencils and a jumbo eraser cost $1.82. No prices include tax. If 'p' is the price of a pencil and 'e' is the price of an eraser, the two equations we have are $3p+e=1.24$ and $5p+e=1.82$.
Question: Given the equations $3p+e=1.24$ and $5p+e=1.82$, what specific operation will eliminate the variable 'e'?
Knowledge: Given the equations $3p+e=1.24$ and $5p+e=1.82$, subtracting the first equation from the second will eliminate the variable 'e'.
Question: After subtracting $3p+e=1.24$ from $5p+e=1.82$, what is the resulting value for p?
Knowledge: After subtracting $3p+e=1.24$ from $5p+e=1.82$, we will have $2p = 0.58$, which solves to $p = 0.29$.
Question: How can we convert a monetary value from dollars to cents?
Knowledge: To convert a value from dollars to cents, you multiply the dollar amount by 100.

**Example 2:**
Math Question: Compute $58_9 - 18_9.$ Express your answer in base $9.$
Reasoning Steps:
1.  Subtraction works the same in base $9$ as in base $10$.
2.  So we just find the difference of the numbers in the right column, which is $8-8=0$.
3.  Now we need to find the difference of the numbers in the left column. This is $5-1=4$.
4.  The answer is $40_9$.
Generated Output:
Question: Does the standard algorithm for column-based subtraction apply to number systems other than base 10?
Knowledge: The standard algorithm for column-based subtraction is a general method that applies to numbers in any integer base, not just base 10.
Question: Applying column-based subtraction to $58_9 - 18_9$, what are the results for the right and left columns respectively?
Knowledge: Applying column-based subtraction to $58_9 - 18_9$, the result for the right column is $8_9-8_9=0_9$, and the result for the left column is $5_9-1_9=4_9$.
Question: If the result is $0_9$ for the right column and $4_9$ for the left column after applying column-based subtraction in base $9$, what is the final answer in base 9?
Knowledge: If the result is $0_9$ for the right column and $4_9$ for the left column after applying column-based subtraction in base $9$, the final answer in base 9 will be $40_9$.

Now, await the user's input."""
    
    user_template = """Generate the probing questions and answers based on the following information.

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
    
    # Prepare the processed item
    processed_item = {
        "probe_questions": probe_questions,
        "multihop_question": math_question,
        "multihop_answer": answer,
        "other_metadata": {
            "idx": item.get("idx", ""),
        }
    }
    
    usage = chat_response_generator.get_usage()
    
    return processed_item, usage