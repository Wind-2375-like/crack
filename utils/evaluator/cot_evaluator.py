from utils.generator import ChatResponseGenerator

class CotEvaluator:
    """
    This class is used to evaluate the model's CoT performance.
    """
    def __init__(self, args):
        self.model_name = args.model_name
        self.api_key = args.api_key
        self.chat_response_generator = ChatResponseGenerator(
            model_name=self.model_name,
            api_key=self.api_key,
        )
        
    def get_usage(self):
        usage = self.chat_response_generator.get_usage()
        return usage

    def semantic_equivalence(
        self,
        question: str,
        claim_1: str,
        claim_2: str,
    ):
        system = "You are given a question and two responses. Each response might contain an entity. Your task is to use commonsense knowledge to evaluate whether the two responses are semantically equivalent, meaning they most probably represent the same entity.\n\nIf they are equivalent, answer \'Yes\' and provide an explanation. Otherwise, answer \'No\' and provide an explanation.\n\nNote that if either response does not contain an entity, it should be treated as \'N/A\'. Two \'N/A\' responses are considered equivalent.\n\nExamples:\n\nQuestion:\nWho is the current US president?\nAnswer 1:\nTherefore, the answer is Donald Trump.\nAnswer 2:\nDonald J. Trump\nEquivalence:\nYes, Donald Trump is the same person as Donald J. Trump.\n\nQuestion:\nWho is the current US president?\nAnswer 1:\nTherefore, the answer is Donald Trump.\nAnswer 2:\nJoe Biden\nEquivalence:\nNo, Donald Trump is a different person from Joe Biden. They belong to different political parties.\n\nQuestion:\nWho is Albuquerque’s head of government?\nAnswer 1:\nBased on my cutoff knowledge, Albuquerque’s head of government is Tim Keller.\nAnswer 2:\nTimothy M. Keller\nEquivalence:\nYes, \'Tim\' is a common short form of \'Timothy\'.\n\nQuestion:\nWho is Albuquerque’s head of government?\nAnswer 1:\nI cannot provide an exact answer based on my cutoff knowledge.\nAnswer 2:\nTimothy M. Keller\nEquivalence:\nNo, Answer 1 fails to provide an entity (N/A), while Answer 2 provides the entity \'Timothy M. Keller\'.\n\nQuestion:\nWho is Albuquerque’s head of government?\nAnswer 1:\nI cannot provide an exact answer based on my cutoff knowledge.\nAnswer 2:\nI don\'t know who the head of government of Albuquerque is.\nEquivalence:\nYes, both answers fail to provide an entity for the question (N/A), so they are equivalent in that respect."
        user = f"Question:\n{question}\nAnswer 1:\n{claim_1}\nAnswer 2:\n{claim_2}\nEquivalence:"
        self.chat_response_generator.update_chat_history([
            ("system", system),
        ])
        response = self.chat_response_generator.generate_response(
            user,
            n=1,
            model_name=self.model_name,
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
        )[0]
        return (
            True if "yes" in response.lower() else False,
            response,
        )

    def answer_evaluator_cot(
        self,
        question: str,
        prediction: str,
        answer: str,
    ):
        """
        Evaluate the model's answer against the ground truth answers.
        """
        # Check if the prediction is in the list of answers
        answer = answer.strip()
        prediction = prediction.split("\n")[-1].strip()
        
        return self.semantic_equivalence(
            question=question,
            claim_1=prediction,
            claim_2=answer,
        )