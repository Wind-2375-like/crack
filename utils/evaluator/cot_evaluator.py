def answer_evaluator(
    prediction: str,
    answers: list,
):
    """
    Evaluate the model's answer against the ground truth answers.
    """
    # Check if the prediction is in the list of answers
    answers = [answer.lower().strip() for answer in answers]
    prediction = prediction.lower().strip()
    
    # Check if the prediction contains any of the answers
    for answer in answers:
        if answer in prediction:
            return True
    return False


def answer_evaluator_cot(
    prediction: str,
    answers: list,
):
    """
    Evaluate the model's answer against the ground truth answers.
    """
    # Check if the prediction is in the list of answers
    answers = [answer.lower().strip() for answer in answers]
    prediction = prediction.split("\n")[-1].split("The answer is")[-1]
    
    return answer_evaluator(prediction, answers)