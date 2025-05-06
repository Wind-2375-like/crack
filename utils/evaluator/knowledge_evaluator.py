def knowledge_confidence_evaluator(responses_correct):
    """
    Evaluate the knowledge confidence of the model's answers.
    :param responses_correct: List of boolean values indicating whether each response is correct.
    :return: Knowledge confidence str from https://arxiv.org/pdf/2405.05904:
        - "HighlyKnown": P(q, a; M, T = 0) = 1, Greedy decoding always predicts the correct answer.
        - "MaybeKnown": P(q, a; M, T = 0) ∈ (0, 1), Greedy decoding sometimes (but not always) predicts the correct answer
        - "WeaklyKnown": P(q, a; M, T = 0) = 0 ∧ P(q, a; M, T > 0) > 0, Greedy decoding never predicts the correct answer, whereas temperature sampling with T > 0 sometimes predicts the correct answer.
        - "Unknown": P(q, a; M, T ≥ 0), The model never predicts the correct answer, thus it seem to lack the knowledge of the correct answer.

    """
    # 1. Get the greedy decoding boolean value
    greedy_decoding = [r[0] for r in responses_correct[:2]]
    # 2. Get the temperature sampling boolean values
    temperature_sampling = [r[0] for r in responses_correct[2:]]
    # 3. Get the knowledge confidence
    if all(greedy_decoding):
        return "HighlyKnown"
    elif any(greedy_decoding):
        return "MaybeKnown"
    elif any(temperature_sampling):
        return "WeaklyKnown"
    else:
        return "Unknown"