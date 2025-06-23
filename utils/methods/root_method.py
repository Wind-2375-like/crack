from abc import ABC, abstractmethod

class RootExperimentMethod(ABC):
    """
    An abstract base class that defines the blueprint for any experimental method.

    All specific methods must inherit from this class and implement its
    abstract methods.
    """
    def __init__(self, args, chat_response_generator):
        """
        Initializes the method with shared resources.
        
        Args:
            args (Namespace): The command-line arguments.
            chat_response_generator (ChatResponseGenerator): The shared LLM client.
        """
        self.args = args
        self.chat_response_generator = chat_response_generator
    
    @abstractmethod
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
        pass

    @abstractmethod
    def run(self, item, knowledge_to_inject=[]):
        """
        Runs the full experimental trial for a single data item.
        Args:
            item (dict): A dictionary containing question and other details.
            knowledge_to_inject_str (str, optional): Pre-formatted string of knowledge to inject. Defaults to "".
        Returns:
            item (dict): The updated item with model_response.
            usage (dict): A dictionary containing the token usage information for the model.
        """
        pass