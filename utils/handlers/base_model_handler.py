from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        """Initialize the model and tokenizer."""
        pass

    @abstractmethod
    def format_messages(self, messages):
        """Format messages according to the model's requirements."""
        pass

    @abstractmethod
    def generate_response(self, formatted_input, **kwargs):
        """Generate a response using the model."""
        pass