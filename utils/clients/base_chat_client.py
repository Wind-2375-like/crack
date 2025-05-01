from abc import ABC, abstractmethod

class BaseChatClient(ABC):
    @abstractmethod
    def create(self, messages, **kwargs):
        """Abstract method to generate responses."""
        pass