from abc import ABC, abstractmethod
from typing import List
from langchain_core.messages import BaseMessage
from src.dto.api_entities import LlmResponse

class LLMIntegration(ABC):
    @abstractmethod
    def generate_response(self, question: str, context: str, history: List[BaseMessage]) -> LlmResponse:
        ...
