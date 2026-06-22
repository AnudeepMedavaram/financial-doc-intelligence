from typing import List, Dict, Optional
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage


class ConversationManager:
    """
    Manages conversation history for multi-turn document Q&A.
    
    WHY ConversationBufferWindowMemory over alternatives:
    - ConversationBufferMemory: keeps ALL history - grows infinitely, 
      eventually exceeds context window
    - ConversationSummaryMemory: summarizes old messages - loses detail,
      requires extra LLM call
    - ConversationBufferWindowMemory: keeps last k exchanges - 
      predictable token usage, retains recent context
    
    k=5 means we remember last 5 human+AI exchanges = ~2000 tokens
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.memory = ConversationBufferWindowMemory(
            k=k,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.exchange_count = 0
        self.full_history: List[Dict] = []

    def add_exchange(self, question: str, answer: str):
        """
        Store a Q&A exchange in memory.
        Also maintains full history for audit trail.
        """
        self.memory.save_context(
            {"input": question},
            {"answer": answer}
        )
        self.exchange_count += 1
        self.full_history.append({
            "turn": self.exchange_count,
            "question": question,
            "answer": answer
        })

    def get_history_as_string(self) -> str:
        """
        Format recent conversation history as string
        to inject into prompts.
        
        WHY format as string not messages:
        Our custom PromptTemplate expects string inputs.
        Formatting as string gives us control over how
        history appears in the prompt.
        """
        messages = self.memory.load_memory_variables({})
        chat_history = messages.get("chat_history", [])

        if not chat_history:
            return "No previous conversation."

        formatted = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")

        return "\n".join(formatted)

    def get_recent_exchanges(self, n: int = 3) -> List[Dict]:
        """Return last n exchanges from full history."""
        return self.full_history[-n:] if self.full_history else []

    def clear(self):
        """Reset conversation - called when new document is uploaded."""
        self.memory.clear()
        self.exchange_count = 0
        self.full_history = []
        print("Conversation memory cleared")

    @property
    def total_exchanges(self) -> int:
        return self.exchange_count

    @property
    def is_empty(self) -> bool:
        return self.exchange_count == 0