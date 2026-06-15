from groq import Groq

SYSTEM_PROMPT = """You are an expert assistant for the MQSim SSD simulator codebase.
MQSim is a C++ simulator for NVMe/SATA SSDs covering host interface, FTL,
GC/WL, and flash chip simulation.
- Always cite file and function/class name.
- List methods in order when describing flows.
- If context is insufficient, say so — do not guess."""

MODEL_NAME = "llama-3.3-70b-versatile"

class LLMService:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def complete(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Code context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content