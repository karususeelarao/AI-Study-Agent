from services.llm_service import LLMService

llm = LLMService()

response = llm.generate(
    "You are a helpful assistant.",
    "Say hello in one sentence."
)

print(response)