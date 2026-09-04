from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model('claude-sonnet-4-5')

# response = model.invoke("Why do parrots talk?")
# print(response)


# for response in model.batch_as_completed([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ]):
#     print(response)

for response in model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)