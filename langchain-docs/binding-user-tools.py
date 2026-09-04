from langchain.tools import tool
from langchain.chat_models import init_chat_model


from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model='claude-sonnet-4-5',
    )

@tool
def get_temperature(location: str) -> str:
    """Get the temperature at a location."""
    return f"It's 95 degress in {location}."

@tool
def greet(name: str) -> str:
    """general greeting"""
    return "Hello " + name

model_with_tools = model.bind_tools([get_temperature, greet], tool_choice="any")

# response = model_with_tools.invoke("Hi My name is Varun")
# print(response.content)
# for tool_call in response.tool_calls:
#     # View tool calls made by the model
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")

# response = model_with_tools.invoke("Is it raining in Gomtinagar Lucknow ?")
# print(response.content)
# for tool_call in response.tool_calls:
#     # View tool calls made by the model
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")


response = model_with_tools.invoke("Whats the weather in Boston ?")
print(response)
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")