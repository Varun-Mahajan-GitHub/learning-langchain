from langchain.agents import create_agent

from dotenv import load_dotenv

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# negative case
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Antartica?"}]}
)

# print(result)
print(result["messages"][-1].content_blocks)