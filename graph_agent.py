from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

load_dotenv()

model = init_chat_model("claude-sonnet-4-6", temperature=0.3)


class State(TypedDict):
    ticket: str
    category: Literal["billing", "technical", "unclear"]
    draft_response: str
    final_response: str


def classify_node(state: State) -> dict:
    """LLM-driven: decide which specialist should handle this ticket."""
    prompt = f"""Classify this support ticket as exactly one word: "billing", "technical", or "unclear".

Use "unclear" if the message is not actually a support request, or if it lacks enough detail to
tell whether it's a billing or technical issue. Do not guess a category just to pick one.

Ticket: {state['ticket']}

Respond with only the single word."""
    result = model.invoke(prompt)
    category = result.content.strip().lower()
    if category not in ("billing", "technical", "unclear"):
        category = "unclear"
    return {"category": category}


def route_by_category(state: State) -> str:
    """Conditional edge: reads state, returns the name of the next node."""
    return state["category"]


def billing_node(state: State) -> dict:
    """LLM-driven: billing-specialist reply."""
    prompt = f"""You are a billing support specialist. Write a short, empathetic reply to this ticket:

{state['ticket']}"""
    result = model.invoke(prompt)
    return {"draft_response": result.content}


def technical_node(state: State) -> dict:
    """LLM-driven: technical-specialist reply."""
    prompt = f"""You are a technical support specialist. Write a short, precise troubleshooting reply to this ticket:

{state['ticket']}"""
    result = model.invoke(prompt)
    return {"draft_response": result.content}


def unclear_node(state: State) -> dict:
    """Deterministic: no model call, always the same clarification request."""
    reply = (
        "Thanks for reaching out. We couldn't tell from your message whether this is a "
        "billing or technical issue, or whether it's a support request at all. Could you "
        "share more detail about what you need help with?"
    )
    return {"draft_response": reply}


def format_response_node(state: State) -> dict:
    """Deterministic: no model call, just formats the final output."""
    final = f"[{state['category'].upper()} TEAM]\n\n{state['draft_response']}\n\n-- Support Team"
    return {"final_response": final}


graph = StateGraph(State)

graph.add_node("classify", classify_node)
graph.add_node("billing", billing_node)
graph.add_node("technical", technical_node)
graph.add_node("unclear", unclear_node)
graph.add_node("format_response", format_response_node)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route_by_category,
    {"billing": "billing", "technical": "technical", "unclear": "unclear"},
)
graph.add_edge("billing", "format_response")
graph.add_edge("technical", "format_response")
graph.add_edge("unclear", "format_response")
graph.add_edge("format_response", END)

app = graph.compile()


if __name__ == "__main__":
    tickets = [
        "I was charged twice for my subscription this month, can you refund the extra charge?",
        "The app crashes every time I try to upload a file larger than 10MB.",
        "Today is Monday"
    ]

    for ticket in tickets:
        result = app.invoke({"ticket": ticket})
        print(f"Ticket: {ticket}")
        print(f"Category: {result['category']}")
        print(result["final_response"])
        print("---")
