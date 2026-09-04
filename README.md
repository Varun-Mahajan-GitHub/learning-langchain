# learning-langchain

Small LangChain / LangGraph examples, building up from a single tool-calling agent to a
multi-node LangGraph agent.

## graph_agent.py — support ticket triage graph

A LangGraph `StateGraph` that classifies a support ticket and routes it to a specialist branch,
demonstrating how deterministic and LLM-driven steps mix in a single graph.

**Nodes:**

| Node | Type | What it does |
|---|---|---|
| `classify` | LLM | Reads the ticket and decides `billing` or `technical` |
| `billing` | LLM | Drafts a billing-specialist reply |
| `technical` | LLM | Drafts a technical-specialist reply |
| `format_response` | deterministic (no model call) | Wraps the draft into the final formatted reply |

**Shape of the graph:**

```
START -> classify --(conditional edge on category)--> billing ------> format_response -> END
                                                 \--> technical --/
```

`classify` is followed by a **conditional edge** (`route_by_category`) that reads
`state["category"]` and picks which branch runs next. Both branches converge back into the same
`format_response` node before the graph ends — showing a node doesn't have to be an "agent," it
can just be plain code, and multiple paths can merge into one downstream node.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

```
python graph_agent.py
```

### Sample run

```
Ticket: I was charged twice for my subscription this month, can you refund the extra charge?
Category: billing
[BILLING TEAM]

Hi there,

Thank you for reaching out, and I'm so sorry for the inconvenience this has caused! I've
flagged your account for review and will get this sorted out for you right away. Once
confirmed, we'll process a full refund for the duplicate charge.

Warm regards,
Billing Support Team

-- Support Team
---
Ticket: The app crashes every time I try to upload a file larger than 10MB.
Category: technical
[TECHNICAL TEAM]

Hi,

Thanks for reporting this. Try clearing the app cache, confirming you're on the latest
version, and testing on a different network. If it persists, send us your device model,
OS version, and app version so we can investigate further.

-- Support Team
---
```

## Tests

```
pytest
```

`test_graph_agent.py` mocks the model (`graph_agent.model`) and calls `app.invoke(...)` directly,
so it exercises the real compiled graph — classification, conditional routing, and formatting —
without hitting the Claude API or needing an API key.

## Other files

- `basic-agent.py` — a minimal single-tool `create_agent` example.
- `langchain-agent.py` — compares `create_agent` vs. `create_deep_agent` on the same task (fetching
  and analyzing a text file), showing how a deep agent's planning/filesystem/subagent tools let it
  recover from a failure the plain agent can't.
- `langchain-docs/` — small scratch scripts from reading the LangChain docs (models, tool binding).
