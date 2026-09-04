from unittest.mock import MagicMock, patch

from graph_agent import app


def test_billing_ticket_routes_to_billing_branch():
    with patch("graph_agent.model") as mock_model:
        mock_model.invoke.side_effect = [
            MagicMock(content="billing"),
            MagicMock(content="We're sorry about the duplicate charge, a refund is on the way."),
        ]

        result = app.invoke({"ticket": "I was charged twice, please refund."})

    assert result["category"] == "billing"
    assert mock_model.invoke.call_count == 2
    assert "[BILLING TEAM]" in result["final_response"]
    assert "refund" in result["final_response"].lower()


def test_technical_ticket_routes_to_technical_branch():
    with patch("graph_agent.model") as mock_model:
        mock_model.invoke.side_effect = [
            MagicMock(content="technical"),
            MagicMock(content="Try clearing your cache and updating the app."),
        ]

        result = app.invoke({"ticket": "The app crashes every time I upload a file."})

    assert result["category"] == "technical"
    assert mock_model.invoke.call_count == 2
    assert "[TECHNICAL TEAM]" in result["final_response"]
    assert "cache" in result["final_response"].lower()


def test_unrecognized_category_falls_back_to_technical():
    with patch("graph_agent.model") as mock_model:
        mock_model.invoke.side_effect = [
            MagicMock(content="not a real category"),
            MagicMock(content="Some troubleshooting reply."),
        ]

        result = app.invoke({"ticket": "Something ambiguous."})

    assert result["category"] == "technical"
