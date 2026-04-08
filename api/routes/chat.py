"""
api/routes/chat.py

/api/chat  — POST: send a message, receive the agent's response.
/api/chat/<thread_id>  — DELETE: clear session history from Supabase.
"""

import logging
import uuid

from flask import Blueprint, jsonify, request

from agent.interface import IATAAgent

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)

# Single agent instance shared across requests (graph is stateless; memory is in Supabase)
_agent = IATAAgent()


@chat_bp.post("/chat")
def chat():
    """
    Process a user message through the multi-agent graph.

    Request body (JSON):
        {
            "message":   "Find papers on efficient transformers",
            "thread_id": "abc-123"   // optional — new UUID created if absent
        }

    Response (JSON):
        {
            "reply":           "Here are the papers I found...",
            "thread_id":       "abc-123",
            "tool_calls_used": ["search_papers"]
        }
    """
    body = request.get_json(silent=True) or {}
    user_message: str = (body.get("message") or "").strip()
    thread_id: str = (body.get("thread_id") or str(uuid.uuid4())).strip()

    if not user_message:
        return jsonify({"error": "The 'message' field is required."}), 400

    logger.info("Chat request | thread=%s | message=%.80s…", thread_id, user_message)

    result = _agent.invoke(user_message, thread_id=thread_id)

    return jsonify({
        "reply":           result["reply"],
        "thread_id":       result["thread_id"],
        "tool_calls_used": result["tool_calls_used"],
    })


@chat_bp.delete("/chat/<thread_id>")
def clear_session(thread_id: str):
    """
    Delete all conversation history for a thread from Supabase.
    Called by the frontend when the user clicks 'New chat'.
    """
    try:
        from supabase import create_client
        from config.settings import settings

        client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        client.table(settings.supabase_conversations_table).delete().eq("thread_id", thread_id).execute()
        logger.info("Cleared session | thread=%s", thread_id)
        return jsonify({"status": "cleared", "thread_id": thread_id})
    except Exception as exc:
        logger.warning("Failed to clear session %s: %s", thread_id, exc)
        return jsonify({"status": "error", "detail": str(exc)}), 500
