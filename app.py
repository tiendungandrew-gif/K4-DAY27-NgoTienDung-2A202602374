"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", layout="wide", page_icon="🤖")
st.title("🤖 HITL PR Review Agent")
st.caption("Human-in-the-Loop Pull Request Review powered by LangGraph & SQLite Audit Trail")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Recent sessions")

    async def get_recent_threads():
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MAX(risk_level)       AS worst_risk,
                       MAX(timestamp)        AS last_event,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 15
                """
            ) as cur:
                return await cur.fetchall()

    try:
        threads = asyncio.run(get_recent_threads())
        if threads:
            for row in threads:
                tid = row["thread_id"]
                short_tid = tid[:8]
                risk = row["worst_risk"]
                pr_num = row["pr_url"].split("/")[-1] if "/" in row["pr_url"] else "PR"
                label = f"PR #{pr_num} [{risk.upper()}] · {short_tid}"
                if st.button(label, key=f"session_{tid}", use_container_width=True):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = row["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
        else:
            st.caption("No audit sessions yet.")
    except Exception as e:
        st.caption(f"Could not load sessions: {e}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "GitHub Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("🚀 Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"⚠️ Approval requested — confidence {conf:.0%}")
    st.info(payload.get("confidence_reasoning", ""))
    st.markdown(payload.get("summary", ""))

    st.markdown("#### Proposed Comments")
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity'].upper()}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("🔍 View Diff Preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Reviewer feedback / note (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🛑 Strong escalation — confidence {conf:.0%}")
    st.warning(payload.get("confidence_reasoning", ""))
    if payload.get("risk_factors"):
        st.error("🚨 **Identified Risks:** " + ", ".join(payload["risk_factors"]))
    st.markdown(payload.get("summary", ""))

    st.markdown("#### Clarifying Questions from AI Reviewer")
    with st.form("escalation_form"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(f"Q{i+1}: {q}", key=f"q_{i}")
        submitted_answers = st.form_submit_button("Submit answers & Synthesize review", type="primary")
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR & analyzing with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result
    st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming graph execution..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"🎉 **{action.upper()}** — Review comment has been posted to GitHub!")
        st.link_button("🌐 View comment on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("✋ **REJECTED** — Review was rejected. No comment was posted.")
    else:
        st.info(f"Final Action: **{action}**")

    if "analysis" in final:
        a = final["analysis"]
        st.markdown(f"**Final Confidence:** `{a.confidence:.0%}`")
        st.markdown(f"**Summary:** {a.summary}")

    st.caption(f"Session Thread ID: `{st.session_state.thread_id}`  ·  Replay via CLI: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")

# If an existing session from sidebar was selected, display its audit timeline
elif st.session_state.thread_id and payload is None and st.session_state.final is None:
    async def get_session_events(tid: str):
        return await replay_events(tid)

    try:
        evs = asyncio.run(get_session_events(st.session_state.thread_id))
        if evs:
            st.info(f"Viewing history for Session `{st.session_state.thread_id}` ({st.session_state.pr_url})")
            st.dataframe(
                [
                    {
                        "Timestamp": e["timestamp"],
                        "Action": e["action"],
                        "Confidence": f"{e['confidence']:.0%}",
                        "Risk": e["risk_level"].upper(),
                        "Decision": e["decision"],
                        "Reviewer": e["reviewer_id"] or "-",
                        "Duration (ms)": e["execution_time_ms"],
                        "Reason": e["reason"] or "",
                    }
                    for e in evs
                ],
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Error loading session history: {e}")
