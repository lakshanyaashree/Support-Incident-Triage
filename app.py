"""
app.py
The Jira webhook endpoint, a live log feed, a live tickets feed, a
DB-backed sidebar, and now (#7) an eval panel backed by agent_decisions,
all served from the same FastAPI app.

Run:
    python app.py
    ngrok http 8000        (in a separate terminal, unchanged)

Then open http://localhost:8000 in a browser to watch it work.

Before running: apply migrations/002_agent_decisions.sql in Supabase's
SQL editor if you haven't already -- agent.py's record_decision() call
will fail without that table.
"""

import json
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from agent import run_triage_agent, execute_tool_calls
from retrieval import (
    incident_already_logged,
    update_incident_status,
    search_incidents,
    get_sidebar_incidents,
    get_recent_decisions,
    label_decision,
    get_incident_summary,
)

# In-memory ring buffer the dashboard polls. Fine for a single-instance
# demo app; if this ever needed to survive a restart or run behind
# multiple workers, it'd need to move to the DB or Redis instead.
LOGS: list[dict] = []

# In-memory latest-state-per-ticket store, keyed by ticket_key, that
# backs the "Live tickets" panel. Same caveat as LOGS above re: restarts
# / multiple workers.
TICKETS: dict[str, dict] = {}

# Ticket keys currently mid-triage in this process. A DB check alone
# (incident_already_logged) isn't enough to catch a genuine duplicate
# webhook fire that arrives while the first run is still in flight --
# the agent's Gemini calls can take a few seconds, and the incidents
# row isn't inserted until execute_tool_calls() finishes. Without this,
# a fast duplicate fire sees incident_already_logged() == False, starts
# its own full (wasteful, possibly conflicting) triage run in parallel.
PROCESSING: set[str] = set()


def log(message: str, level: str = "info") -> None:
    """Writes to the terminal (same as before) AND to the buffer the
    dashboard reads, so the terminal and the browser always agree."""
    print(message)
    LOGS.append({"time": datetime.now().strftime("%H:%M:%S"), "level": level, "message": message})
    del LOGS[:-200]  # keep the last 200 entries so this can't grow forever


def upsert_ticket(ticket_key: str, **fields) -> None:
    """Creates or updates the in-memory record for a ticket. Only the
    fields passed in are overwritten, so e.g. a later status-only update
    doesn't wipe out the title/priority/team set earlier."""
    t = TICKETS.setdefault(ticket_key, {"ticket_key": ticket_key})
    t.update(fields)
    t["updated_at"] = datetime.now().strftime("%H:%M:%S")


def extract_text_from_adf(desc):
    if not desc:
        return ""
    if isinstance(desc, str):
        return desc

    text_parts = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "text":
                text_parts.append(node.get("text", ""))
            for child in node.get("content", []):
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(desc)
    return " ".join(text_parts)


def extract_status_change(payload: dict) -> str | None:
    changelog = payload.get("changelog", {})
    for item in changelog.get("items", []):
        if item.get("field") == "status":
            return item.get("toString")
    return None


def extract_people(fields: dict) -> tuple[str | None, str | None]:
    """Pulls the reporter's and assignee's email out of Jira's payload,
    for reported_by / assigned_to in the incidents table. Either can be
    missing (assignee often is, at creation time; reporter's email can
    be hidden by Jira's privacy settings) -- record_incident() treats a
    None email as 'no matching user' and leaves the column NULL."""
    reporter = fields.get("reporter") or {}
    assignee = fields.get("assignee") or {}
    return reporter.get("emailAddress"), assignee.get("emailAddress")


app = FastAPI()

# Serves everything in ./static, including dashboard.html, at /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def dashboard():
    return FileResponse("static/dashboard.html")


@app.get("/logs")
async def get_logs():
    return LOGS


@app.get("/tickets")
async def get_tickets():
    # Most recently updated first
    return sorted(TICKETS.values(), key=lambda t: t["updated_at"], reverse=True)


@app.get("/search")
async def search(query: str):
    return search_incidents(query)


@app.get("/sidebar")
async def sidebar():
    """Backs the dashboard's side panel: 3 open incidents + the most
    recently closed one, straight from the incidents table (not the
    in-memory TICKETS store), each open incident carrying its team's
    users for the 'active users' list under it."""
    return get_sidebar_incidents(open_limit=3)


@app.get("/decisions")
async def decisions(limit: int = 20, unlabeled_only: bool = False):
    """#7: the eval panel's data source -- recent agent decisions with
    their retrieval confidence, self-check result, and any existing
    human label."""
    return get_recent_decisions(limit=limit, unlabeled_only=unlabeled_only)


@app.post("/decisions/{decision_id}/label")
async def label(decision_id: int, request: Request):
    """#7: a human reviewer marks a decision 'correct' or 'incorrect'
    from the dashboard, building a labeled eval set over time."""
    body = await request.json()
    result_label = body.get("label")
    notes = body.get("notes")
    if result_label not in ("correct", "incorrect"):
        return JSONResponse(status_code=400, content={"error": "label must be 'correct' or 'incorrect'"})
    ok = label_decision(decision_id, result_label, notes)
    return {"status": "ok" if ok else "not_found"}


@app.post("/jira-webhook")
async def jira_webhook(request: Request):
    try:
        payload = await request.json()
        event = payload.get("webhookEvent")
        issue = payload.get("issue", {})
        ticket_key = issue.get("key")

        if event == "jira:issue_created":
            fields = issue.get("fields", {})
            title = fields.get("summary", "")
            description = extract_text_from_adf(fields.get("description", ""))
            reporter_email, assignee_email = extract_people(fields)

            # Bug fix: Jira is known to fire "issue_created" more than
            # once for the same ticket (e.g. create + an immediate
            # follow-up edit can both surface as create-shaped events).
            # Previously we unconditionally reset priority/team to None
            # here before checking incident_already_logged(), so a
            # duplicate fire would wipe out the first run's real
            # decision from the dashboard even though the DB still had
            # it. Now: check for a duplicate FIRST, and if it is one,
            # resync the dashboard from the DB instead of blanking it.
            if incident_already_logged(ticket_key):
                log(f"Skipping {ticket_key} -- already processed (duplicate webhook)", "info")
                existing = get_incident_summary(ticket_key)
                if existing:
                    upsert_ticket(
                        ticket_key,
                        title=existing["title"],
                        status=existing["status"],
                        priority=existing["priority"],
                        team=existing["team"],
                    )
                return {"status": "already_processed"}

            if ticket_key in PROCESSING:
                # Genuine race: an earlier duplicate fire for this same
                # ticket is still mid-triage in this process (its
                # incidents row hasn't been committed yet, so the DB
                # check above didn't catch it). Don't start a second
                # parallel run -- just wait this one out.
                log(f"Skipping {ticket_key} -- already in progress in this process", "info")
                return {"status": "already_in_progress"}

            PROCESSING.add(ticket_key)
            try:
                upsert_ticket(ticket_key, title=title, status="To Do", priority=None, team=None, reasoning=None)
                log(f"Webhook: issue created {ticket_key} -- {title}", "webhook")

                # #3 ReAct loop lives inside run_triage_agent now: it may
                # call search_incidents itself one or more times before
                # producing a final decision (or none, if it can't ground one).
                agent_result = run_triage_agent(ticket_key, title, description)
                log(
                    f"Agent result ({agent_result['iteration_count']} iteration(s), "
                    f"confidence={agent_result['confidence']}, "
                    f"self_check_passed={agent_result['self_check_passed']}): "
                    f"{json.dumps(agent_result['decision'])}",
                    "agent",
                )
                if agent_result["prompt_injection_flag"]:
                    log(f"NOTE: {ticket_key} description flagged by injection heuristic", "error")

                decision = agent_result["decision"]
                if decision:
                    upsert_ticket(
                        ticket_key,
                        priority=decision.get("priority"),
                        team=decision.get("assignee_team"),
                        reasoning=decision.get("reasoning"),
                    )
                else:
                    upsert_ticket(ticket_key, status="Needs human triage")

                execution_results = execute_tool_calls(
                    ticket_key, title, description, agent_result,
                    reported_by_email=reporter_email,
                    assigned_to_email=assignee_email,
                )
                log(f"Execution results: {json.dumps(execution_results)}", "agent")
            finally:
                PROCESSING.discard(ticket_key)

            return {
                "status": "processed",
                "agent_result": {k: v for k, v in agent_result.items() if k != "retrieved_context"},
                "execution_results": execution_results,
            }

        elif event == "jira:issue_updated":
            new_status = extract_status_change(payload)
            if new_status is None:
                return {"status": "ignored", "reason": "no status change"}

            upsert_ticket(ticket_key, status=new_status)

            updated = update_incident_status(ticket_key, new_status)
            log(f"Status sync: {ticket_key} -> '{new_status}' (db updated: {updated})", "webhook")
            return {
                "status": "status_synced" if updated else "ticket_not_found_or_unmapped_status",
                "new_status": new_status,
            }

        return {"status": "ignored"}

    except Exception as e:
        log(f"ERROR handling webhook: {e}", "error")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)