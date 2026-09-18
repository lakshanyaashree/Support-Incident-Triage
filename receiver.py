import json

from fastapi import FastAPI, Request
import uvicorn

from agent import run_triage_agent, execute_tool_calls
from retrieval import incident_already_logged, update_incident_status


def extract_text_from_adf(desc):
    if not desc:
        return ""
    if isinstance(desc, str):
        return desc  # already plain text

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
    """Returns the new status name if this issue_updated event includes
    a status transition, else None (e.g. someone just edited the
    description or added a comment -- not our concern)."""
    changelog = payload.get("changelog", {})
    for item in changelog.get("items", []):
        if item.get("field") == "status":
            return item.get("toString")
    return None


app = FastAPI()


@app.post("/jira-webhook")
async def jira_webhook(request: Request):
    payload = await request.json()
    event = payload.get("webhookEvent")
    issue = payload.get("issue", {})
    ticket_key = issue.get("key")

    if event == "jira:issue_created":
        fields = issue.get("fields", {})
        title = fields.get("summary", "")
        description = extract_text_from_adf(fields.get("description", ""))

        # Guards against the same webhook firing more than once for one
        # ticket (you saw 3 POSTs for a single swim-lane move) -- without
        # this, a duplicate fire would create a second incidents row.
        if incident_already_logged(ticket_key):
            print(f"Skipping {ticket_key} -- already processed")
            return {"status": "already_processed"}

        tool_calls = run_triage_agent(ticket_key, title, description)
        print("AGENT DECIDED:", json.dumps(tool_calls, indent=2))

        execution_results = execute_tool_calls(ticket_key, title, description, tool_calls)
        print("EXECUTION RESULTS:", json.dumps(execution_results, indent=2))

        return {
            "status": "processed",
            "tool_calls": tool_calls,
            "execution_results": execution_results,
        }

    elif event == "jira:issue_updated":
        new_status = extract_status_change(payload)
        if new_status is None:
            return {"status": "ignored", "reason": "no status change"}

        updated = update_incident_status(ticket_key, new_status)
        print(f"STATUS SYNC: {ticket_key} -> '{new_status}' (db row updated: {updated})")
        return {
            "status": "status_synced" if updated else "ticket_not_found_or_unmapped_status",
            "new_status": new_status,
        }

    return {"status": "ignored"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)