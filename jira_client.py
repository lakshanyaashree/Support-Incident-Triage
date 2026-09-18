"""
jira_client.py
Executes the Jira-side effects of a triage decision: sets priority + a
team label on the ticket, then adds a comment with the reasoning and
suggested resolution.

Needs in .env:
    JIRA_BASE_URL=https://yourdomain.atlassian.net
    JIRA_EMAIL=you@yourcompany.com
    JIRA_API_TOKEN=...   (Jira Cloud API token -- id.atlassian.com/manage-profile/security/api-tokens,
                           NOT your account password)

The API token's account needs "Edit issues" + "Add comments" permission
on the project the webhook fires for.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

JIRA_BASE_URL = os.environ["JIRA_BASE_URL"].rstrip("/")
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]

AUTH = (JIRA_EMAIL, JIRA_API_TOKEN)
HEADERS = {"Content-Type": "application/json"}


def update_jira_ticket(
    ticket_key: str,
    priority: str,
    assignee_team: str,
    reasoning: str,
    suggested_resolution: str,
    needs_human_review: bool = False,
) -> None:
    """
    Sets priority + a team label, then posts a comment.

    Note: real Jira "assignee" needs a person's accountId, not a team
    name -- there's no built-in "assign to team" concept. A label is
    the pragmatic stand-in here. If your project has a custom "Team"
    field instead, swap the "labels" line below for e.g.
    "customfield_10050": {"value": assignee_team}.

    needs_human_review (agent.py improvement #1/#6): when the agent's
    confidence was low, or a "Highest" priority got auto-downgraded for
    lacking real precedent, this adds a "needs-review" label and a
    visible note in the comment -- so a human sees it was a soft call,
    not something the agent quietly decided on its own.
    """
    labels = [assignee_team.replace(" ", "-").lower()]
    if needs_human_review:
        labels.append("needs-review")

    fields_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}"
    fields_payload = {
        "fields": {
            "priority": {"name": priority},
            "labels": labels,
        }
    }
    resp = requests.put(fields_url, json=fields_payload, auth=AUTH, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    comment_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}/comment"
    paragraphs = [
        f"\U0001f916 Auto-triaged",
        f"Reasoning: {reasoning}",
        f"Suggested resolution: {suggested_resolution}",
    ]
    if needs_human_review:
        paragraphs.append(
            "\u26a0\ufe0f Low confidence or auto-downgraded priority -- please have a human confirm this triage."
        )

    comment_payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": p}]}
                for p in paragraphs
            ],
        }
    }
    resp2 = requests.post(comment_url, json=comment_payload, auth=AUTH, headers=HEADERS, timeout=10)
    resp2.raise_for_status()


def transition_issue(ticket_key: str, target_status_name: str) -> bool:
    """
    Moves a ticket to a workflow status (e.g. 'In Progress') via Jira's
    Transitions API. Priority/labels are plain fields settable with a
    direct PUT (see update_jira_ticket above), but workflow status isn't
    -- it only changes through a transition ID that's valid from the
    issue's *current* state, so this looks up the issue's available
    transitions first and matches one by name (case-insensitive).

    Returns True if the transition was applied, False (without raising)
    if no transition to that status exists from wherever the ticket
    currently is -- e.g. your board's workflow doesn't allow a direct
    To Do -> In Progress jump. Callers should treat False as
    "best-effort, didn't happen" rather than a hard failure.
    """
    list_url = f"{JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}/transitions"
    resp = requests.get(list_url, auth=AUTH, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    transitions = resp.json().get("transitions", [])

    match = next(
        (t for t in transitions if t["name"].strip().lower() == target_status_name.strip().lower()),
        None,
    )
    if match is None:
        available = [t["name"] for t in transitions]
        print(f"No transition to '{target_status_name}' available for {ticket_key}. Available: {available}")
        return False

    resp2 = requests.post(list_url, json={"transition": {"id": match["id"]}}, auth=AUTH, headers=HEADERS, timeout=10)
    resp2.raise_for_status()
    return True