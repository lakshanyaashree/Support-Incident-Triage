"""
retrieval.py

search_incidents() is unchanged from what you had. Added below it:
  - PRIORITY_TO_SEVERITY: Jira uses Highest/High/Medium/Low, your
    severities table uses critical/high/medium/low (see your own
    terminal output) -- this bridges the two vocabularies.
  - record_incident(): writes the agent's triage decision as a new row,
    embedding it immediately so it's searchable right away. Stored with
    status='open', not 'resolved' -- search_incidents() only pulls
    resolved/closed rows on purpose, so an unverified guess doesn't
    contaminate future retrieval. Promote it to 'resolved' later once a
    human actually confirms the fix worked.
  - incident_already_logged(): guards against the same ticket_key being
    processed twice (e.g. if Jira fires the webhook more than once for
    one action) so you don't get duplicate rows / duplicate Slack pings.
  - get_valid_teams(): returns the real team names from the DB so
    agent.py can constrain the model's assignee_team choice to values
    that actually exist, instead of letting it invent ones like
    "Product" that aren't in the teams table.
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import psycopg2

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=1536)
    )
    return response.embeddings[0].values


def get_connection():
    # Using DATABASE_URL (direct connection string) rather than separate
    # host/port/user/password vars — fewer env vars to keep in sync.
    # Get this from Supabase: Project Settings -> Database -> Connection string -> URI
    return psycopg2.connect(os.environ["DATABASE_URL"])


# Improvement #1 (confidence gating): a retrieved incident below this
# cosine-similarity score isn't treated as real precedent. agent.py uses
# this to decide whether it has "high" or "low" confidence context and
# branches the prompt accordingly, instead of always trusting top-5
# regardless of how weak the match actually is.
SIMILARITY_THRESHOLD = 0.75


def search_incidents(query: str, top_k: int = 5) -> list[dict]:
    """
    Returns a list of dicts, one per similar past incident, e.g.:
    {
        "ticket_number": "INC-1005",
        "title": "...",
        "description": "...",
        "resolution": "...",
        "team": "Security",
        "severity": "critical",
        "status": "resolved",
        "similarity": 0.83
    }
    """
    query_embedding = get_embedding(query)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            i.ticket_number,
            i.title,
            i.description,
            i.resolution,
            t.name AS team,
            s.name AS severity,
            st.name AS status,
            1 - (i.embedding <=> %s::vector) AS similarity
        FROM incidents i
        JOIN teams t ON i.team_id = t.id
        JOIN severities s ON i.severity_id = s.id
        JOIN statuses st ON i.status_id = st.id
        WHERE i.embedding IS NOT NULL
          AND LOWER(st.name) IN ('resolved', 'closed')
        ORDER BY i.embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, query_embedding, top_k)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        (ticket_number, title, description, resolution,
         team, severity, status, similarity) = row
        results.append({
            "ticket_number": ticket_number,
            "title": title,
            "description": description,
            "resolution": resolution,
            "team": team,
            "severity": severity,
            "status": status,
            "similarity": similarity,
        })

    return results


PRIORITY_TO_SEVERITY = {
    "Highest": "critical",
    "High": "high",
    "Medium": "medium",
    "Low": "low",
}

SEVERITY_TO_PRIORITY = {v: k for k, v in PRIORITY_TO_SEVERITY.items()}


def get_valid_teams() -> list[str]:
    """Returns the real team names from the teams table, so agent.py can
    constrain the model's assignee_team choice via a schema enum instead
    of letting it invent team names that don't exist in the DB (e.g. it
    previously returned "Product", which caused record_incident() to
    raise and the ticket to silently never get inserted)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM teams ORDER BY name;")
    teams = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return teams


def get_team_users(team_name: str) -> list[dict]:
    """Returns the users on a given team, for the sidebar's 'active
    users' list under an assigned team. Note: there's no real presence/
    online tracking in this schema (no last_seen or is_active column),
    so every listed user is shown with an 'active' dot purely as a
    visual stand-in -- swap this out if you ever add real presence
    tracking."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.name, u.email
        FROM users u
        JOIN teams t ON u.team_id = t.id
        WHERE LOWER(t.name) = LOWER(%s)
        ORDER BY u.name;
        """,
        (team_name,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"name": r[0], "email": r[1]} for r in rows]


def get_sidebar_incidents(open_limit: int = 3) -> dict:
    """Returns the data for the dashboard's side panel, pulled straight
    from the DB (not the in-memory TICKETS store), so it reflects the
    real incidents table:
      - open: the most recently created incidents whose status isn't
        resolved/closed, each with its team's users attached
      - recent_closed: the single most recently resolved/closed
        incident (by resolved_at)
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT i.ticket_number, i.title, t.name AS team,
               s.name AS severity, st.name AS status, i.created_at
        FROM incidents i
        JOIN teams t ON i.team_id = t.id
        JOIN severities s ON i.severity_id = s.id
        JOIN statuses st ON i.status_id = st.id
        WHERE LOWER(st.name) NOT IN ('resolved', 'closed')
        ORDER BY i.created_at DESC
        LIMIT %s;
        """,
        (open_limit,),
    )
    open_rows = cur.fetchall()

    cur.execute(
        """
        SELECT i.ticket_number, i.title, t.name AS team,
               s.name AS severity, st.name AS status, i.resolved_at
        FROM incidents i
        JOIN teams t ON i.team_id = t.id
        JOIN severities s ON i.severity_id = s.id
        JOIN statuses st ON i.status_id = st.id
        WHERE LOWER(st.name) IN ('resolved', 'closed')
        ORDER BY i.resolved_at DESC NULLS LAST
        LIMIT 1;
        """
    )
    closed_row = cur.fetchone()

    cur.close()
    conn.close()

    open_incidents = []
    for ticket_number, title, team, severity, status, created_at in open_rows:
        open_incidents.append({
            "ticket_number": ticket_number,
            "title": title,
            "team": team,
            "severity": severity,
            "status": status,
            "created_at": created_at.isoformat() if created_at else None,
            "team_users": get_team_users(team),
        })

    recent_closed = None
    if closed_row:
        ticket_number, title, team, severity, status, resolved_at = closed_row
        recent_closed = {
            "ticket_number": ticket_number,
            "title": title,
            "team": team,
            "severity": severity,
            "status": status,
            "resolved_at": resolved_at.isoformat() if resolved_at else None,
        }

    return {"open": open_incidents, "recent_closed": recent_closed}


def get_user_id_by_email(email: str | None) -> int | None:
    """Looks up a users.id by email, for mapping Jira's reporter/assignee
    onto reported_by / assigned_to. Returns None (not an error) if the
    email is missing or doesn't match anyone in the users table -- those
    columns are nullable for exactly this reason."""
    if not email:
        return None
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE LOWER(email) = LOWER(%s)", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


def get_least_loaded_user_for_team(team_name: str) -> dict | None:
    """Picks a real assignee from the team the agent chose, instead of
    leaving assigned_to NULL when Jira's own assignee field is empty
    (the normal case right after ticket creation, before anyone's been
    assigned in Jira itself). Load-balances by current open-incident
    count per user, ties broken alphabetically for determinism. Returns
    None if the team has no users yet."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.name, u.email,
               COUNT(i.id) FILTER (
                   WHERE st.name IS NOT NULL AND LOWER(st.name) NOT IN ('resolved', 'closed')
               ) AS open_count
        FROM users u
        JOIN teams t ON u.team_id = t.id
        LEFT JOIN incidents i ON i.assigned_to = u.id
        LEFT JOIN statuses st ON i.status_id = st.id
        WHERE LOWER(t.name) = LOWER(%s)
        GROUP BY u.id, u.name, u.email
        ORDER BY open_count ASC, u.name ASC
        LIMIT 1;
        """,
        (team_name,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "email": row[2], "open_count": row[3]}


def get_incident_summary(ticket_number: str) -> dict | None:
    """Used when a duplicate 'issue_created' webhook fires for a ticket
    that's already been triaged (Jira is known to fire the same event
    more than once) -- lets the caller resync the in-memory dashboard
    state from the DB instead of resetting priority/team back to None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.title, t.name AS team, s.name AS severity, st.name AS status
        FROM incidents i
        JOIN teams t ON i.team_id = t.id
        JOIN severities s ON i.severity_id = s.id
        JOIN statuses st ON i.status_id = st.id
        WHERE i.ticket_number = %s
        LIMIT 1;
        """,
        (ticket_number,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    title, team, severity, status = row
    return {
        "title": title,
        "team": team,
        "priority": SEVERITY_TO_PRIORITY.get(severity, severity),
        "status": status,
    }


def incident_already_logged(ticket_number: str) -> bool:
    """Idempotency guard -- call this before running the agent so a
    duplicate webhook fire for the same ticket doesn't create a second
    incidents row or send a second Slack message."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM incidents WHERE ticket_number = %s LIMIT 1", (ticket_number,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists


def update_incident_status(ticket_number: str, jira_status_name: str) -> bool:
    """
    Called when a Jira issue transitions (e.g. moved to the "resolved"
    swim lane). Maps the Jira status name onto a row in `statuses` --
    case/spacing-insensitive, so "Waiting on Customer" matches your
    waiting_on_customer row -- and updates the matching incidents row.
    Also stamps resolved_at when landing on resolved/closed, and clears
    it if the ticket is reopened.
    Returns False if the status name doesn't match anything in your
    statuses table, or the ticket isn't in incidents yet (never triaged).
    """
    normalized = jira_status_name.strip().lower().replace(" ", "_")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM statuses WHERE LOWER(name) = %s", (normalized,))
    status_row = cur.fetchone()
    if status_row is None:
        cur.close()
        conn.close()
        print(f"No matching status for Jira status '{jira_status_name}' (normalized: '{normalized}')")
        return False
    status_id = status_row[0]
    is_terminal = normalized in ("resolved", "closed")

    cur.execute(
        """
        UPDATE incidents
        SET status_id = %s,
            updated_at = NOW(),
            resolved_at = CASE WHEN %s THEN NOW() ELSE NULL END
        WHERE ticket_number = %s
        RETURNING id;
        """,
        (status_id, is_terminal, ticket_number),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return row is not None


def record_incident(
    ticket_number: str,
    title: str,
    description: str,
    suggested_resolution: str,
    team: str,
    priority: str,
    status: str = "open",
    reported_by_id: int | None = None,
    assigned_to_id: int | None = None,
) -> int:
    """
    Inserts the agent's triage output as a new incidents row with a
    NOW() timestamp, embedded immediately so the next search_incidents()
    call can find it -- once it's marked resolved.

    reported_by_id / assigned_to_id: already-resolved users.id values
    (the caller resolves Jira's reporter email and picks a real team
    member via get_least_loaded_user_for_team() when Jira's own
    assignee is empty -- see agent.py's execute_tool_calls). Left NULL
    if not resolvable.

    Returns the new row's id.
    """
    severity = PRIORITY_TO_SEVERITY.get(priority, priority.lower())
    embedding = get_embedding(f"{title}\n\n{description}")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id FROM teams WHERE LOWER(name) = LOWER(%s)", (team,))
    team_row = cur.fetchone()
    if team_row is None:
        cur.close()
        conn.close()
        raise ValueError(f"Team '{team}' not found -- add it to the teams table first")
    team_id = team_row[0]

    cur.execute("SELECT id FROM severities WHERE LOWER(name) = LOWER(%s)", (severity,))
    severity_row = cur.fetchone()
    if severity_row is None:
        cur.close()
        conn.close()
        raise ValueError(f"Severity '{severity}' not found -- add it to the severities table first")
    severity_id = severity_row[0]

    cur.execute("SELECT id FROM statuses WHERE LOWER(name) = LOWER(%s)", (status,))
    status_row = cur.fetchone()
    if status_row is None:
        cur.close()
        conn.close()
        raise ValueError(f"Status '{status}' not found -- add it to the statuses table first")
    status_id = status_row[0]

    cur.execute(
        """
        INSERT INTO incidents
            (ticket_number, title, description, resolution,
             team_id, severity_id, status_id, reported_by, assigned_to,
             embedding, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id;
        """,
        (ticket_number, title, description, suggested_resolution,
         team_id, severity_id, status_id, reported_by_id, assigned_to_id,
         embedding),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id


def record_decision(
    ticket_number: str,
    iteration_count: int,
    retrieved_context: list,
    top_similarity: float,
    confidence: str,
    self_check_passed: bool,
    self_check_notes: str,
    decision: dict,
    prompt_injection_flag: bool = False,
) -> int:
    """Improvement #7 (evaluation/observability): logs one full triage
    run -- what was retrieved, how confident the agent was, whether its
    own self-check passed, and the final decision -- so you have an
    audit trail to sample and label later rather than just trusting the
    agent's output blindly."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agent_decisions
            (ticket_number, iteration_count, retrieved_context, top_similarity,
             confidence, self_check_passed, self_check_notes, decision,
             prompt_injection_flag)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (
            ticket_number,
            iteration_count,
            json.dumps(retrieved_context, default=str),
            top_similarity,
            confidence,
            self_check_passed,
            self_check_notes,
            json.dumps(decision, default=str),
            prompt_injection_flag,
        ),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id


def get_recent_decisions(limit: int = 20, unlabeled_only: bool = False) -> list[dict]:
    """Powers the dashboard's eval panel -- a reviewer scans recent
    decisions and marks each 'correct' / 'incorrect', building up a
    labeled eval set over time."""
    conn = get_connection()
    cur = conn.cursor()
    where = "WHERE human_label IS NULL" if unlabeled_only else ""
    cur.execute(
        f"""
        SELECT id, ticket_number, iteration_count, top_similarity, confidence,
               self_check_passed, self_check_notes, decision,
               prompt_injection_flag, created_at, human_label, human_label_notes
        FROM agent_decisions
        {where}
        ORDER BY created_at DESC
        LIMIT %s;
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        (id_, ticket_number, iteration_count, top_similarity, confidence,
         self_check_passed, self_check_notes, decision, prompt_injection_flag,
         created_at, human_label, human_label_notes) = row
        results.append({
            "id": id_,
            "ticket_number": ticket_number,
            "iteration_count": iteration_count,
            "top_similarity": top_similarity,
            "confidence": confidence,
            "self_check_passed": self_check_passed,
            "self_check_notes": self_check_notes,
            "decision": decision,
            "prompt_injection_flag": prompt_injection_flag,
            "created_at": created_at.isoformat() if created_at else None,
            "human_label": human_label,
            "human_label_notes": human_label_notes,
        })
    return results


def label_decision(decision_id: int, label: str, notes: str | None = None) -> bool:
    """Reviewer marks a logged decision as 'correct' or 'incorrect' from
    the dashboard, building the eval set improvement #7 is for."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE agent_decisions
        SET human_label = %s, human_label_notes = %s, labeled_at = NOW()
        WHERE id = %s
        RETURNING id;
        """,
        (label, notes, decision_id),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return row is not None


# Standalone test — only runs when you execute this file directly,
# not when agent.py imports search_incidents() from it.
if __name__ == "__main__":
    query = input("Enter the support incident to search for:\n> ")
    results = search_incidents(query)

    print(f"\nFound {len(results)} similar incidents:\n")
    for i, r in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result #{i}")
        print(f"Ticket:     {r['ticket_number']}")
        print(f"Title:      {r['title']}")
        print(f"Team:       {r['team']}")
        print(f"Severity:   {r['severity']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"\nResolution:\n{r['resolution']}\n")