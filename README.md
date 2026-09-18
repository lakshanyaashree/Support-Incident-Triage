# Incident Triage Agent

An agentic incident-triage system that listens for new Jira tickets, retrieves similar past incidents from a vector database, and uses an LLM in a **ReAct (Reason → Act → Observe) loop** to decide severity, ownership, and a grounded resolution — then executes that decision back into Jira and logs it for evaluation.

Built to go beyond a single-shot "retrieve once, prompt once" RAG pipeline: the agent can call its own retrieval tool mid-reasoning, verifies its own output with a second LLM pass before acting, and every decision is logged for human review.

---

## What it does

1. A support ticket is created in Jira.
2. Jira fires a webhook to a FastAPI endpoint.
3. The ticket's title + description are embedded and matched against a `pgvector` (Supabase/Postgres) index of past **resolved** incidents.
4. An LLM (Gemini) reasons over the retrieved context inside a **ReAct loop**: it can call `search_incidents` again with a refined query if the initial context is weak, or commit to a decision via `update_jira`.
5. A **self-check critic** (a second LLM call) verifies the decision actually cites a real retrieved incident before it's trusted.
6. **Hard safety rules** downgrade any ungrounded high-priority call and flag it for human review — the model can't talk its way around this one.
7. The decision executes: Jira gets its priority/labels/comment updated and (best-effort) transitioned to "In Progress"; a new row is written to the incidents table so the knowledge base grows; the run is load-balanced to a real team member.
8. Every run — retrieved context, confidence, self-check result, final decision — is logged to an `agent_decisions` table for later labeling and evaluation.
9. A live dashboard shows open tickets, a DB-backed sidebar, and an eval panel for labeling agent decisions "correct"/"incorrect."

---

## Architecture

```
Jira ticket ──▶ Webhook ──▶ FastAPI receiver (dedup + lock)
                                   │
                                   ▼
                          RAG retrieval (pgvector)
                          confidence gate vs. threshold
                                   │
                                   ▼
                    ┌──────── ReAct agent loop ────────┐
                    │  Gemini + tools:                 │
                    │   - search_incidents (re-query)  │
                    │   - update_jira (final decision) │
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                          Self-check critic
                       (verifies cited precedent,
                        regenerates once on failure)
                                   │
                                   ▼
                          Safety guardrails
                    (downgrade ungrounded priority,
                     flag needs_human_review)
                                   │
                                   ▼
                          Execute + log
                 Jira PUT + transition · incidents INSERT
                       agent_decisions INSERT (eval)
```

---

## Why this is agentic, not just "an LLM call with a prompt"

| Concern | How it's handled |
|---|---|
| **Confidence gating** | A cosine-similarity threshold decides "high" vs "low" confidence *before* generation, and the prompt is branched accordingly instead of always trusting top-5 blindly. |
| **Hallucination mitigation** | The tool schema requires a `source_ticket_number` citation for any suggested resolution. A second LLM call acts as a critic, checking the citation is real and the resolution actually follows from it. |
| **ReAct loop** | `search_incidents` is a callable tool, not a pre-fetched context block — the model can request a more targeted search before committing, up to a capped number of iterations. |
| **Retry / regeneration** | API calls retry with backoff on failure. If the self-check fails, the agent regenerates once with an explicit correction note before falling back to a flagged, ungrounded state. |
| **Few-shot over fine-tuning** | Two worked examples are baked into the prompt instead of fine-tuning on a small, fast-changing knowledge base — cheaper, no training pipeline, and easy to justify (retrieval > weights for data that changes daily). |
| **Prompt injection defense** | Untrusted ticket text is wrapped in explicit delimiters with an instruction to treat it as data, not commands. A keyword heuristic flags suspicious phrasing for review. A hard post-generation rule downgrades any "Highest" priority lacking real retrieved precedent, regardless of the model's self-reported confidence. |
| **Evaluation / observability** | Every run is logged to `agent_decisions` with its retrieval confidence, self-check result, and final decision. A dashboard panel lets a human label runs "correct"/"incorrect," building a labeled eval set over time. |

---

## Tech stack

- **FastAPI** — webhook receiver + dashboard API
- **Gemini API** (`google-genai`) — embeddings + tool-calling LLM
- **Supabase / Postgres + pgvector** — incident knowledge base and vector similarity search
- **Jira Cloud REST API** — reading tickets via webhook, writing priority/labels/comments/transitions
- **HTML/CSS/vanilla JS** — polling-based live dashboard (no frontend framework)

---

## Project structure

```
.
├── app.py                       # FastAPI app: webhook, dashboard endpoints
├── agent.py                     # ReAct loop, self-check, guardrails, tool execution
├── retrieval.py                 # Embeddings, vector search, all DB access
├── jira_client.py               # Jira REST API: field updates, comments, transitions
├── static/
│   └── dashboard.html           # Live dashboard (tickets, sidebar, eval panel)
└── migrations/
    └── 002_agent_decisions.sql  # Eval/observability table
```

---

## Database schema (core tables)

- **`incidents`** — ticket_number, title, description, resolution, team_id, severity_id, status_id, reported_by, assigned_to, embedding (`vector(1536)`), timestamps
- **`teams`**, **`severities`**, **`statuses`** — lookup tables
- **`users`** — name, email, team_id
- **`agent_decisions`** — one row per triage run: retrieved context (JSONB), confidence, self-check result, final decision (JSONB), human label

See `migrations/002_agent_decisions.sql` for the eval table's full definition.

---

## Setup

1. **Database**: create the schema (`incidents`, `teams`, `severities`, `statuses`, `users` with pgvector enabled), then run `migrations/002_agent_decisions.sql`.
2. **Environment variables** (`.env`):
   ```
   GEMINI_API_KEY=...
   DATABASE_URL=postgresql://...       # Supabase connection string
   JIRA_BASE_URL=https://yourdomain.atlassian.net
   JIRA_EMAIL=you@yourcompany.com
   JIRA_API_TOKEN=...                  # id.atlassian.com/manage-profile/security/api-tokens
   ```
3. **Install dependencies**:
   ```
   pip install fastapi uvicorn google-genai psycopg2-binary python-dotenv requests
   ```
4. **Jira webhook**: Settings → System → Webhooks → fire on `Issue Created` and `Issue Updated`, pointing at `https://<your-tunnel>/jira-webhook` (e.g. via ngrok for local dev).
5. **Run**:
   ```
   python app.py
   ```
   Open `http://localhost:8000` for the dashboard.

---

## Known scope / limitations

- The ReAct loop is capped at a small number of iterations and only ever queries the same vector index — a deliberately scoped loop, not an open-ended agent with broad tool access.
- "Active" status next to team members in the sidebar is a visual stand-in; the schema has no real presence/online tracking.
- Auto-transitioning a ticket to "In Progress" is best-effort — if a Jira workflow doesn't allow a direct transition from the ticket's current state, it's skipped and logged rather than failing the run.
- `reported_by` often stays `NULL` since Jira Cloud frequently withholds reporter email in webhook payloads for privacy reasons; `assigned_to` is populated via load-balanced fallback to a real member of the assigned team when Jira's own assignee field is empty.
