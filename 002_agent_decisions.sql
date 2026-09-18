-- Run this once in Supabase's SQL editor before starting the updated app.
-- Backs improvement #7 (evaluation/observability): every triage decision
-- the agent makes gets logged here -- what it retrieved, how confident it
-- was, whether its own self-check passed, and (later) whether a human
-- reviewer agreed with it. This is what lets you say in an interview
-- "I measured X% agreement between the agent and human review" instead
-- of just "it seemed to work."

CREATE TABLE agent_decisions (
    id BIGSERIAL PRIMARY KEY,

    ticket_number VARCHAR(30) NOT NULL,

    -- How many ReAct loop iterations it took (1 = decided immediately
    -- off the seeded context, >1 = it called search_incidents itself
    -- one or more times to refine its query first).
    iteration_count INT NOT NULL,

    -- The incidents retrieved and shown to the model across the whole
    -- loop, as JSON, so you can audit exactly what context it had.
    retrieved_context JSONB,

    top_similarity FLOAT,

    -- 'high' if top_similarity cleared SIMILARITY_THRESHOLD, else 'low'.
    confidence VARCHAR(20),

    -- Result of the post-generation self-check critic call (#2 / #4):
    -- did the cited source_ticket_number actually exist in retrieved
    -- context, and did a second model pass judge the resolution as
    -- actually following from it.
    self_check_passed BOOLEAN,
    self_check_notes TEXT,

    -- The final structured decision (priority, team, reasoning,
    -- suggested_resolution, source_ticket_number, needs_human_review).
    decision JSONB,

    -- Heuristic flag from a keyword scan of the ticket description for
    -- prompt-injection-style phrasing (#6). Not a hard block -- just
    -- something to look at.
    prompt_injection_flag BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Filled in later by a human reviewer via the dashboard's eval panel.
    human_label VARCHAR(20),      -- 'correct' | 'incorrect' | NULL (unlabeled)
    human_label_notes TEXT,
    labeled_at TIMESTAMPTZ
);

CREATE INDEX idx_agent_decisions_ticket ON agent_decisions (ticket_number);
CREATE INDEX idx_agent_decisions_unlabeled ON agent_decisions (human_label) WHERE human_label IS NULL;
