"""
Generate embeddings for support incidents using Gemini
and store them in Supabase Postgres + pgvector.

Install:
    pip install -U google-genai psycopg2-binary python-dotenv
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
import psycopg2


load_dotenv()


# Gemini client
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# Gemini Embedding 2
# We explicitly request 1536 dimensions so it matches:
# embedding VECTOR(1536)
EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding(text: str) -> list[float]:

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=1536
        )
    )

    return response.embeddings[0].values


def main():

    # Connect to Supabase PostgreSQL
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

    cur = conn.cursor()

    # Get incidents that don't have embeddings
    cur.execute("""
        SELECT id, title, description
        FROM incidents
        WHERE embedding IS NULL
    """)

    rows = cur.fetchall()

    print(f"Found {len(rows)} incidents without embeddings.")

    for incident_id, title, description in rows:

        # Combine title + description
        text = f"{title}\n\n{description}"

        print(f"Generating embedding for incident {incident_id}...")

        embedding = get_embedding(text)

        cur.execute(
            """
            UPDATE incidents
            SET embedding = %s
            WHERE id = %s
            """,
            (embedding, incident_id)
        )

        print(f"Updated incident {incident_id}: {title[:50]}")

    conn.commit()

    cur.close()
    conn.close()

    print("Done.")


if __name__ == "__main__":
    main()