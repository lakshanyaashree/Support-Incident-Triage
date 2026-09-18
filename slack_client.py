"""
slack_client.py
Posts the agent's internal message to Slack.

Uses a Bot token + chat.postMessage rather than an Incoming Webhook --
a webhook URL is bound to a single channel at creation time, so it
can't honor the `channel` the model chooses. A bot token can post to
any channel it's been invited to.

Needs in .env:
    SLACK_BOT_TOKEN=xoxb-...

Setup: create a Slack app -> OAuth & Permissions -> add the
"chat:write" bot scope -> install to workspace -> copy the
"Bot User OAuth Token" -> invite the bot into #incidents (or whatever
channel you're posting to) with `/invite @your-bot-name`.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]


def notify_slack_channel(channel: str, message: str) -> None:
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={"channel": channel, "text": message},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        # Slack returns 200 OK even on failure -- the real error is in the body.
        # Common ones: "channel_not_found" (bot not invited), "not_in_channel",
        # "invalid_auth" (wrong/expired token).
        raise RuntimeError(f"Slack API error: {data.get('error')}")
