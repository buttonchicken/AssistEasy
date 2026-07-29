# AssistEasy

A personal Telegram bot assistant. It holds a normal conversation (backed by Gemini, with Groq as a failover) and manages three kinds of daily/scheduled alerts: commute routes, DSA/system-design practice sets, and scraped job listings.

## Features

- **Conversational assistant** — general chat via `ChatGoogleGenerativeAI` (Gemini), automatically failing over to backup Gemini keys and then Groq if a key is rate-limited or exhausted. Conversation memory is a LangGraph `StateGraph` checkpointed to Postgres, so it survives bot restarts/redeploys — `/start` clears a chat's memory.
- **Route alerts** (`/makerouteasy`) — daily commute distance/ETA between an origin and destination, geocoded via Google Maps (falls back to Nominatim) with driving directions via Google Distance Matrix (falls back to OSRM).
- **Grind alerts** (`/grindalert`) — a daily study set: one system-design topic (scraped from the [system-design-primer](https://github.com/donnemartin/system-design-primer) README) plus two real LeetCode DSA problems (via LeetCode's public GraphQL API), summarized by the LLM.
- **Job alerts** (`/jobalert`) — daily scraped LinkedIn job listings filtered by domain, experience, location, and posting freshness.

All alerts are password-gated behind a shared access password, stored server-side, and delivered on a daily schedule you choose (24-hour `HH:MM`, evaluated in IST).

## Tech stack

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) (webhook mode)
- [Tornado](https://www.tornadoweb.org/) — serves the Telegram webhook, plus `/` and `/ping` health checks
- [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/) — LLM orchestration and persistent chat memory
- [Postgres](https://www.postgresql.org/) (hosted on Render) — stores scheduled alerts and LangGraph chat checkpoints
- `requests` + `BeautifulSoup` — scraping (LinkedIn jobs, system-design topics)

## Setup

```bash
python3 -m venv aeasy
source aeasy/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
TELEGRAM_TOKEN=              # from BotFather
GEMINI_API_KEY=              # primary Gemini key
GEMINI_API_KEY_1=            # optional backup Gemini key
GROQ_API_KEY=                # optional final failover
GOOGLE_MAPS_API_KEY=         # optional; falls back to Nominatim/OSRM if unset
DATABASE_URL=                # Postgres connection string (alerts + chat memory)
RENDER_EXTERNAL_URL=         # public HTTPS URL this instance is reachable at, for the Telegram webhook
PORT=8080                    # optional, defaults to 8080
```

Run it:

```bash
python main.py
```

On startup the bot initializes the Postgres tables/checkpointer, registers a Telegram webhook at `${RENDER_EXTERNAL_URL}/webhook`, and starts a background loop that checks once a minute for alerts due to fire.

## Bot commands

| Command | Description |
|---|---|
| `/start` | Greet the bot and clear your conversation memory |
| `/makerouteasy` | Set up a daily commute route alert |
| `/myroutes` / `/deleteroute <id>` | List / delete your route alerts |
| `/grindalert` | Set up a daily DSA + system-design study alert |
| `/mygrinds` / `/deletegrind <id>` | List / delete your grind alerts |
| `/jobalert` | Set up a daily scraped job-listing alert |
| `/myjobs` / `/deletejob <id>` | List / delete your job alerts |
| `/myalerts` | List every alert you have scheduled, across all types |
| `/cancel` | Abort an in-progress alert setup wizard |

Anything that isn't a recognized command is treated as a normal chat message.

## Data

A single `alerts` table in Postgres stores all alert types (route, grind, job, ...), keyed by Telegram chat ID with a JSON payload column for type-specific fields. Chat conversation history is stored separately as LangGraph checkpoints in the same database.
