import os
import re
import logging
import asyncio
import datetime
import urllib.parse
import requests
import tornado.web
import tornado.escape
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import routes_db

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# The bot always operates in IST, regardless of the host server's system timezone.
IST = ZoneInfo("Asia/Kolkata")

def now_ist() -> datetime.datetime:
    return datetime.datetime.now(IST)

llm = None
active_api_key_name = "GEMINI_API_KEY"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant talking to a user on Telegram. Keep responses clear and concise. "
    "Do not use asterisks (*) for formatting (no bold, no italics, no bullet points). For bullet points, "
    "use a dash (-) or a unicode bullet point (•). You must NEVER reveal, share, or mention the password "
    "to the user under any circumstances. If the user asks for the passcode or password, tell them you do not know it."
)

# Persistent chat memory: a LangGraph checkpointer backed by the shared Postgres database,
# keyed on Telegram chat_id as the thread_id. Survives process restarts/redeploys,
# unlike a plain in-memory dict or a SQLite file on Render's ephemeral disk.
chat_graph = None
_chat_checkpointer_cm = None

async def call_model(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

chat_graph_builder = StateGraph(MessagesState)
chat_graph_builder.add_node("chatbot", call_model)
chat_graph_builder.add_edge(START, "chatbot")
chat_graph_builder.add_edge("chatbot", END)

async def init_chat_graph():
    global chat_graph, _chat_checkpointer_cm
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("Missing environment variable: DATABASE_URL (required for chat memory persistence)")
    _chat_checkpointer_cm = AsyncPostgresSaver.from_conn_string(database_url)
    checkpointer = await _chat_checkpointer_cm.__aenter__()
    await checkpointer.setup()
    chat_graph = chat_graph_builder.compile(checkpointer=checkpointer)

async def close_chat_graph():
    if _chat_checkpointer_cm is not None:
        await _chat_checkpointer_cm.__aexit__(None, None, None)

def get_active_api_key():
    global active_api_key_name
    key = os.getenv(active_api_key_name)
    if not key and active_api_key_name == "GEMINI_API_KEY":
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY")
    return key

def initialize_llm():
    global llm, active_api_key_name
    api_key = get_active_api_key()

    if active_api_key_name == "GROQ_API_KEY":
        logging.info("Initializing ChatGroq with key: GROQ_API_KEY")
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=api_key
        )
    else:
        logging.info(f"Initializing ChatGoogleGenerativeAI with key: {active_api_key_name}")
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0.7,
            google_api_key=api_key
        )

def switch_api_key():
    global active_api_key_name
    if active_api_key_name == "GEMINI_API_KEY":
        backup_key = os.getenv("GEMINI_API_KEY_1")
        if backup_key:
            logging.warning("FAILOVER: Primary GEMINI_API_KEY exhausted. Switching to backup GEMINI_API_KEY_1.")
            active_api_key_name = "GEMINI_API_KEY_1"
            initialize_llm()
            return True
        else:
            logging.warning("Primary GEMINI_API_KEY exhausted, and GEMINI_API_KEY_1 is not set. Checking GROQ_API_KEY...")
            active_api_key_name = "GEMINI_API_KEY_1"
            
    if active_api_key_name == "GEMINI_API_KEY_1":
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            logging.warning("FAILOVER: Gemini API keys exhausted. Switching to fallback GROQ_API_KEY.")
            active_api_key_name = "GROQ_API_KEY"
            initialize_llm()
            return True
        else:
            logging.error("FAILOVER ERROR: Neither GEMINI_API_KEY_1 nor GROQ_API_KEY is configured in the environment.")
            
    return False

# Initialize LLM with primary key on startup
initialize_llm()

# Stateful route setup tracking
user_route_setups = {}

def add_route(chat_id: str, origin: str, origin_lat: float, origin_lon: float, destination: str, destination_lat: float, destination_lon: float, scheduled_time: str) -> int:
    payload = {
        "origin": origin,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "destination": destination,
        "destination_lat": destination_lat,
        "destination_lon": destination_lon
    }
    return routes_db.add_alert(chat_id, "route", scheduled_time, payload)

def get_user_routes(chat_id: str):
    return routes_db.get_user_alerts(chat_id, "route")

def delete_user_route(chat_id: str, route_id: int) -> bool:
    return routes_db.delete_user_alert(chat_id, "route", route_id)

def get_routes_to_trigger(current_time: str, current_date: str):
    return routes_db.get_alerts_to_trigger("route", current_time, current_date)

def update_last_sent(route_id: int, current_date: str):
    routes_db.update_last_sent(route_id, current_date)

def add_grind_alert(chat_id: str, scheduled_time: str) -> int:
    return routes_db.add_alert(chat_id, "grind", scheduled_time, {})

def get_user_grind_alerts(chat_id: str):
    return routes_db.get_user_alerts(chat_id, "grind")

def delete_user_grind_alert(chat_id: str, grind_id: int) -> bool:
    return routes_db.delete_user_alert(chat_id, "grind", grind_id)

def get_grind_alerts_to_trigger(current_time: str, current_date: str):
    return routes_db.get_alerts_to_trigger("grind", current_time, current_date)

def update_grind_last_sent(grind_id: int, current_date: str):
    routes_db.update_last_sent(grind_id, current_date)

# --- Job Alert DB helpers ---

def add_job_alert(chat_id: str, domain: str, experience: str, location: str, freshness_seconds: int, scheduled_time: str) -> int:
    payload = {"domain": domain, "experience": experience, "location": location, "freshness_seconds": freshness_seconds}
    return routes_db.add_alert(chat_id, "job", scheduled_time, payload)

def get_user_job_alerts(chat_id: str):
    return routes_db.get_user_alerts(chat_id, "job")

def delete_user_job_alert(chat_id: str, job_id: int) -> bool:
    return routes_db.delete_user_alert(chat_id, "job", job_id)

def get_job_alerts_to_trigger(current_time: str, current_date: str):
    return routes_db.get_alerts_to_trigger("job", current_time, current_date)

def update_job_last_sent(job_id: int, current_date: str):
    routes_db.update_last_sent(job_id, current_date)

def get_all_user_alerts(chat_id: str):
    return routes_db.get_all_user_alerts(chat_id)

# --- BLR Homecoming Alert DB helpers ---

def add_blrhomecoming_alert(chat_id: str, scheduled_time: str) -> int:
    return routes_db.add_alert(chat_id, "blrhomecoming", scheduled_time, {})

def get_user_blrhomecoming_alerts(chat_id: str):
    return routes_db.get_user_alerts(chat_id, "blrhomecoming")

def delete_user_blrhomecoming_alert(chat_id: str, alert_id: int) -> bool:
    return routes_db.delete_user_alert(chat_id, "blrhomecoming", alert_id)

def get_blrhomecoming_alerts_to_trigger(current_time: str, current_date: str):
    return routes_db.get_alerts_to_trigger("blrhomecoming", current_time, current_date)

def update_blrhomecoming_last_sent(alert_id: int, current_date: str):
    routes_db.update_last_sent(alert_id, current_date)

def search_locations(query: str):
    encoded_query = urllib.parse.quote_plus(query)
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if api_key:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_query}&key={api_key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "OK":
                results = []
                for r in data["results"][:5]:
                    results.append({
                        "display_name": r["formatted_address"],
                        "lat": r["geometry"]["location"]["lat"],
                        "lon": r["geometry"]["location"]["lng"]
                    })
                return results, None
            elif data.get("status") == "ZERO_RESULTS":
                return [], None
            return [], f"Google API returned: {data.get('status')}"
        except Exception as e:
            logging.error(f"Google Geocoding error: {e}. Falling back to Nominatim.")
            
    # Fallback to Nominatim
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=5"
    headers = {"User-Agent": "AssistEasyBot/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data:
            results.append({
                "display_name": item["display_name"],
                "lat": item["lat"],
                "lon": item["lon"]
            })
        return results, None
    except Exception as e:
        return [], f"Nominatim API error: {e}"

def build_selection_keyboard(options):
    keyboard = []
    for idx, opt in enumerate(options):
        name = opt["display_name"]
        if len(name) > 55:
            name = name[:52] + "..."
        keyboard.append([InlineKeyboardButton(name, callback_data=f"loc:{idx}")])
    return InlineKeyboardMarkup(keyboard)

def get_route_info_by_coords(lat1, lon1, lat2, lon2, origin_name: str, destination_name: str):
    # Google Maps Directions URL
    encoded_origin = urllib.parse.quote_plus(origin_name)
    encoded_dest = urllib.parse.quote_plus(destination_name)
    map_link = f"https://www.google.com/maps/dir/?api=1&origin={encoded_origin}&destination={encoded_dest}"
    
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        try:
            url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat1},{lon1}&destinations={lat2},{lon2}&key={api_key}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "OK" and data.get("rows"):
                element = data["rows"][0]["elements"][0]
                if element.get("status") == "OK":
                    distance = element["distance"]["text"]
                    duration = element["duration"]["text"]
                    return {
                        "distance": distance,
                        "duration": duration,
                        "map_link": map_link
                    }, None
                else:
                    logging.warning(f"Google Distance Matrix returned element status: {element.get('status')}. Falling back to OSRM.")
            else:
                logging.warning(f"Google Distance Matrix returned status: {data.get('status')}. Falling back to OSRM.")
        except Exception as e:
            logging.error(f"Error querying Google Distance Matrix: {e}. Falling back to OSRM.")

    # Fallback to OSRM
    route_url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        r = requests.get(route_url, timeout=10)
        r.raise_for_status()
        route_data = r.json()
        if "routes" not in route_data or not route_data["routes"]:
            return None, "No route found between these coordinates"
            
        route = route_data["routes"][0]
        distance_meters = route["distance"]
        duration_seconds = route["duration"]
        
        distance_km = distance_meters / 1000.0
        distance_str = f"{distance_km:.1f} km"
        
        minutes = int(duration_seconds / 60)
        if minutes < 60:
            duration_str = f"{minutes} mins"
        else:
            hours = minutes // 60
            remaining_mins = minutes % 60
            duration_str = f"{hours}h {remaining_mins}m"
            
        return {
            "distance": distance_str,
            "duration": duration_str,
            "map_link": map_link
        }, None
    except Exception as e:
        return None, f"Error calling routing service: {e}"

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    chat_id = str(update.effective_chat.id)
    if chat_id not in user_route_setups:
        await query.edit_message_text("This session has expired. Please start over with /makerouteasy.")
        return
        
    setup = user_route_setups[chat_id]
    state = setup["state"]
    data = query.data
    
    if not data.startswith("loc:"):
        return
        
    try:
        idx = int(data.split(":")[1])
        selected_loc = setup["options"][idx]
    except (IndexError, ValueError):
        await query.edit_message_text("Error selecting location. Please start over with /makerouteasy.")
        return
        
    if state == "SELECTING_ORIGIN":
        setup["origin_name"] = selected_loc["display_name"]
        setup["origin_lat"] = selected_loc["lat"]
        setup["origin_lon"] = selected_loc["lon"]
        
        setup["state"] = "AWAITING_DESTINATION"
        setup["options"] = None
        
        await query.edit_message_text(
            f"Origin set to: {selected_loc['display_name']}\n\n"
            "Now, please enter your destination (arrival location), or type /cancel to abort."
        )
        
    elif state == "SELECTING_DESTINATION":
        setup["destination_name"] = selected_loc["display_name"]
        setup["destination_lat"] = selected_loc["lat"]
        setup["destination_lon"] = selected_loc["lon"]
        
        setup["state"] = "AWAITING_TIME"
        setup["options"] = None
        
        current_time = now_ist().strftime("%H:%M")
        await query.edit_message_text(
            f"Destination set to: {selected_loc['display_name']}\n\n"
            f"Finally, please enter the daily scheduled time in 24-hour HH:MM format (e.g., 08:30 or 17:45).\n"
            f"Note: The bot's current time is {current_time}."
        )

async def makerouteasy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_route_setups[chat_id] = {
        "state": "AWAITING_PASSWORD",
        "target_alert_type": "route"
    }
    await update.message.reply_text(
        "To set up a daily route alert, please enter the access password:"
    )

async def cancel_route_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if chat_id in user_route_setups:
        del user_route_setups[chat_id]
        await update.message.reply_text("Alert setup has been cancelled.")
    else:
        await update.message.reply_text("No active setup to cancel.")

async def list_routes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    routes = get_user_routes(chat_id)
    if not routes:
        await update.message.reply_text("You have no scheduled routes. Set one up using /makerouteasy !")
        return
    
    msg = "Your scheduled routes:\n\n"
    for r in routes:
        msg += f"ID: {r['id']}\nRoute: {r['origin']} ➡️ {r['destination']}\nTime: {r['scheduled_time']} daily\nTo delete: /deleteroute {r['id']}\n\n"
    await update.message.reply_text(msg.strip())

async def delete_route(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args
    if not args:
        await update.message.reply_text("Please provide the ID of the route to delete, e.g., /deleteroute 3.")
        return
    try:
        route_id = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid route ID. Please use a number, e.g., /deleteroute 3.")
        return
        
    success = delete_user_route(chat_id, route_id)
    if success:
        await update.message.reply_text(f"Successfully deleted route ID {route_id}.")
    else:
        await update.message.reply_text(f"Could not find route with ID {route_id} scheduled by you.")

# Fallback pools, only used if the live fetches below fail (network error, site/API changes, etc.)
SYSTEM_DESIGN_DOMAINS_FALLBACK = [
    "Bitly (URL shortener)", "Uber (Ride sharing / Geolocation)", "Netflix or YouTube (Video streaming / CDN)",
    "Dropbox or Google Drive (File sync & storage)", "Twitter or Facebook (News feed architecture)",
    "WhatsApp or Slack (Real-time chat & presence)", "Airbnb or Booking.com (Hotel reservation)",
    "Tinder (Matchmaking & location query)", "Amazon (Distributed shopping cart & checkout)",
    "Web Crawler (Politeness, duplicate detection, URL frontier)", "Distributed Rate Limiter",
    "Distributed Cache (e.g., Memcached / Redis)", "Ad Click Event Aggregator (Stream processing)",
    "Snowflake (Distributed Unique ID generator)", "Push Notification System",
    "Google Search Autocomplete (Trie, MapReduce)", "Gaming Leaderboard (Redis Sorted Sets)",
    "API Gateway", "Distributed Job Scheduler"
]

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
SYSTEM_DESIGN_PRIMER_README_URL = "https://raw.githubusercontent.com/donnemartin/system-design-primer/master/README.md"
_SYSTEM_DESIGN_TOPICS_CACHE = {"topics": [], "fetched_at": None}
_SYSTEM_DESIGN_CACHE_TTL = datetime.timedelta(hours=24)

def _readme_section(text: str, header: str) -> str:
    match = re.search(r"^## " + re.escape(header) + r"\n(.*?)\n## ", text, re.S | re.M)
    return match.group(1) if match else ""

def _scrape_system_design_topics() -> list:
    """Scrape real interview-style design prompts and core system design concepts from the system-design-primer README."""
    resp = requests.get(SYSTEM_DESIGN_PRIMER_README_URL, timeout=10)
    resp.raise_for_status()
    text = resp.text

    interview_prompts = re.findall(
        r"^### (.+)$", _readme_section(text, "System design interview questions with solutions"), re.M
    )

    core_start = text.find("## Performance vs scalability")
    core_end = text.find("## Appendix")
    core_topics = []
    if core_start != -1 and core_end != -1:
        core_topics = re.findall(r"^## (.+)$", text[core_start:core_end], re.M)

    return interview_prompts + core_topics

async def get_system_design_topic() -> str:
    """Return a random system design case study title, live-scraped and cached for a day."""
    import random
    now = now_ist()
    cache = _SYSTEM_DESIGN_TOPICS_CACHE
    if not (cache["topics"] and cache["fetched_at"] and now - cache["fetched_at"] < _SYSTEM_DESIGN_CACHE_TTL):
        try:
            loop = asyncio.get_event_loop()
            topics = await loop.run_in_executor(None, _scrape_system_design_topics)
            if topics:
                cache["topics"] = topics
                cache["fetched_at"] = now
        except Exception as e:
            logging.error(f"Failed to scrape system design topics, falling back to static list: {e}")
    return random.choice(cache["topics"] or SYSTEM_DESIGN_DOMAINS_FALLBACK)

def _fetch_random_leetcode_problems(count: int = 2) -> list:
    """Fetch `count` distinct random real (non-paywalled) problems via LeetCode's public GraphQL API."""
    import random
    query = """
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
      problemsetQuestionList: questionList(categorySlug: $categorySlug, limit: $limit, skip: $skip, filters: $filters) {
        total: totalNum
        questions: data {
          title
          titleSlug
          difficulty
          isPaidOnly
          topicTags { name }
        }
      }
    }
    """
    headers = {"Content-Type": "application/json", "Referer": "https://leetcode.com"}

    def run_query(skip, limit):
        resp = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": {"categorySlug": "", "skip": skip, "limit": limit, "filters": {}}},
            headers=headers, timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["data"]["problemsetQuestionList"]

    total = run_query(0, 1)["total"]

    problems = []
    seen_slugs = set()
    for _ in range(count * 5):
        if len(problems) >= count:
            break
        skip = random.randint(0, total - 1)
        data = run_query(skip, 1)["questions"]
        if not data:
            continue
        q = data[0]
        if q["isPaidOnly"] or q["titleSlug"] in seen_slugs:
            continue
        seen_slugs.add(q["titleSlug"])
        problems.append({
            "title": q["title"],
            "difficulty": q["difficulty"],
            "tags": [t["name"] for t in q["topicTags"]],
            "link": f"https://leetcode.com/problems/{q['titleSlug']}/",
        })
    return problems

async def fetch_grind_problems():
    sys_topic = await get_system_design_topic()

    dsa_problems = []
    try:
        loop = asyncio.get_event_loop()
        dsa_problems = await loop.run_in_executor(None, _fetch_random_leetcode_problems, 2)
    except Exception as e:
        logging.error(f"Failed to fetch random LeetCode problems: {e}")

    if len(dsa_problems) == 2:
        dsa_instructions = (
            "2. Two DSA (Data Structures and Algorithms) problems. These are real LeetCode problems -- "
            "use their exact name, difficulty, and link as given below, and write a brief 1-sentence description "
            "of what each problem asks. Do NOT invent a different problem, difficulty, or link.\n"
            + "\n".join(
                f"   - \"{p['title']}\" (Difficulty: {p['difficulty']}, Topics: {', '.join(p['tags'][:3]) or 'General'}) - {p['link']}"
                for p in dsa_problems
            )
        )
    else:
        dsa_instructions = (
            "2. Two random DSA (Data Structures and Algorithms) problems covering different concepts. "
            "For each, provide the name, difficulty (Easy/Medium/Hard), a brief 1-sentence description, "
            "and their official LeetCode link. Ensure the links are valid."
        )

    prompt = (
        f"Generate a technical study set containing:\n"
        f"1. One System Design problem/topic related to: {sys_topic}. Provide a brief overview of key challenges, database choices, and scaling.\n"
        f"{dsa_instructions}\n\n"
        f"Formatting constraints: Do NOT use asterisks (*) for bold or italics. For lists, use simple dashes (-) or numbers. Keep it clean and readable."
    )
    for attempt in range(3):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content
            if isinstance(content, list):
                reply_text = "".join(
                    part.get("text", "") if isinstance(part, dict) and part.get("type") == "text" else str(part)
                    for part in content
                )
            else:
                reply_text = str(content)
            return reply_text
        except Exception as e:
            err_msg = str(e)
            logging.error(f"Error fetching grind problems from Gemini/Groq (attempt {attempt+1}): {e}")
            if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower() or "rate_limit" in err_msg.lower() or "limit" in err_msg.lower()) and attempt < 2:
                if switch_api_key():
                    continue
            return "Could not fetch grind problems today. Please try again later."

def scrape_linkedin_jobs(domain: str, location: str, freshness_seconds: int, max_results: int = 5) -> str:
    """Scrape real job listings from LinkedIn's public guest search endpoint."""
    from bs4 import BeautifulSoup
    import time
    import random

    # Build query URL
    keywords = urllib.parse.quote_plus(domain)
    loc_encoded = urllib.parse.quote_plus(location) if location.lower() not in ("remote", "any") else urllib.parse.quote_plus("Worldwide")
    freshness_param = f"&f_TPR=r{freshness_seconds}" if freshness_seconds else ""
    url = (
        f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        f"?keywords={keywords}&location={loc_encoded}&start=0&sortBy=DD{freshness_param}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.linkedin.com/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"LinkedIn scraper HTTP error: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    job_cards = soup.find_all("li")

    results = []
    for card in job_cards:
        if len(results) >= max_results:
            break
        try:
            title_el = card.find("h3", class_="base-search-card__title")
            company_el = card.find("h4", class_="base-search-card__subtitle")
            location_el = card.find("span", class_="job-search-card__location")
            link_el = card.find("a", class_="base-card__full-link")
            posted_el = card.find("time")

            title = title_el.get_text(strip=True) if title_el else "N/A"
            company = company_el.get_text(strip=True) if company_el else "N/A"
            loc = location_el.get_text(strip=True) if location_el else "N/A"
            link = link_el["href"].split("?")[0] if link_el and link_el.get("href") else "N/A"
            posted = posted_el["datetime"] if posted_el and posted_el.get("datetime") else ""

            if title == "N/A" and company == "N/A":
                continue
            results.append({"title": title, "company": company, "location": loc, "link": link, "posted": posted})
        except Exception:
            continue

    return results

async def fetch_jobs(domain: str, experience: str, location: str, freshness_seconds: int) -> str:
    """Scrape LinkedIn for real job listings and return a formatted string."""
    loop = asyncio.get_event_loop()
    jobs = await loop.run_in_executor(None, scrape_linkedin_jobs, domain, location, freshness_seconds)

    freshness_labels = {
        3600: "past 1 hour", 7200: "past 2 hours", 28800: "past 8 hours",
        86400: "past 24 hours", 172800: "past 2 days", 604800: "past week", 2592000: "past month"
    }
    freshness_label = freshness_labels.get(freshness_seconds, f"past {freshness_seconds//3600} hours")

    if not jobs:
        # Fallback: return a clickable LinkedIn search link
        kw = urllib.parse.quote_plus(domain)
        loc_enc = urllib.parse.quote_plus(location)
        tpr = f"&f_TPR=r{freshness_seconds}" if freshness_seconds else ""
        fallback_url = f"https://www.linkedin.com/jobs/search/?keywords={kw}&location={loc_enc}&sortBy=DD{tpr}"
        return (
            f"Could not scrape live listings right now (LinkedIn may be rate-limiting).\n"
            f"Search manually here:\n{fallback_url}"
        )

    lines = [f"Found {len(jobs)} job(s) posted in the {freshness_label}:\n"]
    for i, j in enumerate(jobs, 1):
        lines.append(
            f"{i}. {j['title']}\n"
            f"   Company: {j['company']}\n"
            f"   Location: {j['location']}\n"
            f"   Posted: {j['posted']}\n"
            f"   Link: {j['link']}"
        )
    return "\n\n".join(lines)

# --- BLR Homecoming Alert: fixed source list + keyword scraper ---

BLR_HOMECOMING_KEYWORDS = ["esg", "sustainability", "brsr"]

# Link path fragments that match the keywords but are never actual job postings
# (social feed posts, hashtag pages, signup walls, marketing/course content).
BLR_HOMECOMING_NOISE_PATTERNS = [
    "/hashtag/", "/signup", "/posts/", "/pulse/", "/feed/",
    "climate-change-courses", "/events/",
]

BLR_HOMECOMING_SOURCES = [
    {"name": "Ather Energy", "url": "https://careers.atherenergy.com/jobs"},
    {"name": "Climes", "url": "https://climes.notion.site/Join-the-team-fb658bd0f5924180842b249bf412ba4a"},
    {"name": "Terra.do Climate Job Board", "url": "https://www.terra.do/climate-jobs/job-board/?isEpp=false&recent=true&remote=false"},
    {"name": "Varaha", "url": "https://www.varaha.earth/careers"},
    {"name": "String Bio", "url": "https://www.stringbio.com/life-at-string.html#careers"},
    {"name": "Yulu", "url": "https://careers.yulu.bike/"},
    {"name": "Sun Mobility (LinkedIn)", "url": "https://www.linkedin.com/jobs/search/?currentJobId=4426606139&keywords=sun%20mobility&location=worldwide"},
    {"name": "Kazam Energy", "url": "https://kazam.energy/careers"},
    {"name": "Saahas Zero Waste", "url": "https://saahas.zohorecruit.com/recruit/Portal.na?digest=j@GbK5pmYRXio5XkM29UDHGAUnkE7bVo.a85bxdBEmQ-&iframe=false&mode=home&embedsource=CareerSite"},
    {"name": "Loopworm Bio (LinkedIn)", "url": "https://www.linkedin.com/company/loopwormbio/jobs/"},
    {"name": "Hasiru Dala Innovations", "url": "https://hasirudalainnovations.com/careers/"},
    {"name": "Orb Energy (LinkedIn)", "url": "https://www.linkedin.com/company/orbenergy/jobs/"},
    {"name": "Solar Square (Keka)", "url": "https://solarsquare.keka.com/careers"},
]

def scrape_keyword_matches(url: str, keywords: list, max_results: int = 5):
    """Fetch a careers page and look for links/text matching any of the given keywords.

    Returns (matches, status) where status is one of:
      "ok"        - matches is a list of {"text", "link"} dicts (possibly empty)
      "mentioned" - keyword appears somewhere on the page but not in a distinct link
      "error"     - matches is None, status is "error" and error text is in `matches`
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return str(e), "error"

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    matches = []
    seen = set()
    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href") or ""
        # Ignore query strings/fragments when matching hrefs -- tracking params
        # (e.g. LinkedIn's base64 trackingId=...) can coincidentally contain a keyword.
        href_path = urllib.parse.urlsplit(href).path
        combined = f"{text} {href_path}".lower()
        if any(np in combined for np in BLR_HOMECOMING_NOISE_PATTERNS):
            continue
        if any(kw in combined for kw in keywords):
            full_link = urllib.parse.urljoin(url, href) if href else url
            key = (text.strip().lower(), full_link)
            if key in seen:
                continue
            seen.add(key)
            matches.append({"text": (text.strip() or "(untitled link)")[:120], "link": full_link})
            if len(matches) >= max_results:
                break

    if matches:
        return matches, "ok"

    page_text = soup.get_text(" ", strip=True).lower()
    if any(kw in page_text for kw in keywords):
        return [], "mentioned"

    return [], "ok"

def scrape_blr_homecoming() -> list:
    """Scrape all BLR Homecoming source sites in parallel for ESG/sustainability/BRSR keyword matches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_source = {
            executor.submit(scrape_keyword_matches, src["url"], BLR_HOMECOMING_KEYWORDS): src
            for src in BLR_HOMECOMING_SOURCES
        }
        for future in as_completed(future_to_source):
            src = future_to_source[future]
            try:
                matches, status = future.result()
            except Exception as e:
                matches, status = str(e), "error"
            results[src["url"]] = {"name": src["name"], "url": src["url"], "matches": matches, "status": status}

    return [results[src["url"]] for src in BLR_HOMECOMING_SOURCES]

async def fetch_blr_homecoming_report() -> str:
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, scrape_blr_homecoming)

    lines = [f"Keywords tracked: {', '.join(BLR_HOMECOMING_KEYWORDS)}\n"]
    for r in report:
        lines.append(f"🏢 {r['name']}")
        if r["status"] == "error":
            lines.append(f"   Could not fetch page ({r['matches']}). Check manually: {r['url']}")
        elif r["status"] == "mentioned":
            lines.append(f"   Keyword mentioned on the page, but no specific role link found. Check manually: {r['url']}")
        elif r["matches"]:
            for m in r["matches"]:
                lines.append(f"   - {m['text']}\n     {m['link']}")
        else:
            lines.append(f"   No matches found today.")
        lines.append("")

    return "\n".join(lines).strip()

async def grindalert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_route_setups[chat_id] = {
        "state": "AWAITING_PASSWORD",
        "target_alert_type": "grind"
    }
    await update.message.reply_text(
        "To set up a daily Grind Alert, please enter the access password:"
    )

async def list_grinds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_grinds = get_user_grind_alerts(chat_id)
    if not user_grinds:
        await update.message.reply_text("You have no scheduled Grind Alerts. Set one up using /grindalert !")
        return
    
    msg = "Your scheduled Grind Alerts:\n\n"
    for g in user_grinds:
        msg += f"ID: {g['id']}\nTime: {g['scheduled_time']} daily\nTo delete: /deletegrind {g['id']}\n\n"
    await update.message.reply_text(msg.strip())

async def delete_grind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args
    if not args:
        await update.message.reply_text("Please provide the ID of the Grind Alert to delete, e.g., /deletegrind 1.")
        return
    try:
        grind_id = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid ID. Please use a number, e.g., /deletegrind 1.")
        return
        
    success = delete_user_grind_alert(chat_id, grind_id)
    if success:
        await update.message.reply_text(f"Successfully deleted Grind Alert ID {grind_id}.")
    else:
        await update.message.reply_text(f"Could not find Grind Alert with ID {grind_id} scheduled by you.")

async def jobalert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_route_setups[chat_id] = {
        "state": "AWAITING_PASSWORD",
        "target_alert_type": "job"
    }
    await update.message.reply_text(
        "To set up a daily Job Alert, please enter the access password:"
    )

async def list_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_jobs = get_user_job_alerts(chat_id)
    if not user_jobs:
        await update.message.reply_text("You have no scheduled Job Alerts. Set one up using /jobalert !")
        return
    msg = "Your scheduled Job Alerts:\n\n"
    for j in user_jobs:
        freshness_labels = {
            3600: "Past 1 hour", 7200: "Past 2 hours", 28800: "Past 8 hours",
            86400: "Past 24 hours", 172800: "Past 2 days", 604800: "Past week", 2592000: "Past month"
        }
        freshness_label = freshness_labels.get(j.get("freshness_seconds", 86400), "Past 24 hours")
        msg += (
            f"ID: {j['id']}\n"
            f"Domain: {j.get('domain', 'N/A')}\n"
            f"Experience: {j.get('experience', 'N/A')}\n"
            f"Location: {j.get('location', 'N/A')}\n"
            f"Freshness: {freshness_label}\n"
            f"Time: {j['scheduled_time']} daily\n"
            f"To delete: /deletejob {j['id']}\n\n"
        )
    await update.message.reply_text(msg.strip())

async def delete_job(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args
    if not args:
        await update.message.reply_text("Please provide the ID of the Job Alert to delete, e.g., /deletejob 5.")
        return
    try:
        job_id = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid ID. Please use a number, e.g., /deletejob 5.")
        return
    success = delete_user_job_alert(chat_id, job_id)
    if success:
        await update.message.reply_text(f"Successfully deleted Job Alert ID {job_id}.")
    else:
        await update.message.reply_text(f"Could not find Job Alert with ID {job_id} scheduled by you.")

async def blrhomecoming(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_route_setups[chat_id] = {
        "state": "AWAITING_PASSWORD",
        "target_alert_type": "blrhomecoming"
    }
    await update.message.reply_text(
        "To set up a daily BLR Homecoming Alert, please enter the access password:"
    )

async def list_blrhomecoming(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_alerts = get_user_blrhomecoming_alerts(chat_id)
    if not user_alerts:
        await update.message.reply_text("You have no scheduled BLR Homecoming Alerts. Set one up using /blrhomecoming !")
        return
    msg = "Your scheduled BLR Homecoming Alerts:\n\n"
    for a in user_alerts:
        msg += f"ID: {a['id']}\nTime: {a['scheduled_time']} daily\nTo delete: /deleteblrhomecoming {a['id']}\n\n"
    await update.message.reply_text(msg.strip())

async def delete_blrhomecoming(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    args = context.args
    if not args:
        await update.message.reply_text("Please provide the ID of the BLR Homecoming Alert to delete, e.g., /deleteblrhomecoming 1.")
        return
    try:
        alert_id = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid ID. Please use a number, e.g., /deleteblrhomecoming 1.")
        return
    success = delete_user_blrhomecoming_alert(chat_id, alert_id)
    if success:
        await update.message.reply_text(f"Successfully deleted BLR Homecoming Alert ID {alert_id}.")
    else:
        await update.message.reply_text(f"Could not find BLR Homecoming Alert with ID {alert_id} scheduled by you.")

async def list_all_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    alerts = get_all_user_alerts(chat_id)
    if not alerts:
        await update.message.reply_text("You have no scheduled alerts.")
        return
        
    msg = "Your Scheduled Alerts:\n\n"
    for a in alerts:
        if a["alert_type"] == "route":
            msg += (
                f"ID: {a['id']} (Route Alert)\n"
                f"Route: {a['origin']} ➡️ {a['destination']}\n"
                f"Time: {a['scheduled_time']} daily\n"
                f"To delete: /deleteroute {a['id']}\n\n"
            )
        elif a["alert_type"] == "grind":
            msg += (
                f"ID: {a['id']} (Grind Alert)\n"
                f"Time: {a['scheduled_time']} daily\n"
                f"To delete: /deletegrind {a['id']}\n\n"
            )
        elif a["alert_type"] == "job":
            freshness_labels = {
                3600: "Past 1 hour", 7200: "Past 2 hours", 28800: "Past 8 hours",
                86400: "Past 24 hours", 172800: "Past 2 days", 604800: "Past week", 2592000: "Past month"
            }
            freshness_label = freshness_labels.get(a.get("freshness_seconds", 86400), "Past 24 hours")
            msg += (
                f"ID: {a['id']} (Job Alert)\n"
                f"Domain: {a.get('domain', 'N/A')}\n"
                f"Experience: {a.get('experience', 'N/A')}\n"
                f"Location: {a.get('location', 'N/A')}\n"
                f"Freshness: {freshness_label}\n"
                f"Time: {a['scheduled_time']} daily\n"
                f"To delete: /deletejob {a['id']}\n\n"
            )
        elif a["alert_type"] == "blrhomecoming":
            msg += (
                f"ID: {a['id']} (BLR Homecoming Alert)\n"
                f"Time: {a['scheduled_time']} daily\n"
                f"To delete: /deleteblrhomecoming {a['id']}\n\n"
            )
    await update.message.reply_text(msg.strip())

async def check_and_send_grind_alerts():
    now = now_ist()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")
    
    to_trigger = get_grind_alerts_to_trigger(current_time, current_date)
    if not to_trigger:
        return
        
    logging.info(f"Triggering {len(to_trigger)} grind alerts scheduled for {current_time}...")
    problem_set_text = await fetch_grind_problems()
    
    for g in to_trigger:
        chat_id = g["chat_id"]
        alert_id = g["id"]
        
        msg = f"Daily Grind Alert! 🧠🚀\n\n{problem_set_text}"
        try:
            await app.bot.send_message(chat_id=chat_id, text=msg)
            update_grind_last_sent(alert_id, current_date)
            logging.info(f"Successfully sent Grind Alert to chat {chat_id} for alert ID {alert_id}.")
        except Exception as e:
            logging.error(f"Failed to send Grind Alert to chat {chat_id}: {e}")

async def check_and_send_route_alerts():
    now = now_ist()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")
    
    routes = get_routes_to_trigger(current_time, current_date)
    if not routes:
        return
        
    logging.info(f"Triggering {len(routes)} route updates scheduled for {current_time}...")
    
    for r in routes:
        route_id = r["id"]
        chat_id = r["chat_id"]
        origin = r["origin"]
        destination = r["destination"]
        lat1 = r["origin_lat"]
        lon1 = r["origin_lon"]
        lat2 = r["destination_lat"]
        lon2 = r["destination_lon"]
        
        res, err = get_route_info_by_coords(lat1, lon1, lat2, lon2, origin, destination)
        if err:
            logging.error(f"Scheduler failed to calculate route for ID {route_id}: {err}")
            msg = (
                f"Daily Route Update 🚗\n"
                f"Route: {origin} ➡️ {destination}\n\n"
                f"Could not calculate ETA and distance details today: {err}\n"
                f"Google Maps Link: https://www.google.com/maps/dir/?api=1&origin={urllib.parse.quote_plus(origin)}&destination={urllib.parse.quote_plus(destination)}"
            )
        else:
            msg = (
                f"Daily Route Update 🚗\n"
                f"Route: {origin} ➡️ {destination}\n"
                f"Estimated Duration: {res['duration']}\n"
                f"Shortest Distance: {res['distance']}\n\n"
                f"Google Maps Link: {res['map_link']}"
            )
            
        try:
            await app.bot.send_message(chat_id=chat_id, text=msg)
            update_last_sent(route_id, current_date)
            logging.info(f"Successfully sent route alert to chat {chat_id} for route ID {route_id}.")
        except Exception as e:
            logging.error(f"Failed to send route alert to chat {chat_id}: {e}")

async def check_and_send_job_alerts():
    now = now_ist()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")

    to_trigger = get_job_alerts_to_trigger(current_time, current_date)
    if not to_trigger:
        return

    logging.info(f"Triggering {len(to_trigger)} job alerts scheduled for {current_time}...")

    for j in to_trigger:
        chat_id = j["chat_id"]
        alert_id = j["id"]
        domain = j.get("domain", "Tech")
        experience = j.get("experience", "Any")
        location = j.get("location", "Remote")
        freshness_seconds = j.get("freshness_seconds", 86400)

        job_listings = await fetch_jobs(domain, experience, location, freshness_seconds)
        msg = f"Daily Job Alert! 💼\nDomain: {domain} | Experience: {experience} | Location: {location}\n\n{job_listings}"
        try:
            await app.bot.send_message(chat_id=chat_id, text=msg)
            update_job_last_sent(alert_id, current_date)
            logging.info(f"Sent Job Alert to chat {chat_id} for alert ID {alert_id}.")
        except Exception as e:
            logging.error(f"Failed to send Job Alert to chat {chat_id}: {e}")

def _chunk_message(text: str, max_len: int = 3800) -> list:
    """Split a long message into Telegram-safe chunks without breaking mid-line."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current.strip())
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

async def check_and_send_blrhomecoming_alerts():
    now = now_ist()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")

    to_trigger = get_blrhomecoming_alerts_to_trigger(current_time, current_date)
    if not to_trigger:
        return

    logging.info(f"Triggering {len(to_trigger)} BLR Homecoming alerts scheduled for {current_time}...")
    report = await fetch_blr_homecoming_report()
    chunks = _chunk_message(f"Daily BLR Homecoming Alert! 🌱\n\n{report}")

    for a in to_trigger:
        chat_id = a["chat_id"]
        alert_id = a["id"]
        try:
            for chunk in chunks:
                await app.bot.send_message(chat_id=chat_id, text=chunk)
            update_blrhomecoming_last_sent(alert_id, current_date)
            logging.info(f"Sent BLR Homecoming Alert to chat {chat_id} for alert ID {alert_id}.")
        except Exception as e:
            logging.error(f"Failed to send BLR Homecoming Alert to chat {chat_id}: {e}")

async def scheduler_loop():
    logging.info("Starting background route alert scheduler loop...")
    while True:
        try:
            await check_and_send_route_alerts()
            await check_and_send_grind_alerts()
            await check_and_send_job_alerts()
            await check_and_send_blrhomecoming_alerts()
        except Exception as e:
            logging.error(f"Error in scheduler check: {e}")
            
        now = now_ist()
        sleep_seconds = 60 - now.second
        if sleep_seconds <= 0:
            sleep_seconds = 60
        await asyncio.sleep(sleep_seconds)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    try:
        await chat_graph.checkpointer.adelete_thread(chat_id)
    except Exception as e:
        logging.error(f"Failed to clear chat history for {chat_id}: {e}")
    await update.message.reply_text("Hello! I'm virtual Aditya, how can I help you today?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)

    # Stateful route wizard handling
    if chat_id in user_route_setups:
        setup = user_route_setups[chat_id]
        state = setup["state"]
        
        if state == "AWAITING_PASSWORD":
            if user_text.strip() == "lifeknoteasy":
                target = setup["target_alert_type"]
                if target == "route":
                    setup["state"] = "AWAITING_ORIGIN"
                    setup.update({
                        "origin_name": None,
                        "origin_lat": None,
                        "origin_lon": None,
                        "destination_name": None,
                        "destination_lat": None,
                        "destination_lon": None,
                        "options": None
                    })
                    await update.message.reply_text(
                        "Password accepted! Let's set up your daily route alert! 🚗\n\n"
                        "First, please enter your origin (departure location), or type /cancel to abort."
                    )
                elif target == "grind":
                    setup["state"] = "AWAITING_GRIND_TIME"
                    current_time = now_ist().strftime("%H:%M")
                    await update.message.reply_text(
                        "Password accepted! Let's set up your daily Grind Alert! 🧠\n\n"
                        "At what time daily would you like to receive 1 System Design problem and 2 LeetCode DSA problems?\n"
                        f"Please enter the time in 24-hour HH:MM format (e.g., 08:00 or 20:30).\n"
                        f"Note: The bot's current time is {current_time}."
                    )
                elif target == "job":
                    setup["state"] = "AWAITING_JOB_DOMAIN"
                    await update.message.reply_text(
                        "Password accepted! Let's set up your daily Job Alert! 💼\n\n"
                        "What job domain are you looking for? (e.g., Backend Engineering, Data Science, ML Engineer, Product Manager)\n\n"
                        "Type /cancel to abort."
                    )
                elif target == "blrhomecoming":
                    setup["state"] = "AWAITING_BLR_TIME"
                    current_time = now_ist().strftime("%H:%M")
                    await update.message.reply_text(
                        "Password accepted! Let's set up your daily BLR Homecoming Alert! 🌱\n\n"
                        "This will scan a fixed list of climate/ESG-adjacent Bangalore employers for roles "
                        "mentioning 'esg', 'sustainability', or 'brsr'.\n\n"
                        "At what time daily would you like to receive it?\n"
                        f"Please enter the time in 24-hour HH:MM format (e.g., 08:00 or 20:30).\n"
                        f"Note: The bot's current time is {current_time}."
                    )
            else:
                await update.message.reply_text(
                    "Incorrect password. Please try again, or type /cancel to abort."
                )
            return
            
        elif state == "AWAITING_ORIGIN":
            # Search location matches
            processing_msg = await update.message.reply_text("Searching for origin location...")
            results, err = search_locations(user_text)
            if err:
                await processing_msg.edit_text(f"Error searching location: {err}. Please try again.")
                return
            if not results:
                await processing_msg.edit_text(f"Could not find any locations matching '{user_text}'. Please try again or be more specific.")
                return
                
            if len(results) == 1:
                setup["origin_name"] = results[0]["display_name"]
                setup["origin_lat"] = results[0]["lat"]
                setup["origin_lon"] = results[0]["lon"]
                setup["state"] = "AWAITING_DESTINATION"
                await processing_msg.edit_text(
                    f"Origin set to: {results[0]['display_name']}\n\n"
                    "Now, please enter your destination (arrival location), or type /cancel to abort."
                )
            else:
                setup["options"] = results
                setup["state"] = "SELECTING_ORIGIN"
                reply_markup = build_selection_keyboard(results)
                await processing_msg.delete() # delete search message
                await update.message.reply_text(
                    f"Multiple matches found for '{user_text}'. Please select the correct origin:",
                    reply_markup=reply_markup
                )
            return
            
        elif state == "AWAITING_DESTINATION":
            # Search location matches
            processing_msg = await update.message.reply_text("Searching for destination location...")
            results, err = search_locations(user_text)
            if err:
                await processing_msg.edit_text(f"Error searching location: {err}. Please try again.")
                return
            if not results:
                await processing_msg.edit_text(f"Could not find any locations matching '{user_text}'. Please try again or be more specific.")
                return
                
            if len(results) == 1:
                setup["destination_name"] = results[0]["display_name"]
                setup["destination_lat"] = results[0]["lat"]
                setup["destination_lon"] = results[0]["lon"]
                setup["state"] = "AWAITING_TIME"
                current_time = now_ist().strftime("%H:%M")
                await processing_msg.edit_text(
                    f"Destination set to: {results[0]['display_name']}\n\n"
                    f"Finally, please enter the daily scheduled time in 24-hour HH:MM format (e.g., 08:30 or 17:45).\n"
                    f"Note: The bot's current time is {current_time}."
                )
            else:
                setup["options"] = results
                setup["state"] = "SELECTING_DESTINATION"
                reply_markup = build_selection_keyboard(results)
                await processing_msg.delete() # delete search message
                await update.message.reply_text(
                    f"Multiple matches found for '{user_text}'. Please select the correct destination:",
                    reply_markup=reply_markup
                )
            return
            
        elif state == "AWAITING_TIME":
            time_input = user_text.strip()
            try:
                parts = time_input.split(":")
                if len(parts) != 2:
                    raise ValueError
                h = int(parts[0])
                m = int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError
                formatted_time = f"{h:02d}:{m:02d}"
            except ValueError:
                await update.message.reply_text(
                    "Invalid time format. Please enter the time in HH:MM format (24-hour, e.g., 08:30 or 17:45)."
                )
                return
                
            origin_name = setup["origin_name"]
            origin_lat = setup["origin_lat"]
            origin_lon = setup["origin_lon"]
            dest_name = setup["destination_name"]
            dest_lat = setup["destination_lat"]
            dest_lon = setup["destination_lon"]
            
            processing_msg = await update.message.reply_text("Saving route and checking travel details...")
            res, err = get_route_info_by_coords(origin_lat, origin_lon, dest_lat, dest_lon, origin_name, dest_name)
            if err:
                logging.error(f"Error checking route details: {err}")
                add_route(chat_id, origin_name, origin_lat, origin_lon, dest_name, dest_lat, dest_lon, formatted_time)
                del user_route_setups[chat_id]
                await processing_msg.edit_text(
                    f"Successfully scheduled daily route alert for {formatted_time}!\n\n"
                    f"Route: {origin_name} ➡️ {dest_name}\n\n"
                    f"Warning: Could not fetch initial distance/ETA details: {err}. We will try again when the daily alert triggers."
                )
            else:
                add_route(chat_id, origin_name, origin_lat, origin_lon, dest_name, dest_lat, dest_lon, formatted_time)
                del user_route_setups[chat_id]
                
                msg = (
                    f"Successfully scheduled daily route alert for {formatted_time}!\n\n"
                    f"Route: {origin_name} ➡️ {dest_name}\n"
                    f"Estimated Duration: {res['duration']}\n"
                    f"Shortest Distance: {res['distance']}\n\n"
                    f"Google Maps Link: {res['map_link']}"
                )
                await processing_msg.edit_text(msg)
            return

        elif state == "AWAITING_GRIND_TIME":
            time_input = user_text.strip()
            try:
                parts = time_input.split(":")
                if len(parts) != 2:
                    raise ValueError
                h = int(parts[0])
                m = int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError
                formatted_time = f"{h:02d}:{m:02d}"
            except ValueError:
                await update.message.reply_text(
                    "Invalid time format. Please enter the time in HH:MM format (24-hour, e.g., 08:30 or 17:45)."
                )
                return
                
            add_grind_alert(chat_id, formatted_time)
            del user_route_setups[chat_id]
            
            await update.message.reply_text(
                f"Successfully scheduled daily Grind Alert at {formatted_time}! 🚀\n"
                f"Every day at this time, you will receive your System Design and DSA problems study set."
            )
            return

        elif state == "AWAITING_JOB_DOMAIN":
            setup["domain"] = user_text.strip()
            setup["state"] = "AWAITING_JOB_EXPERIENCE"
            await update.message.reply_text(
                f"Domain: {setup['domain']} ✓\n\n"
                "What is your experience level? (e.g., Fresher, 1-2 years, 3-5 years, Senior, 8+ years)\n\n"
                "Type /cancel to abort."
            )
            return

        elif state == "AWAITING_JOB_EXPERIENCE":
            setup["experience"] = user_text.strip()
            setup["state"] = "AWAITING_JOB_LOCATION"
            await update.message.reply_text(
                f"Experience: {setup['experience']} ✓\n\n"
                "What is your preferred work location or remote preference?\n"
                "(e.g., Remote, Bangalore, New York, Hybrid - London, Any)\n\n"
                "Type /cancel to abort."
            )
            return

        elif state == "AWAITING_JOB_LOCATION":
            setup["location"] = user_text.strip()
            setup["state"] = "AWAITING_JOB_FRESHNESS"
            await update.message.reply_text(
                f"Location: {setup['location']} ✓\n\n"
                "How recent should the job postings be?\n"
                "Reply with one of the following options:\n"
                "  1 - Past 1 hour\n"
                "  2 - Past 2 hours\n"
                "  3 - Past 8 hours\n"
                "  4 - Past 24 hours\n"
                "  5 - Past 2 days\n"
                "  6 - Past week\n"
                "  7 - Past month\n\n"
                "Type /cancel to abort."
            )
            return

        elif state == "AWAITING_JOB_FRESHNESS":
            freshness_map = {
                "1": 3600,
                "2": 7200,
                "3": 28800,
                "4": 86400,
                "5": 172800,
                "6": 604800,
                "7": 2592000
            }
            freshness_labels = {
                "1": "Past 1 hour", "2": "Past 2 hours", "3": "Past 8 hours",
                "4": "Past 24 hours", "5": "Past 2 days", "6": "Past week", "7": "Past month"
            }
            choice = user_text.strip()
            if choice not in freshness_map:
                await update.message.reply_text(
                    "Invalid choice. Please reply with a number from 1 to 7.\n"
                    "  1 - Past 1 hour\n  2 - Past 2 hours\n  3 - Past 8 hours\n"
                    "  4 - Past 24 hours\n  5 - Past 2 days\n  6 - Past week\n  7 - Past month"
                )
                return
            setup["freshness_seconds"] = freshness_map[choice]
            setup["freshness_label"] = freshness_labels[choice]
            setup["state"] = "AWAITING_JOB_TIME"
            current_time = now_ist().strftime("%H:%M")
            await update.message.reply_text(
                f"Freshness: {freshness_labels[choice]} ✓\n\n"
                "At what time daily would you like to receive your job listings?\n"
                f"Please enter the time in 24-hour HH:MM format (e.g., 09:00 or 18:00).\n"
                f"Note: The bot's current time is {current_time}.\n\n"
                "Type /cancel to abort."
            )
            return

        elif state == "AWAITING_JOB_TIME":
            time_input = user_text.strip()
            try:
                parts = time_input.split(":")
                if len(parts) != 2:
                    raise ValueError
                h = int(parts[0])
                m = int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError
                formatted_time = f"{h:02d}:{m:02d}"
            except ValueError:
                await update.message.reply_text(
                    "Invalid time format. Please enter the time in HH:MM format (e.g., 09:00)."
                )
                return

            add_job_alert(chat_id, setup["domain"], setup["experience"], setup["location"], setup.get("freshness_seconds", 86400), formatted_time)
            del user_route_setups[chat_id]

            await update.message.reply_text(
                f"Daily Job Alert scheduled for {formatted_time}! 💼\n\n"
                f"Domain: {setup['domain']}\n"
                f"Experience: {setup['experience']}\n"
                f"Location: {setup['location']}\n"
                f"Freshness: {setup.get('freshness_label', 'Past 24 hours')}\n\n"
                f"Every day at this time you will receive live scraped job listings from LinkedIn."
            )
            return

        elif state == "AWAITING_BLR_TIME":
            time_input = user_text.strip()
            try:
                parts = time_input.split(":")
                if len(parts) != 2:
                    raise ValueError
                h = int(parts[0])
                m = int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError
                formatted_time = f"{h:02d}:{m:02d}"
            except ValueError:
                await update.message.reply_text(
                    "Invalid time format. Please enter the time in HH:MM format (24-hour, e.g., 08:30 or 17:45)."
                )
                return

            add_blrhomecoming_alert(chat_id, formatted_time)
            del user_route_setups[chat_id]

            await update.message.reply_text(
                f"Successfully scheduled daily BLR Homecoming Alert at {formatted_time}! 🌱\n"
                f"Every day at this time, you'll get ESG/sustainability/BRSR role matches scraped from the tracked employer list."
            )
            return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = None
    for attempt in range(3):
        try:
            result = await chat_graph.ainvoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={"configurable": {"thread_id": chat_id}}
            )
            response = result["messages"][-1]
            break
        except Exception as e:
            err_msg = str(e)
            logging.error(f"Error executing LangGraph pipeline (attempt {attempt+1}): {e}")
            if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower() or "rate_limit" in err_msg.lower() or "limit" in err_msg.lower()) and attempt < 2:
                if switch_api_key():
                    continue
            await update.message.reply_text("Sorry, I encountered an error while processing your request.")
            return

    try:
        content = response.content
        if isinstance(content, list):
            reply_text = "".join(
                part.get("text", "") if isinstance(part, dict) and part.get("type") == "text" else str(part)
                for part in content
            )
        else:
            reply_text = str(content)
            
        cleaned_lines = []
        for line in reply_text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("* "):
                indent = line[:len(line) - len(stripped)]
                line = indent + "• " + stripped[2:]
            line = line.replace("*", "")
            cleaned_lines.append(line)
        reply_text = "\n".join(cleaned_lines)

        await update.message.reply_text(reply_text)
    except Exception as e:
        logging.error(f"Error formatting response content: {e}")
        await update.message.reply_text("Sorry, I encountered an error while formatting the response.")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Bot is active")

class PingHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "text/plain")
        self.set_header("Content-Length", "2")
        self.write("OK")

    def head(self):
        self.set_header("Content-Type", "text/plain")
        self.set_header("Content-Length", "2")
        self.finish()

class WebhookHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = tornado.escape.json_decode(self.request.body)
            update = Update.de_json(data, app.bot)
            await app.process_update(update)
            self.write("OK")
        except Exception as e:
            logging.error(f"Error processing update: {e}")
            self.set_status(400)
            self.write("Error")

    def get(self):
        self.write("Webhook endpoint is active")

app = None

async def main():
    global app
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        raise ValueError("Missing environment variables: TELEGRAM_TOKEN or GEMINI_API_KEY")

    # Initialize DB
    routes_db.init_db()
    await init_chat_graph()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("makerouteasy", makerouteasy))
    app.add_handler(CommandHandler("cancel", cancel_route_setup))
    app.add_handler(CommandHandler("myroutes", list_routes))
    app.add_handler(CommandHandler("deleteroute", delete_route))
    app.add_handler(CommandHandler("grindalert", grindalert))
    app.add_handler(CommandHandler("mygrinds", list_grinds))
    app.add_handler(CommandHandler("deletegrind", delete_grind))
    app.add_handler(CommandHandler("jobalert", jobalert))
    app.add_handler(CommandHandler("myjobs", list_jobs))
    app.add_handler(CommandHandler("deletejob", delete_job))
    app.add_handler(CommandHandler("blrhomecoming", blrhomecoming))
    app.add_handler(CommandHandler("myblrhomecoming", list_blrhomecoming))
    app.add_handler(CommandHandler("deleteblrhomecoming", delete_blrhomecoming))
    app.add_handler(CommandHandler("myalerts", list_all_alerts))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    await app.initialize()
    await app.start()

    # Start background scheduler loop
    asyncio.create_task(scheduler_loop())

    external_url = os.getenv("RENDER_EXTERNAL_URL")
    port = os.getenv("PORT", "8080")

    if not external_url:
        raise ValueError("Missing environment variable: RENDER_EXTERNAL_URL (required for Telegram Webhook)")

    port_int = int(port)
    
    # Configure Tornado application to serve GET and POST
    tornado_app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/webhook", WebhookHandler),
        (r"/ping", PingHandler),
    ])
    tornado_app.listen(port_int)
    logging.info(f"Starting Tornado server on port {port_int}...")
    logging.info(f"Webhook URL configured: {external_url}/webhook")

    # Set webhook on Telegram
    await app.bot.set_webhook(url=f"{external_url}/webhook")
    logging.info("Telegram webhook set successfully.")

    # Keep the loop running
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logging.info("Shutting down bot and server...")
        await app.stop()
        await app.shutdown()
        await close_chat_graph()

if __name__ == '__main__':
    asyncio.run(main())