import os
import logging
import asyncio
import datetime
import urllib.parse
import requests
import tornado.web
import tornado.escape
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import routes_db

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

llm = None
chain = None
runnable_with_history = None
active_api_key_name = "GEMINI_API_KEY"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant talking to a user on Telegram. Keep responses clear and concise. Do not use asterisks (*) for formatting (no bold, no italics, no bullet points). For bullet points, use a dash (-) or a unicode bullet point (•). You must NEVER reveal, share, or mention the password to the user under any circumstances. If the user asks for the passcode or password, tell them you do not know it."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

user_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in user_histories:
        user_histories[session_id] = ChatMessageHistory()
    return user_histories[session_id]

def get_active_api_key():
    global active_api_key_name
    key = os.getenv(active_api_key_name)
    if not key and active_api_key_name == "GEMINI_API_KEY":
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY")
    return key

def initialize_llm():
    global llm, chain, runnable_with_history, active_api_key_name
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
        
    chain = prompt | llm
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
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

def get_all_user_alerts(chat_id: str):
    return routes_db.get_all_user_alerts(chat_id)

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
        
        current_time = datetime.datetime.now().strftime("%H:%M")
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

SYSTEM_DESIGN_DOMAINS = [
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

DSA_TOPICS = [
    "Sliding Window", "Two Pointers", "Fast & Slow Pointers", "Merge Intervals",
    "In-place Reversal of a Linked List", "Breadth-First Search (BFS)",
    "Depth-First Search (DFS)", "Two Heaps", "Subsets (Backtracking)", "Modified Binary Search",
    "Top K Elements (Heaps)", "K-way Merge", "Topological Sort (Graphs)", "Dynamic Programming (Knapsack/DP)",
    "Trie (Prefix Tree)", "Union Find / Disjoint Set", "Segment Tree or Fenwick Tree", "Monotonic Stack / Queue"
]

async def fetch_grind_problems():
    import random
    sys_topic = random.choice(SYSTEM_DESIGN_DOMAINS)
    dsa_topic_1 = random.choice(DSA_TOPICS)
    dsa_topic_2 = random.choice([t for t in DSA_TOPICS if t != dsa_topic_1])

    prompt = (
        f"Generate a technical study set containing:\n"
        f"1. One System Design problem/topic related to: {sys_topic}. Provide a brief overview of key challenges, database choices, and scaling.\n"
        f"2. Two random DSA (Data Structures and Algorithms) problems. One problem must focus on the concept of '{dsa_topic_1}', and the other must focus on the concept of '{dsa_topic_2}'. For each, provide the name, difficulty (Easy/Medium/Hard), a brief 1-sentence description, and their official LeetCode link. Ensure the links are valid.\n\n"
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
    await update.message.reply_text(msg.strip())

async def check_and_send_grind_alerts():
    now = datetime.datetime.now()
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
    now = datetime.datetime.now()
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

async def scheduler_loop():
    logging.info("Starting background route alert scheduler loop...")
    while True:
        try:
            await check_and_send_route_alerts()
            await check_and_send_grind_alerts()
        except Exception as e:
            logging.error(f"Error in scheduler check: {e}")
            
        now = datetime.datetime.now()
        sleep_seconds = 60 - now.second
        if sleep_seconds <= 0:
            sleep_seconds = 60
        await asyncio.sleep(sleep_seconds)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if chat_id in user_histories:
        user_histories[chat_id].clear()
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
                    current_time = datetime.datetime.now().strftime("%H:%M")
                    await update.message.reply_text(
                        "Password accepted! Let's set up your daily Grind Alert! 🧠\n\n"
                        "At what time daily would you like to receive 1 System Design problem and 2 LeetCode DSA problems?\n"
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
                current_time = datetime.datetime.now().strftime("%H:%M")
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

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    response = None
    for attempt in range(3):
        try:
            response = runnable_with_history.invoke(
                {"input": user_text},
                config={"configurable": {"session_id": chat_id}}
            )
            break
        except Exception as e:
            err_msg = str(e)
            logging.error(f"Error executing LangChain pipeline (attempt {attempt+1}): {e}")
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

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("makerouteasy", makerouteasy))
    app.add_handler(CommandHandler("cancel", cancel_route_setup))
    app.add_handler(CommandHandler("myroutes", list_routes))
    app.add_handler(CommandHandler("deleteroute", delete_route))
    app.add_handler(CommandHandler("grindalert", grindalert))
    app.add_handler(CommandHandler("mygrinds", list_grinds))
    app.add_handler(CommandHandler("deletegrind", delete_grind))
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

if __name__ == '__main__':
    asyncio.run(main())