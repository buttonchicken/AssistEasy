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

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant talking to a user on Telegram. Keep responses clear and concise. Do not use asterisks (*) for formatting (no bold, no italics, no bullet points). For bullet points, use a dash (-) or a unicode bullet point (•)."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

user_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in user_histories:
        user_histories[session_id] = ChatMessageHistory()
    return user_histories[session_id]

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Stateful route setup tracking
user_route_setups = {}

# In-memory route storage
user_routes = []
route_id_counter = 1

def add_route(chat_id: str, origin: str, destination: str, scheduled_time: str) -> int:
    global route_id_counter
    route = {
        "id": route_id_counter,
        "chat_id": chat_id,
        "origin": origin,
        "destination": destination,
        "scheduled_time": scheduled_time,
        "last_sent": None
    }
    user_routes.append(route)
    route_id_counter += 1
    return route["id"]

def get_user_routes(chat_id: str):
    return [r for r in user_routes if r["chat_id"] == chat_id]

def delete_user_route(chat_id: str, route_id: int) -> bool:
    global user_routes
    initial_len = len(user_routes)
    user_routes = [r for r in user_routes if not (r["chat_id"] == chat_id and r["id"] == route_id)]
    return len(user_routes) < initial_len

def get_routes_to_trigger(current_time: str, current_date: str):
    return [
        r for r in user_routes
        if r["scheduled_time"] == current_time and (r["last_sent"] is None or r["last_sent"] != current_date)
    ]

def update_last_sent(route_id: int, current_date: str):
    for r in user_routes:
        if r["id"] == route_id:
            r["last_sent"] = current_date
            break

def get_route_info(origin: str, destination: str):
    # Google Maps Directions URL (works without API key)
    encoded_origin = urllib.parse.quote_plus(origin)
    encoded_dest = urllib.parse.quote_plus(destination)
    map_link = f"https://www.google.com/maps/dir/?api=1&origin={encoded_origin}&destination={encoded_dest}"
    
    # Check if Google Maps API key is available
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        try:
            url = f"https://maps.googleapis.com/maps/api/directions/json?origin={encoded_origin}&destination={encoded_dest}&key={api_key}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "OK" and data.get("routes"):
                leg = data["routes"][0]["legs"][0]
                distance = leg["distance"]["text"]
                duration = leg["duration"]["text"]
                return {
                    "distance": distance,
                    "duration": duration,
                    "map_link": map_link
                }, None
            else:
                logging.warning(f"Google Maps API returned status: {data.get('status')}. Falling back to OSRM.")
        except Exception as e:
            logging.error(f"Error querying Google Maps API: {e}. Falling back to OSRM.")

    # Fallback: Nominatim for geocoding + OSRM for routing
    headers = {"User-Agent": "AssistEasyBot/1.0"}
    
    # Geocode origin
    origin_url = f"https://nominatim.openstreetmap.org/search?q={encoded_origin}&format=json&limit=1"
    try:
        r = requests.get(origin_url, headers=headers, timeout=10)
        r.raise_for_status()
        origin_data = r.json()
        if not origin_data:
            return None, f"Could not find origin: {origin}"
        origin_lat = origin_data[0]["lat"]
        origin_lon = origin_data[0]["lon"]
    except Exception as e:
        return None, f"Error geocoding origin: {e}"
        
    # Geocode destination
    dest_url = f"https://nominatim.openstreetmap.org/search?q={encoded_dest}&format=json&limit=1"
    try:
        r = requests.get(dest_url, headers=headers, timeout=10)
        r.raise_for_status()
        dest_data = r.json()
        if not dest_data:
            return None, f"Could not find destination: {destination}"
        dest_lat = dest_data[0]["lat"]
        dest_lon = dest_data[0]["lon"]
    except Exception as e:
        return None, f"Error geocoding destination: {e}"
        
    # Route via OSRM
    route_url = f"http://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=false"
    try:
        r = requests.get(route_url, timeout=10)
        r.raise_for_status()
        route_data = r.json()
        if "routes" not in route_data or not route_data["routes"]:
            return None, "No route found between these locations"
            
        route = route_data["routes"][0]
        distance_meters = route["distance"]
        duration_seconds = route["duration"]
        
        # Format distance
        distance_km = distance_meters / 1000.0
        distance_str = f"{distance_km:.1f} km"
        
        # Format duration
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

async def makerouteasy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_route_setups[chat_id] = {
        "state": "AWAITING_ORIGIN",
        "origin": None,
        "destination": None
    }
    await update.message.reply_text(
        "Let's set up your daily route alert! 🚗\n\n"
        "First, please enter your origin (departure location), or type /cancel to abort."
    )

async def cancel_route_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if chat_id in user_route_setups:
        del user_route_setups[chat_id]
        await update.message.reply_text("Route setup has been cancelled.")
    else:
        await update.message.reply_text("No active route setup to cancel.")

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
        
        res, err = get_route_info(origin, destination)
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
        except Exception as e:
            logging.error(f"Error in check_and_send_route_alerts: {e}")
            
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
        
        if state == "AWAITING_ORIGIN":
            setup["origin"] = user_text
            setup["state"] = "AWAITING_DESTINATION"
            await update.message.reply_text(
                f"Origin set to: {user_text}\n\n"
                "Now, please enter your destination (arrival location), or type /cancel to abort."
            )
            return
            
        elif state == "AWAITING_DESTINATION":
            setup["destination"] = user_text
            setup["state"] = "AWAITING_TIME"
            current_time = datetime.datetime.now().strftime("%H:%M")
            await update.message.reply_text(
                f"Destination set to: {user_text}\n\n"
                f"Finally, please enter the daily scheduled time in 24-hour HH:MM format (e.g., 08:30 or 17:45).\n"
                f"Note: The bot's current time is {current_time}."
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
                
            origin = setup["origin"]
            destination = setup["destination"]
            
            processing_msg = await update.message.reply_text("Saving route and checking initial travel details...")
            res, err = get_route_info(origin, destination)
            if err:
                logging.error(f"Error checking route details: {err}")
                add_route(chat_id, origin, destination, formatted_time)
                del user_route_setups[chat_id]
                await processing_msg.edit_text(
                    f"Successfully scheduled daily route alert for {formatted_time}!\n\n"
                    f"Route: {origin} ➡️ {destination}\n\n"
                    f"Warning: Could not fetch initial distance/ETA details: {err}. We will try again when the daily alert triggers."
                )
            else:
                add_route(chat_id, origin, destination, formatted_time)
                del user_route_setups[chat_id]
                
                msg = (
                    f"Successfully scheduled daily route alert for {formatted_time}!\n\n"
                    f"Route: {origin} ➡️ {destination}\n"
                    f"Estimated Duration: {res['duration']}\n"
                    f"Shortest Distance: {res['distance']}\n\n"
                    f"Google Maps Link: {res['map_link']}"
                )
                await processing_msg.edit_text(msg)
            return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        response = runnable_with_history.invoke(
            {"input": user_text},
            config={"configurable": {"session_id": chat_id}}
        )
        
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
        logging.error(f"Error executing LangChain pipeline: {e}")
        await update.message.reply_text("Sorry, I encountered an error while processing your request.")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Bot is active")

class PingHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("OK")

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

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("makerouteasy", makerouteasy))
    app.add_handler(CommandHandler("cancel", cancel_route_setup))
    app.add_handler(CommandHandler("myroutes", list_routes))
    app.add_handler(CommandHandler("deleteroute", delete_route))
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