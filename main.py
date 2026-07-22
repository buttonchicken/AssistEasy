import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
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
    ("system", "You are a helpful AI assistant talking to a user on Telegram. Keep responses clear and concise."),
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if chat_id in user_histories:
        user_histories[chat_id].clear()
    await update.message.reply_text("Hello! I'm your Gemini AI assistant powered by LangChain. How can I help you today?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = str(update.effective_chat.id)
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
            
        await update.message.reply_text(reply_text)
    except Exception as e:
        logging.error(f"Error executing LangChain pipeline: {e}")
        await update.message.reply_text("Sorry, I encountered an error while processing your request.")

class DummyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is active")
        
    def log_message(self, format, *args):
        return

def run_dummy_server():
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), DummyServer)
    logging.info(f"Starting dummy web server on port {port} for Render health checks...")
    server.serve_forever()

if __name__ == '__main__':
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    
    if not TELEGRAM_TOKEN or not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Missing environment variables: TELEGRAM_TOKEN or GEMINI_API_KEY")

    # Start a dummy web server in a background thread to satisfy Render's health checks
    port = os.getenv("PORT")
    if port:
        threading.Thread(target=run_dummy_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("Bot is running with LangChain integration...")
    app.run_polling()