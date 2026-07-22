import os
import logging
import asyncio
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

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Bot is active")

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
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    await app.initialize()
    await app.start()

    external_url = os.getenv("RENDER_EXTERNAL_URL")
    port = os.getenv("PORT", "8080")

    if not external_url:
        raise ValueError("Missing environment variable: RENDER_EXTERNAL_URL (required for Telegram Webhook)")

    port_int = int(port)
    
    # Configure Tornado application to serve GET and POST
    tornado_app = tornado.web.Application([
        (r"/", MainHandler),
        (r"/webhook", WebhookHandler),
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