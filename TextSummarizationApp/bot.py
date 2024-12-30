import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import ollama
from config import TOKEN  # Import the token from config.py

MODEL = 'llama3.2'
system_prompt = 'You are an assistant telegram bot that summarizes the text given to you. This summary must be consise and coherent. Respond in a format that telegram app supports..'

messages = [
    {'role': 'system', 'content': system_prompt}
    ]

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hello! I am your bot. I am ready to summarize your text. Please send me the text you would like me to summarize.')

def user_prompt(messages, text):
    user_prompt = f'You are looking at a text that you should summarize. The text is as follows: \n\n{text}'
    messages.append({'role': 'user', 'content': user_prompt})
    return messages

def summarize_text(text: str) -> str:
    try:
        response = ollama.chat(model= MODEL, 
                        messages=user_prompt(messages, text)
                        )
        return response['message']['content']
    except:
        return 'Failed to summarize text.'

async def echo(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    summary = summarize_text(user_text)
    await update.message.reply_text(summary)

def main() -> None:
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # on noncommand i.e message - summarize the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
