import logging
import sys
import hashlib

import openai

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from getpass import getpass

from rich.console import Console

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
)
log = logging.getLogger(__name__)

console = Console()

# SQLAlchemy setup
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password_hash = Column(String)


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    chat_room = Column(String)
    sender = Column(String)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.now)


# SQLite database connection
engine = create_engine('sqlite:///chat.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user():
    username = input("Enter new username: ")
    if session.query(User).filter_by(username=username).first():
        console.log(
            "Username already exists. Please try a different username.",
            style="bold red"
        )
        return None

    password = getpass("Enter new password: ")
    hashed_password = hash_password(password)
    new_user = User(username=username, password_hash=hashed_password)
    session.add(new_user)
    session.commit()
    return username


def login_user():
    username = input("Enter username: ")
    password = getpass("Enter password: ")
    hashed_password = hash_password(password)

    user = session.query(User).filter_by(username=username, password_hash=hashed_password).first()
    if user:
        return username
    else:
        console.log("Invalid username or password.", style="bold red")
        return None


def save_message(chat_room, sender, message):
    new_message = Message(chat_room=chat_room, sender=sender, message=message)
    session.add(new_message)
    session.commit()


def get_chat_history(chat_room):
    messages = session.query(Message).filter_by(chat_room=chat_room).order_by(Message.timestamp).all()
    return [f"{message.sender}: {message.message}" for message in messages]


def get_gpt4_response(prompt, chat_history):
    # openai.api_key = 'your-api-key'  # Replace with your actual OpenAI API key

    combined_prompt = "\n".join(chat_history[-50:]) + f"\n{prompt}"  # Limit history to last 50 messages
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_prompt}
        ]
    )
    return response.choices[0].message.content


def run_chat(username):
    chat_room = username
    console.log(
        f"Welcome to your personal GPT-4 Chat CLI, {username}. Type 'quit' to exit.",
        style="bold blue"
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        chat_context = get_chat_history(chat_room)
        gpt_response = get_gpt4_response(user_input, chat_context)

        save_message(chat_room, "You", user_input)
        save_message(chat_room, "GPT-4", gpt_response)

        console.log(f"GPT-4: {gpt_response}", style="bold green")

    console.log("Chat ended.", style="bold blue")


def main():
    log.info("Welcome to the Chat Application")
    choice = input("Do you want to [L]ogin or [R]egister? (L/R): ").lower()

    username = None
    while not username:
        if choice == 'r':
            username = register_user()
        elif choice == 'l':
            username = login_user()
        else:
            choice = input("Please enter 'L' to login or 'R' to register: ").lower()

    if username:
        run_chat(username)


if __name__ == "__main__":
    main()
