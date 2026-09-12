import os
import re
import gradio as gr
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. CUSTOMER SUPPORT TRAINING DATA
# ==========================================

data = {
    "intent": [
        "order_status",
        "order_status",
        "order_status",

        "return_product",
        "return_product",
        "return_product",

        "refund",
        "refund",
        "refund",

        "payment",
        "payment",
        "payment",

        "password",
        "password",
        "password",

        "human_support",
        "human_support",

        "greeting",
        "greeting",
        "greeting"
    ],

    "question": [
        "Where is my order",
        "How can I track my order",
        "What is the status of my delivery",

        "I want to return my product",
        "How can I return an item",
        "Can I return my order",

        "How can I get a refund",
        "When will I receive my refund",
        "I want my money back",

        "What payment methods do you accept",
        "Can I pay using UPI",
        "Do you accept credit cards",

        "I forgot my password",
        "How can I reset my password",
        "I cannot login to my account",

        "I want to talk to a human",
        "Connect me with customer support",

        "Hello",
        "Hi",
        "Good morning"
    ],

    "response": [
        "You can track your order from the Orders section of your account.",
        "You can track your order from the Orders section of your account.",
        "You can track your order from the Orders section of your account.",

        "You can request a return from the Orders section within 7 days of delivery.",
        "You can request a return from the Orders section within 7 days of delivery.",
        "You can request a return from the Orders section within 7 days of delivery.",

        "Refunds are normally processed within 5 to 7 business days.",
        "Refunds are normally processed within 5 to 7 business days.",
        "Refunds are normally processed within 5 to 7 business days.",

        "We accept UPI, credit cards, debit cards and net banking.",
        "We accept UPI, credit cards, debit cards and net banking.",
        "We accept UPI, credit cards, debit cards and net banking.",

        "You can reset your password using the Forgot Password option on the login page.",
        "You can reset your password using the Forgot Password option on the login page.",
        "You can reset your password using the Forgot Password option on the login page.",

        "Sure. I can help connect you with customer support.",
        "Sure. I can help connect you with customer support.",

        "Hello! How can I help you today?",
        "Hello! How can I help you today?",
        "Hello! How can I help you today?"
    ]
}


df = pd.DataFrame(data)


# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_question"] = df["question"].apply(preprocess_text)


# ==========================================
# 3. TRAIN NLP MODEL
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

question_vectors = vectorizer.fit_transform(
    df["clean_question"]
)


# ==========================================
# 4. CHATBOT FUNCTION
# ==========================================

def chatbot(message, history):

    if not message or not message.strip():
        return "Please enter a question."

    clean_message = preprocess_text(message)

    message_vector = vectorizer.transform(
        [clean_message]
    )

    similarities = cosine_similarity(
        message_vector,
        question_vectors
    )[0]

    best_index = similarities.argmax()

    best_score = similarities[best_index]


    # ======================================
    # 5. FALLBACK LOGIC
    # ======================================

    if best_score < 0.25:

        return (
            "I'm sorry, I don't fully understand your question. "
            "Could you please rephrase it? "
            "You can ask me about orders, returns, refunds, "
            "payments, passwords, or customer support."
        )


    # ======================================
    # 6. GET BEST RESPONSE
    # ======================================

    response = df.iloc[best_index]["response"]

    return response


# ==========================================
# 7. CHATBOT USER INTERFACE
# ==========================================

demo = gr.ChatInterface(
    fn=chatbot,

    title="🤖 Intelligent Customer Support Chatbot",

    description=(
        "Ask questions about orders, returns, refunds, "
        "payments, passwords and customer support."
    ),

    examples=[
        "Where is my order?",
        "I want to return my product",
        "How can I get a refund?",
        "Can I pay using UPI?",
        "I forgot my password",
        "I want to talk to a human"
    ]
)


# ==========================================
# 8. RENDER SERVER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
