import os
import csv
import sqlite3
import base64
import gradio as gr
import json
import numpy as np
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent

# Load environment variables
load_dotenv(HERE / ".env")
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("LLM_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

EMBED_API_URL = os.getenv("EMBEDDING_API_URL")
EMBED_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME")


# Normaler LLM-Client für den Chat
client = OpenAI(
    api_key=API_KEY,
    base_url=API_ENDPOINT,
    timeout=60.0
)


# Separater Client für Embeddings / RAG
embed_client = OpenAI(
    api_key=EMBED_API_KEY,
    base_url=EMBED_API_URL,
    timeout=60.0
)

models = embed_client.models.list()

for model in models.data:
    print(model.id)

    
DB_PATH = HERE / "cafe.db"
RULES_FILE = HERE / "cafe_rules.txt"
WISDOMS_FILE = HERE / "wisdoms.txt"

def embed(texts):
    response = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    return np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )
def load_rule_chunks():
    """Read cafe_rules.txt and split it into sections."""
    chunks = []

    with open(RULES_FILE, encoding="utf-8") as f:
        content = f.read()

    for part in content.split("## "):
        part = part.strip()

        if not part:
            continue

        title, _, text = part.partition("\n")

        chunks.append({
            "title": title.strip(),
            "text": text.strip()
        })

    return chunks


CHUNKS = load_rule_chunks()

CHUNK_VECTORS = embed([
    f"{chunk['title']}\n{chunk['text']}"
    for chunk in CHUNKS
])

MIN_SCORE = 0.45


def search_cafe_rules(question, k=1):
    """Search the cafe rules for sections related to the question."""

    q = embed([question])[0]

    scores = CHUNK_VECTORS @ q / (
        np.linalg.norm(CHUNK_VECTORS, axis=1)
        * np.linalg.norm(q)
    )

    best = [
        i
        for i in np.argsort(scores)[::-1][:k]
        if scores[i] >= MIN_SCORE
    ]

    if not best:
        return {
            "found": False,
            "note": "The cafe rules do not cover this question."
        }

    return {
        "found": True,
        "chunks": [
            {
                "title": CHUNKS[i]["title"],
                "text": CHUNKS[i]["text"],
                "score": round(float(scores[i]), 3)
            }
            for i in best
        ]
    }

def build_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS cafe_info")
    cursor.execute("DROP TABLE IF EXISTS menu")

    cursor.execute("""
        CREATE TABLE cafe_info (
            topic TEXT PRIMARY KEY,
            content TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE menu (
            name TEXT PRIMARY KEY,
            category TEXT,
            price REAL,
            available TEXT
        )
    """)

    with open(HERE / "cafe_info.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute(
                "INSERT INTO cafe_info (topic, content) VALUES (?, ?)",
                (row["topic"], row["content"])
            )

    with open(HERE / "menu.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute(
                """
                INSERT INTO menu (name, category, price, available)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row["name"],
                    row["category"],
                    float(row["price"]),
                    row["available"]
                )
            )

    conn.commit()
    conn.close()


build_database()
def get_cafe_info(topic):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT content FROM cafe_info WHERE topic = ?",
        (topic,)
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"found": False, "topic": topic}

    return {
        "found": True,
        "topic": topic,
        "content": row[0]
    }


def search_menu(query):
    if not query or not query.strip():
        return {
            "found": False,
            "query": query,
            "items": [],
            "error": "Please provide a menu item or category to search for."
        }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name, category, price, available
        FROM menu
        WHERE LOWER(name) LIKE LOWER(?)
        OR LOWER(category) LIKE LOWER(?)
        """,
        (f"%{query}%", f"%{query}%")
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "found": False,
            "query": query,
            "items": []
        }

    items = []

    for row in rows:
        items.append({
            "name": row[0],
            "category": row[1],
            "price": row[2],
            "available": row[3]
        })

    return {
        "found": True,
        "query": query,
        "items": items
    }


def calculate_order_total(items):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total = 0
    details = []

    for item in items:
        name = item["name"]
        quantity = item["quantity"]

        cursor.execute(
            """
            SELECT price, available
            FROM menu
            WHERE LOWER(name) = LOWER(?)
            """,
            (name,)
        )

        row = cursor.fetchone()

        if row is None:
            details.append({
                "name": name,
                "error": "Item not found"
            })
            continue

        price = row[0]
        available = row[1]

        if available.lower() != "yes":
            details.append({
                "name": name,
                "error": "Item is not available"
            })
            continue

        subtotal = price * quantity
        total += subtotal

        details.append({
            "name": name,
            "quantity": quantity,
            "price_each": price,
            "subtotal": subtotal
        })

    conn.close()

    return {
        "items": details,
        "total": total
    }


def transfer_to_human(reason):
    
    return {
        "transfer": True,
        "reason": reason,
        "message": "A staff member should handle this request."
    }

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cafe_info",
            "description": (
                "Look up general information about the cafe, such as opening hours, "
                "address, payment, parking, takeaway or student discounts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "The information topic. Examples: opening_hours, address, "
                            "payment, parking, takeaway, student_discount."
                        )
                    }
                },
                "required": ["topic"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": (
                "Search the cafe menu for a drink, food item or category and return "
                "matching items with price and availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The item or category to search for, for example coffee, "
                            "tea, cake, latte or sandwich."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_order_total",
            "description": (
                "Calculate the total price of an order. Use this tool whenever "
                "the customer asks how much an order costs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Exact menu item name."
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "Number of this item."
                                }
                            },
                            "required": ["name", "quantity"]
                        }
                    }
                },
                "required": ["items"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_cafe_rules",
            "description": (
                "Search the cafe rules for questions about lost items, pets, "
                "laptops, study rules, allergies, birthday cakes, reservations, "
                "complaints, Wi-Fi and other cafe policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The customer's complete question."
                    }
                },
                "required": ["question"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": (
                "Transfer the customer to a human staff member when they ask to speak "
                "to a person, make a complaint, or ask something that cannot be answered "
                "from the available cafe data or cafe rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason why a human staff member is needed."
                    }
                },
                "required": ["reason"]
            }
        }
    }
]

def execute_tool(tool_name, arguments):
    if tool_name == "get_cafe_info":
        return get_cafe_info(**arguments)

    elif tool_name == "search_menu":
        return search_menu(**arguments)

    elif tool_name == "calculate_order_total":
        return calculate_order_total(**arguments)

    elif tool_name == "search_cafe_rules":
        return search_cafe_rules(**arguments)

    elif tool_name == "transfer_to_human":
        return transfer_to_human(**arguments)

    else:
        return {
            "error": "Unknown tool",
            "tool_name": tool_name
        }


    import json

SYSTEM_PROMPT = """
You are the AI customer service agent for a cafe.

Rules:
1. Answer only from tool results. Never invent prices, opening hours, availability or cafe rules.
2. Only answer questions related to the cafe.
3. Use get_cafe_info for general cafe information such as opening hours, address, payment, parking, takeaway and student discounts.
4. Use search_menu for menu questions.
5. Use calculate_order_total whenever the customer asks for the total price of an order.
6. Use search_cafe_rules for questions about cafe rules such as lost items, pets, laptops, study rules, allergies, birthday cakes, reservations, complaints and Wi-Fi.
7. For cafe rules, answer only from the result of search_cafe_rules. Never invent a cafe rule.
8. Use transfer_to_human for complaints, requests for a human, or questions that cannot be answered from the available cafe data or cafe rules.
9. Answer in the same language as the customer.
10. Keep answers short and helpful.
11. When using search_menu, never use an empty query. Search only for the specific item or category the customer asked about.
12. If a question is not covered by the available cafe information or cafe rules, do not call unrelated tools. Call transfer_to_human directly.
13. Whenever you mention a price or order total, always format it with the currency NT$, for example: NT$350.
14. When the customer asks for prices, answer clearly and concisely. Avoid repeating the same prices unnecessarily.
"""



def run_agent(user_message, chat_history=None, max_steps=5):
    if chat_history is None:
        chat_history = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for item in chat_history:
        if isinstance(item, dict):
            messages.append(item)

    messages.append({
        "role": "user",
        "content": user_message
    })

    tool_trace = []

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return {
                "answer": message.content,
                "tool_trace": tool_trace
            }

        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            result = execute_tool(tool_name, arguments)

            tool_trace.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    return {
        "answer": "Sorry, I could not complete the request.",
        "tool_trace": tool_trace
    }


def chat_with_agent(message, history):
    result = run_agent(message, history)

    answer = result["answer"]
    tool_trace = result["tool_trace"]

    trace_text = ""

    if tool_trace:
        trace_text += "\n\n---\n### Tools used\n"

        for step in tool_trace:
            trace_text += f"\n**Tool:** `{step['tool']}`\n"
            trace_text += f"**Arguments:** `{step['arguments']}`\n"

            if step["tool"] == "search_menu":
                items = step["result"].get("items", [])

                if items:
                    names = [item["name"] for item in items]
                    trace_text += f"**Result:** {', '.join(names)}\n"
                else:
                    trace_text += "**Result:** No matching items found.\n"

            elif step["tool"] == "calculate_order_total":
                total = step["result"].get("total")
                trace_text += f"**Result:** Total = {total}\n"

            elif step["tool"] == "get_cafe_info":
                content = step["result"].get(
                    "content",
                    "No information found."
                )
                trace_text += f"**Result:** {content}\n"

            elif step["tool"] == "search_cafe_rules":
                chunks = step["result"].get("chunks", [])

                if chunks:
                    for chunk in chunks:
                        trace_text += (
                            f"**RAG match:** {chunk['title']} "
                            f"(score {chunk['score']})\n"
                        )
                else:
                    trace_text += "**Result:** No matching cafe rule found.\n"

            elif step["tool"] == "transfer_to_human":
                trace_text += "**Result:** Transfer requested.\n"

            else:
                trace_text += f"**Result:** {step['result']}\n"

    return answer + trace_text

def get_background_image():
    image_path = HERE / "background.png"

    if not image_path.exists():
        return ""

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    return f"data:image/png;base64,{encoded}"


background_image = get_background_image()


CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');

html, body {{
    margin: 0 !important;
    min-height: 100% !important;
}}

body::before {{
    content: "";
    position: fixed;
    inset: 0;

    background:
        linear-gradient(
            rgba(35, 22, 15, 0.50),
            rgba(35, 22, 15, 0.50)
        ),
        url('{background_image}');

    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;

    z-index: 0;
    pointer-events: none;
}}

.gradio-container {{
    position: relative;
    z-index: 1;

    min-height: 100vh !important;

    background: transparent !important;
}}


/* =========================
   HEADER
   ========================= */

#cafe-header {{
    width: 90%;
    max-width: 1325px;

    margin: 10px auto 20px auto;
    padding: 20px;

    box-sizing: border-box;

    text-align: center;

    background: rgba(250, 244, 235, 0.88);

    border-radius: 24px;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);
}}

.logo-wrap {{
    display: inline-block;

    position: relative;

    max-width: 100%;

    overflow: visible;

    padding:
        5px 15px 20px 15px;
}}

.handwritten-logo {{
    display: inline-block;

    font-family:
        'Great Vibes',
        cursive;

    font-size: 86px;
    font-weight: 410;
    line-height: 1.2;

    color: #4b2e20;

    white-space: nowrap;

    max-width: 100%;

    padding-bottom: 5px;

    clip-path:
        inset(0 100% 0 0);

    animation:
        handwriting
        3.2s
        cubic-bezier(0.45, 0, 0.2, 1)
        forwards;
}}

.pen-line {{
    position: absolute;

    left: 10px;
    bottom: 8px;

    height: 2px;
    width: 0;

    background: #4b2e20;

    border-radius: 100%;

    transform:
        rotate(-2deg);

    animation:
        flourish
        1.2s
        ease-out
        2.6s
        forwards;
}}

#cafe-header p {{
    color: #6f5140;

    font-size: 17px;

    margin:
        4px 0;
}}


/* =========================
   CHAT CONTAINER
   ========================= */

#chat-area {{
    width: 90%;
max-width: 1325px;
min-width: 500px;

    margin:
        0 auto 30px auto;

    padding: 14px;

    box-sizing: border-box;

    background:
        rgba(250, 244, 235, 0.92);

    border-radius: 24px;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25);

    resize: both;

    overflow: auto;

    display: flex;

    flex-direction: column;

    align-items: stretch;
}}


/* All direct elements inside chat-area */
#chat-area > * {{
    width: 100% !important;

    max-width: 100% !important;

    box-sizing: border-box;
}}


/* =========================
   CHATBOT OUTPUT
   ========================= */

#chat-area .chatbot {{
    width: 100% !important;

    max-width: 100% !important;

    box-sizing: border-box;
}}


/* =========================
   INPUT AREA
   ========================= */

#chat-area form {{
    width: 100% !important;

    max-width: 100% !important;

    box-sizing: border-box;
}}

#chat-area textarea {{
    width: 100% !important;

    max-width: 100% !important;

    box-sizing: border-box;

    border-radius: 14px !important;
}}

#chat-area input {{
    width: 100% !important;

    max-width: 100% !important;

    box-sizing: border-box;
}}


/* =========================
   BUTTONS
   ========================= */

button {{
    border-radius: 14px !important;
}}


/* =========================
   ANIMATIONS
   ========================= */

@keyframes handwriting {{
    0% {{
        clip-path:
            inset(0 100% 0 0);

        transform:
            translateX(-5px);

        opacity: 0;
    }}

    10% {{
        opacity: 1;
    }}

    100% {{
        clip-path:
            inset(0 0 0 0);

        transform:
            translateX(0);

        opacity: 1;
    }}
}}

@keyframes flourish {{
    from {{
        width: 0;

        opacity: 0;
    }}

    to {{
        width: 95%;

        opacity: 1;
    }}
}}


/* =========================
   TABLET
   ========================= */

@media (max-width: 900px) {{

    #cafe-header {{
        width: 94%;

        padding:
            16px;
    }}

    .handwritten-logo {{
        font-size: 64px;
    }}

    #chat-area {{
        width: 94vw;

        min-width: 0;

        max-width: 94vw;

        padding: 10px;

        resize: vertical;
    }}
}}


/* =========================
   SMARTPHONE
   ========================= */

@media (max-width: 600px) {{

    #cafe-header {{
        width: 96%;

        padding:
            12px;
    }}

    .handwritten-logo {{
        font-size: 48px;

        white-space: normal;

        text-align: center;
    }}

    #cafe-header p {{
        font-size: 14px;
    }}

    #chat-area {{
        width: 96vw;

        max-width: 96vw;

        min-width: 0;

        padding: 8px;

        resize: vertical;
    }}
}}
"""


def load_wisdoms():
    with open(WISDOMS_FILE, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def start_chat():
    wisdoms = load_wisdoms()
    wisdom = random.choice(wisdoms)

    return [
        {
            "role": "assistant",
            "content": (
                "Hello! Welcome to Rast Cafe ☕\n\n"
                f"☕ {wisdom}\n\n"
                "I can help you with our menu, prices, opening hours, "
                "cafe services and house rules. How can I help you?"
            )
        }
    ]


with gr.Blocks(css=CUSTOM_CSS) as demo:

    gr.HTML("""
    <div id="cafe-header">

        <div class="logo-wrap">
            <div class="handwritten-logo">Rast Cafe</div>
            <div class="pen-line"></div>
        </div>

        <p>Your cozy coffee companion</p>
        <p>Ask about our menu, prices, opening hours and café services.</p>

    </div>
    """)

    with gr.Column(elem_id="chat-area"):

        chatbot = gr.Chatbot(
     resizable=True
)

        gr.ChatInterface(
            fn=chat_with_agent,
            chatbot=chatbot,
            examples=[
                "Do you have cappuccino?",
                "How much are two cappuccinos and one cheesecake?",
                "What are your opening hours?",
                "How long do you keep lost items?",
                "Can I bring my dog?",
                "I want to speak to a staff member."
            ]
        )

    demo.load(
        fn=start_chat,
        outputs=chatbot
    )


if __name__ == "__main__":
    demo.launch()