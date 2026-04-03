from langgraph.graph import StateGraph, START, END
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint
from datetime import datetime
import calendar
#--------------------------------------------------
from langchain.tools import tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


checkpointr = InMemorySaver()

model = ChatOllama(model="qwen3.5:0.8b", temperature=0.9, streaming=True)

@tool
def datetime_now() -> str:
    """Get the current date and time and day of the week."""
    now = datetime.now()
    
    day_name = calendar.day_name[now.weekday()]
    
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({day_name})"

@tool
def add(x: int, y: int) -> int:
    """Adds two numbers together."""
    return x + y

@tool
def subtract(x: int, y: int) -> int:
    """Subtract the second number from the first."""
    return x - y

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y

#--------------------------------------------------
@tool
def train_status(train_number: str, day: str = "today") -> str:
    """Get live train running status using train number and day (today/yesterday)."""
    
    options = Options()
    options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(f"https://www.railrestro.com/live-train-running-status/{train_number}?day={day}")
        wait = WebDriverWait(driver, 20)

        card = wait.until(
            EC.visibility_of_element_located((
                By.XPATH,
                "//div[contains(@class,'card-body') and contains(@class,'text-center')]"
            ))
        )

        strongs = card.find_elements(By.TAG_NAME, "strong")

        if len(strongs) >= 2:
            station = strongs[0].text
            updated = strongs[1].text
            return f"Train {train_number} is at {station}. Last updated: {updated}"
        else:
            return "Train data not found."

    except Exception as e:
        return f"Error: {str(e)}"

    finally:
        driver.quit()
#--------------------------------------------------

tools = [add, subtract, multiply, datetime_now, train_status]

model_with_tools = model.bind_tools(tools)

class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: TestState):
    messages = [("system",
    """You are a helpful assistant named Bengali Baba (nickname: Baba).
    You studied at IIT Kharagpur.
    Solve math problems step by step using the available tools, but stop after giving the final answer.
    You can provide train status using the tools if the user asks.
    Use the following tools when needed: add, subtract, multiply, datetime_now, train_status.
    Respond concisely and clearly. Do not repeat sentences.
    Respond in English.
    """)] + state["messages"]
    
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
def route_tool(state: TestState):
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
        # return {"messages": [f"The answer is {tool_result}"]}
    return END

graph = StateGraph(TestState)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", ToolNode(tools))

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", route_tool)
graph.add_edge("tool_node", "chatbot")

abc= graph.compile(checkpointer=checkpointr)

config ={"configurable": {"thread_id": "1"}, "max_tokens": 100 }

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    
    inputs ={"messages": [("user", user_input)]}
    print("Bot: ", end= "", flush=True)
    
    #steaming response
    for chunk, metadata in abc.stream(inputs, config, stream_mode="messages"):
        if hasattr(chunk, 'content'):
            print(chunk.content, end="", flush=True)
    print()
