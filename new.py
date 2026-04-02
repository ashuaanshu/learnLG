from langgraph.graph import StateGraph, START, END
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint
import datetime
#--------------------------------------------------
from langchain.tools import tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC




checkpointr = InMemorySaver()

model = ChatOllama(model="llama3.2", temperature=0.9)

@tool
def datetime_now() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().isoformat()

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
    """You are a helpful assistant. 
    - Use math tools for calculations.
    - Use train_status tool when user asks about train running status.
    - Always call tools when needed instead of guessing.""")]+state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages":[response]}

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
    if user_input == "exit":
        break
    result = abc.invoke({'messages': [("user", user_input)]},
                        config=config
                        )
    print(result["messages"][-1].content)
    rprint(f"[bold red]{result['messages'][-1].usage_metadata}[/bold red]")
    


