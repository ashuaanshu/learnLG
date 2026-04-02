from langgraph.graph import StateGraph, START, END
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint



checkpointr = InMemorySaver()

model = ChatOllama(model="llama3.2", temperature=0.9)

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


tools = [add, subtract, multiply]

model_with_tools = model.bind_tools(tools)

class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: TestState):
    messages = [("system",
    "You are a helpful assistant and math agent. If you get a math question, "
    "use the available tools to solve it until you reach the final answer, "
    "and explain the process step-by-step.")]+state["messages"]
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
    


