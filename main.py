from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
# from ollama_chat import ChatOllama
# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model = "phi3", temperature=0)

class state(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
    
graph_builder = StateGraph(state)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

graph = graph_builder.compile()


result = graph.invoke({
    "messages": [HumanMessage(content="hello")]
})


print(result["messages"][-1].content)