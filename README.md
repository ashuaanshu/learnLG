#  AI Agent 🤖

An AI-powered autonomous assistant built using

* LangGraph
* LangChain
* Ollama
* Selenium

The assistant can:

* Solve math problems
* Answer questions
* Get live train running status
* Fetch weather information
* Scrape websites
* Remember scraped content inside workflow state
* Stream responses in real time

---

# Features 

## AI Chatbot

Conversational assistant powered by local LLMs using Ollama.

## Tool Calling

Uses LangGraph + LangChain tools architecture.

Built-in tools:

* Addition
* Subtraction
* Multiplication
* Current datetime
* Live train status
* Weather fetching
* Website scraping

## Real-Time Streaming

Streams responses token-by-token.

## Stateful Workflow

Maintains workflow state using LangGraph.

## Website Scraping

Can scrape website text content and answer based on it.

## Memory Awareness

Stores scraped website content in graph state to avoid repeated scraping.

---

# Tech Stack 

* Python
* LangGraph
* LangChain
* Ollama
* Selenium
* ChromeDriver

---

# Project Structure 

```bash
project/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Installation 

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd project
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama 🦙

Download and install Ollama:

* Linux/macOS/Windows:
  https://ollama.com

---

# Pull Model

```bash
ollama pull qwen3.5:0.8b
```

You can also use:

* llama3.1
* qwen2.5
* mistral
* deepseek-r1

---

# Install ChromeDriver 🌐

Make sure:

* Google Chrome is installed
* ChromeDriver version matches your Chrome browser

Download:
https://chromedriver.chromium.org/

Add ChromeDriver to PATH.

---

# Run the Project 

```bash
python app.py
```

---

# Example Usage 💬

## Math

```text
User: add 10 and 20
Bot: 30
```

---

## Train Status

```text
User: where is train 12345
Bot: Train 12345 is at New Delhi...
```

---

## Weather

```text
User: weather in Delhi
Bot: weather in Delhi: Sunny +34°C
```

---

## Website Scraping

```text
User: summarize https://example.com
```

The assistant:

1. Scrapes website content
2. Stores content in graph state
3. Answers from scraped data

---

# Architecture 

```text
User
 ↓
Chatbot Node
 ↓
Tool Router
 ↓
Tool Node
 ↓
Save Data Node
 ↓
Chatbot
```

---

# LangGraph Workflow 

## Nodes

* chatbot
* tool_node
* save_data

## State

```python
class TestState(TypedDict):
    messages: Annotated[list, add_messages]
    scraped_content: str
```

---

# Current Limitations 

* Selenium is resource-heavy
* No async support
* No persistent database memory
* Context can grow large over time
* scrape_website uses basic extraction only

---

# Future Improvements 

* Replace Selenium with BeautifulSoup
* Add vector database memory
* Add async execution
* Add web search tool
* Add document Q&A
* Add PDF support
* Add persistent storage
* Improve security validation

---

# Requirements 

Example `requirements.txt`

```txt
langgraph
langchain
langchain-ollama
selenium
rich
requests
```

---
built by @ashuaanshu

---

# License 📄

MIT License
