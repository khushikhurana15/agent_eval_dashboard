# agent/agent_core.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from agent.tools.rag_tool import rag_tool
from agent.tools.web_search_tool import web_search_tool
from agent.tools.calculator_tool import calculator_tool

load_dotenv()

SYSTEM_PROMPT = """You are a research assistant with access to three tools:

1. rag_tool - searches uploaded PDF documents (AI/ML interview prep content)
2. web_search_tool - searches the internet for ANY current, live, or 
   real-time information — including news, prices, sports results, 
   current events, or anything else that could be outdated in your 
   training knowledge. ALWAYS use this tool for such questions instead 
   of relying on your own knowledge or refusing.
3. calculator_tool - performs mathematical calculations

Decision rules:
- If the question is about the PDF content (ML/AI concepts, interview prep), 
  use rag_tool FIRST.
- If rag_tool returns a "LOW_CONFIDENCE" message, that means the answer isn't 
  in the PDFs. In that case, try web_search_tool next if the question seems 
  like something the internet could answer. If neither tool can help, 
  honestly tell the user you don't know — do NOT make up an answer.
- If the question involves numbers, percentages, or math, use calculator_tool.
- If the question is about current events, live data, prices, or general 
  knowledge clearly outside the PDFs, use web_search_tool directly — 
  do not refuse just because you personally don't know the answer.
- Always explain briefly which tool you used and why, before giving the final answer.
"""

def get_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    tools = [rag_tool, web_search_tool, calculator_tool]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    return agent


def run_agent(question: str):
    agent = get_agent()

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    return result["messages"]


def extract_text(content):
    """
    Some providers/newer models can return content as a list of blocks
    (e.g. [{'type': 'text', 'text': '...', 'extras': {...}}]) instead of
    a plain string. This normalizes both formats into plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def parse_agent_trace(messages):
    steps = []
    final_answer = ""
    pending_calls = {}

    for msg in messages:
        if msg.type == "ai":
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    pending_calls[call["id"]] = {
                        "tool_name": call["name"],
                        "tool_input": call["args"],
                    }
            text = extract_text(msg.content)
            if text:
                final_answer = text

        elif msg.type == "tool":
            call_id = getattr(msg, "tool_call_id", None)
            step = pending_calls.get(call_id, {"tool_name": "unknown", "tool_input": {}})
            steps.append({
                "tool_name": step["tool_name"],
                "tool_input": step["tool_input"],
                "tool_output": extract_text(msg.content),
            })

    return final_answer, steps

# ---- Quick standalone test ----
# Tests ONE question at a time to conserve rate limits/quota, whichever
# provider is configured above. Change TEST_QUESTION below to try a
# different case.
if __name__ == "__main__":
    TEST_QUESTION = "What is the difference between bagging and boosting?"

    print(f"QUESTION: {TEST_QUESTION}\n")

    try:
        messages = run_agent(TEST_QUESTION)
        answer, steps = parse_agent_trace(messages)

        if steps:
            for i, step in enumerate(steps, 1):
                print(f"Step {i}: used '{step['tool_name']}'")
                print(f"  Input:  {step['tool_input']}")
                print(f"  Output: {step['tool_output'][:200]}")
        else:
            print("(No tools were used — answered directly.)")

        print(f"\nFINAL ANSWER:\n{answer}")

    except Exception as e:
        # Show the REAL error — don't guess which provider or reason without
        # looking. A 429 could mean quota, rate limit, or a billing issue,
        # and the message text tells us which.
        print(f"⚠️  Agent run failed with: {type(e).__name__}: {e}")