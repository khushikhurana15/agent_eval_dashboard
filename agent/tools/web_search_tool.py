# tools/web_search_tool.py

from langchain.tools import tool
from ddgs import DDGS   # yahi line change hui hai

@tool
def web_search_tool(query: str) -> str:
    """
    Use this tool to search the internet for current events, live/real-time 
    information, recent news, or general knowledge questions that are NOT 
    related to the uploaded PDF documents (which cover AI/ML interview prep).
    Input should be a search query as plain text.
    """
    try:
        results = DDGS().text(query, max_results=3)

        if not results:
            return "No relevant results found on the web for this query."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            snippet = result.get("body", "")
            formatted_results.append(f"{i}. {title}: {snippet}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"