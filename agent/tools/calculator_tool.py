# tools/calculator_tool.py

from langchain.tools import tool
import re

@tool
def calculator_tool(expression: str) -> str:
    """
    Use this tool ONLY for mathematical calculations like 
    percentages, addition, subtraction, multiplication, division.
    Input should be a plain math expression, e.g. '45000 * 0.18' 
    or '(200 + 300) / 2'.
    Do NOT use this for general questions or PDF content questions.
    """
    try:
        # Only allow numbers, operators, decimal points, and spaces
        # This blocks any code injection attempts
        if not re.match(r'^[0-9+\-*/().%\s]+$', expression):
            return "Error: Invalid characters in expression. Only numbers and math operators allowed."
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    
    except Exception as e:
        return f"Error calculating: {str(e)}. Please provide a valid math expression."