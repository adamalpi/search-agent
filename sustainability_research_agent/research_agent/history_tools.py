import json
import logging
from typing import Any, Dict, List, Optional

from langchain.tools import Tool
from pydantic import BaseModel, Field, ValidationError

from research_agent.database import query_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Pydantic Schema for Tool Arguments ---
class QueryHistorySchema(BaseModel):
    """Input schema for the Query Analysis History tool."""

    limit: int = Field(default=5, description="Maximum number of recent COMPLETED tasks to retrieve.")
    industry_filter: Optional[str] = Field(
        default=None,
        description="Filter COMPLETED tasks by industry name (case-insensitive).",
    )


# --- Tool Function ---
def _query_analysis_history_func(tool_input: str) -> str:
    """
    Parses JSON input, queries the database for completed analysis tasks, and formats the results.
    Input should be a JSON string like '{"limit": 5, "industry_filter": "Automotive"}'.
    """
    logging.info(f"Tool: Received input string: {tool_input}")
    limit = 5
    industry_filter = None

    try:
        data = json.loads(tool_input)
        validated_data = QueryHistorySchema.model_validate(data)
        limit = validated_data.limit
        industry_filter = validated_data.industry_filter
        logging.info(f"Tool: Parsed arguments - limit={limit}, industry='{industry_filter}'")

    except json.JSONDecodeError:
        logging.warning(f"Tool: Input is not valid JSON: {tool_input}. Using default parameters.")
        # Proceed with default parameters if input is not valid JSON
        # Alternatively, return an error message:
        # return f"Error: Input must be a valid JSON string. Received: {tool_input}"
    except ValidationError as e:
        logging.warning(f"Tool: Input validation failed: {e}. Using default parameters.")
        # Proceed with default parameters if validation fails
        # Alternatively, return an error message:
        # return f"Error: Input validation failed: {e}. Received: {tool_input}"
    except Exception as e:
        logging.error(f"Tool: Unexpected error processing input '{tool_input}': {e}", exc_info=True)
        return f"Error processing tool input: {e}"

    # --- Original query logic starts here ---
    logging.info(f"Tool: Querying analysis history (limit={limit}, industry='{industry_filter}')")
    try:
        tasks: List[Dict[str, Any]] = query_tasks(limit=limit, industry_filter=industry_filter)

        if not tasks:
            filter_msg = f" for industry '{industry_filter}'" if industry_filter else ""
            return f"No completed analysis tasks found{filter_msg} matching the criteria."

        formatted_results = f"Found {len(tasks)} completed analysis tasks:\n"
        for i, task in enumerate(tasks, 1):
            task_id = task.get("task_id", "N/A")
            industry = task.get("industry", "N/A")
            timestamp = task.get("timestamp", "N/A")
            summary = str(task.get("result_summary", "N/A"))
            # Build the result string piece by piece
            formatted_results += f"{i}. Task ID: {task_id}\n"
            formatted_results += f"   Industry: {industry}\n"
            formatted_results += f"   Completed: {timestamp}\n"
            formatted_results += f"   Summary: {summary}\n---\n"

        return formatted_results.strip()

    except Exception as e:
        logging.error(f"Error querying analysis history: {e}", exc_info=True)
        return f"An error occurred while querying the analysis history: {e}"


# --- LangChain Tool Definition ---
query_analysis_history_tool = Tool(
    name="Query Analysis History",
    func=_query_analysis_history_func,
    description='Queries the history of successfully completed sustainability analysis tasks, returning the full summary for each. Use it to answer questions about past analyses, like listing the \'last N\' tasks or finding tasks for a specific industry. Input MUST be a JSON string specifying parameters, e.g., \'{{"limit": 3}}\' or \'{{"industry_filter": "Automotive"}}\' or \'{{"limit": 10, "industry_filter": "Tech"}}\'. \'limit\' defaults to 5 if not provided.',  # noqa: E501
    # args_schema removed - tool now expects a single string input
)

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Testing Query Analysis History Tool...")
    # Assuming database.py has run and populated some data
    print("\nQuerying last 2 tasks:")
    print(query_analysis_history_tool.run('{"limit": 2}'))  # Input must be JSON string
    print("\nQuerying last 2 Automotive tasks:")
    print(query_analysis_history_tool.run('{"limit": 2, "industry_filter": "Automotive"}'))  # Input must be JSON string
    print("\nQuerying tasks for 'NonExistentIndustry':")
    print(query_analysis_history_tool.run('{"industry_filter": "NonExistentIndustry"}'))  # Input must be JSON string
    print("\nQuerying with invalid JSON:")
    print(query_analysis_history_tool.run('{"limit": "not-an-int"}'))  # Example of validation failure
    print("\nQuerying with non-JSON string:")
    print(query_analysis_history_tool.run("limit 3"))  # Example of JSON decode failure
