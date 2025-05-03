import logging
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any

# Import necessary components from our existing modules
# Ensure agent and graph are initialized before FastAPI starts
# We might need to adjust agent.py/graph_builder.py slightly if they exit on error
try:
    # Import components from agent.py
    from agent import (
        react_agent_executor,
    )  # Removed unused GEMINI_MODEL, GEMINI_SUMMARY_MODEL

    # Import the builder function from graph_builder.py
    from graph_builder import build_graph
except ImportError as e:
    print(
        f"FATAL: Could not import agent/graph components. Ensure agent.py and graph_builder.py are correct. Error: {e}"
    )
    exit(1)
except Exception as e:
    print(f"FATAL: Error during agent/graph initialization. Error: {e}")
    exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)  # Quieter Uvicorn logs

# --- FastAPI App Setup ---
app = FastAPI(
    title="Internet Search and Analysis Agent API",
    description="API for interacting with the ReAct agent and triggering sustainability report analysis.",
    version="1.0.0",
)

# --- Build Graph App ---
# Build the graph app instance when the server starts
try:
    analysis_graph_app = build_graph()
except Exception as e:
    logging.error(
        f"Failed to build the analysis graph during API server startup: {e}",
        exc_info=True,
    )
    print("FATAL: Could not build the analysis graph. Exiting.")
    exit(1)


# --- Data Models (Pydantic) ---
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


class AnalysisRequest(BaseModel):
    industry: str


class AnalysisTaskStatus(BaseModel):
    task_id: str
    status: str  # e.g., PENDING, RUNNING, COMPLETED, FAILED
    result: Any = None  # Will hold the final synthesis or error message


class AnalysisSubmitResponse(BaseModel):
    message: str
    task_id: str


# --- Background Task Management ---
# Simple in-memory storage for task status and results
# For production, use a more robust solution like Redis/Celery/DB
analysis_tasks: Dict[str, AnalysisTaskStatus] = {}


async def run_analysis_background(task_id: str, industry: str):
    """Runs the LangGraph analysis in the background."""
    logging.info(
        f"Starting background analysis task {task_id} for industry: {industry}"
    )
    analysis_tasks[task_id] = AnalysisTaskStatus(task_id=task_id, status="RUNNING")
    try:
        inputs = {"industry": industry}
        # Use graph_app.ainvoke for async execution if available/needed
        # For simplicity, running invoke in executor thread pool via asyncio.to_thread
        # Note: LangGraph's invoke might block the event loop if not handled carefully.
        # Consider using graph_app.astream or running in a separate process for true non-blocking.
        final_state = await asyncio.to_thread(analysis_graph_app.invoke, inputs)

        result = final_state.get("synthesis_result") or final_state.get(
            "error_message", "Graph finished, but no result/error found."
        )
        status = "COMPLETED" if not final_state.get("error_message") else "FAILED"
        analysis_tasks[task_id] = AnalysisTaskStatus(
            task_id=task_id, status=status, result=result
        )
        logging.info(f"Completed background analysis task {task_id}. Status: {status}")

    except Exception as e:
        logging.error(
            f"Error in background analysis task {task_id}: {e}", exc_info=True
        )
        analysis_tasks[task_id] = AnalysisTaskStatus(
            task_id=task_id,
            status="FAILED",
            result=f"An unexpected error occurred: {e}",
        )


# --- API Endpoints ---


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handles general queries using the ReAct agent."""
    logging.info(f"Received query: {request.query}")
    try:
        # Run react_agent_executor.invoke in a thread pool to avoid blocking event loop
        response = await asyncio.to_thread(
            react_agent_executor.invoke, {"input": request.query}
        )
        agent_response = response.get("output", "Agent did not provide a final answer.")
        return QueryResponse(response=agent_response)
    except Exception as e:
        logging.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.post("/analyze", response_model=AnalysisSubmitResponse)
async def submit_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submits a sustainability report analysis task to run in the background."""
    logging.info(f"Received analysis request for industry: {request.industry}")
    task_id = str(uuid.uuid4())
    analysis_tasks[task_id] = AnalysisTaskStatus(task_id=task_id, status="PENDING")
    background_tasks.add_task(run_analysis_background, task_id, request.industry)
    logging.info(f"Submitted analysis task {task_id} for industry: {request.industry}")
    return AnalysisSubmitResponse(message="Analysis task submitted.", task_id=task_id)


@app.get("/analysis/{task_id}", response_model=AnalysisTaskStatus)
async def get_analysis_status(task_id: str):
    """Retrieves the status and result (if available) of an analysis task."""
    logging.info(f"Checking status for analysis task: {task_id}")
    task = analysis_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    return task


@app.get("/")
async def read_root():
    return {"message": "Internet Search and Analysis Agent API is running."}


# --- Main execution (for running with uvicorn) ---
# This part is usually not included if running via `uvicorn api_server:app`
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting API server with Uvicorn...")
#     # Ensure GOOGLE_API_KEY is set before starting
#     if not os.getenv("GOOGLE_API_KEY"):
#          print("Error: GOOGLE_API_KEY environment variable not set.")
#          exit(1)
#     uvicorn.run(app, host="0.0.0.0", port=8000)
