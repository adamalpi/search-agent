import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from logging_config import setup_logging  # Import the setup function
from pydantic import BaseModel, Field
from research_agent.database import (
    init_db,
    log_task_status,
)

# Ensure agent and graph are initialized before FastAPI starts
# We might need to adjust agent.py/graph_builder.py slightly if they exit on error
try:
    from research_agent.graph_builder import (
        build_unified_graph,
    )
except ImportError as e:
    print(
        f"FATAL: Could not import agent/graph components. Ensure agent.py and graph_builder.py are correct. Error: {e}"
    )
    exit(1)
except Exception as e:
    print(f"FATAL: Error during agent/graph initialization. Error: {e}")
    exit(1)


setup_logging()

# --- FastAPI App Setup ---
app = FastAPI(
    title="Internet Search and Analysis Agent API",
    description="API for interacting with the ReAct agent and triggering sustainability report analysis.",
    version="1.0.0",
)

try:
    unified_graph_app = build_unified_graph()
except Exception as e:
    logging.error(
        f"Failed to build the unified graph during API server startup: {e}",
        exc_info=True,
    )
    print("FATAL: Could not build the unified graph. Exiting.")
    exit(1)


init_db()


# --- Data Models (Pydantic) ---
class QueryRequest(BaseModel):
    query: str
    # Use Any for messages to simplify Pydantic validation, actual type is List[BaseMessage]
    # We'll convert dicts to BaseMessage objects before passing to the graph
    messages: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]]


class AnalysisRequest(BaseModel):
    industry: str


class AnalysisTaskStatus(BaseModel):
    task_id: str
    status: str  # e.g., PENDING, RUNNING, COMPLETED, FAILED
    result: Any = None
    start_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None


class AnalysisSubmitResponse(BaseModel):
    message: str
    task_id: str


# --- Background Task Management ---
# Simple in-memory storage for task status and results
# For production, use a more robust solution like Redis/Celery/DB
analysis_tasks: Dict[str, AnalysisTaskStatus] = {}


async def run_analysis_background(task_id: str, industry: str):
    """Runs the LangGraph analysis in the background."""
    start_time = datetime.now()
    logging.info(f"Starting background analysis task {task_id} for industry: {industry} at {start_time.isoformat()}")
    # Log initial RUNNING status to DB and update in-memory dict
    analysis_tasks[task_id] = AnalysisTaskStatus(task_id=task_id, status="RUNNING", start_time=start_time)
    log_task_status(task_id=task_id, industry=industry, status="RUNNING", start_time=start_time.isoformat())
    try:
        inputs = {"industry": industry}
        final_state = await unified_graph_app.ainvoke(inputs)

        if final_state.get("error_message"):
            result = final_state["error_message"]
            status = "FAILED"
        elif final_state.get("synthesis_result"):
            result = final_state["synthesis_result"]
            status = "COMPLETED"
        else:
            result = "Graph finished, but no synthesis result or error message found."
            status = "UNKNOWN"  # Or potentially FAILED depending on expected outcome

        end_time = datetime.now()
        duration_seconds = int((end_time - start_time).total_seconds())

        # Update in-memory status (for immediate checks via /analysis/{task_id})
        analysis_tasks[task_id] = AnalysisTaskStatus(
            task_id=task_id, status=status, result=result, start_time=start_time, duration_seconds=duration_seconds
        )
        log_task_status(
            task_id=task_id,
            industry=industry,
            status=status,
            result_summary=str(result),
            start_time=start_time.isoformat(),  # Keep start time consistent
            duration_seconds=duration_seconds,
        )
        logging.info(f"Completed background analysis task {task_id}. Status: {status}, Duration: {duration_seconds}s")

    except Exception as e:
        end_time = datetime.now()
        duration_seconds = int((end_time - start_time).total_seconds())
        logging.error(f"Error in background analysis task {task_id}: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {e}"
        analysis_tasks[task_id] = AnalysisTaskStatus(
            task_id=task_id,
            status="FAILED",
            result=error_message,
            start_time=start_time,
            duration_seconds=duration_seconds,
        )
        log_task_status(
            task_id=task_id,
            industry=industry,
            status="FAILED",
            result_summary=error_message,
            start_time=start_time.isoformat(),
            duration_seconds=duration_seconds,
        )


# --- API Endpoints ---


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handles general queries using the unified graph."""
    logging.info(f"Received query: {request.query}, History length: {len(request.messages)}")

    previous_messages: List[BaseMessage] = []
    for msg_data in request.messages:
        if msg_data.get("type") == "human":
            previous_messages.append(HumanMessage(**msg_data))
        elif msg_data.get("type") == "ai":
            previous_messages.append(AIMessage(**msg_data))

    graph_input = {
        "input_query": request.query,
        "messages": previous_messages + [HumanMessage(content=request.query)],
    }

    try:
        final_state = await unified_graph_app.ainvoke(graph_input)

        if final_state.get("error_message"):
            logging.error(f"Error processing query via graph: {final_state['error_message']}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {final_state['error_message']}",
            )

        agent_response = final_state.get("agent_response", "Agent did not provide a final answer.")
        updated_messages_lc = final_state.get("messages", [])

        updated_messages_dict = [msg.dict() for msg in updated_messages_lc]

        return QueryResponse(response=agent_response, messages=updated_messages_dict)

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
