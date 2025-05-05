import os
import time

import requests
import streamlit as st

# --- Configuration ---
# Assumes the FastAPI server is running locally on port 8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --- Helper Functions ---


def query_agent(query: str):
    """Sends a query to the FastAPI /query endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}/query", json={"query": query}, timeout=60)  # Add timeout
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json().get("response", "No response field found.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during query: {e}")
        return None


def submit_analysis_request(industry: str):
    """Submits an analysis request to the FastAPI /analyze endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", json={"industry": industry}, timeout=15)
        response.raise_for_status()
        return response.json()  # Returns {"message": "...", "task_id": "..."}
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred submitting analysis: {e}")
        return None


def get_analysis_status(task_id: str):
    """Gets the status of an analysis task from the FastAPI /analysis/{task_id} endpoint."""
    if not task_id:
        return None
    try:
        response = requests.get(f"{API_BASE_URL}/analysis/{task_id}", timeout=15)
        response.raise_for_status()
        return response.json()  # Returns {"task_id": ..., "status": ..., "result": ...}
    except requests.exceptions.RequestException as e:
        # Don't show error constantly during polling, maybe just log
        print(f"Polling Error: {e}")  # Log to console instead of UI
        return None  # Indicate status couldn't be fetched
    except Exception as e:
        st.error(f"An unexpected error occurred checking status: {e}")
        return None


# --- Streamlit UI ---

st.set_page_config(page_title="Search & Analysis Agent", layout="wide")
st.title("🔎 Internet Search & Sustainability Analysis Agent")

# --- General Query Section ---
st.header("General Query")
query_input = st.text_input("Ask the agent anything:", key="query_input")
if st.button("Submit Query", key="submit_query"):
    if query_input:
        with st.spinner("Thinking..."):
            agent_response = query_agent(query_input)
            if agent_response:
                st.markdown("**Agent Response:**")
                st.markdown(agent_response)  # Use markdown for potential formatting
    else:
        st.warning("Please enter a query.")

st.divider()

# --- Sustainability Analysis Section ---
st.header("Sustainability Report Analysis")
industry_input = st.text_input("Enter industry name to analyze:", key="industry_input")

# Initialize session state for task tracking
if "analysis_task_id" not in st.session_state:
    st.session_state.analysis_task_id = None
if "analysis_status" not in st.session_state:
    st.session_state.analysis_status = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "submitted_industry" not in st.session_state:
    st.session_state.submitted_industry = None


if st.button("Start Analysis", key="start_analysis"):
    if industry_input:
        st.session_state.analysis_task_id = None  # Reset previous task
        st.session_state.analysis_status = None
        st.session_state.analysis_result = None
        st.session_state.submitted_industry = industry_input
        with st.spinner(f"Submitting analysis request for '{industry_input}'..."):
            submit_response = submit_analysis_request(industry_input)
            if submit_response and "task_id" in submit_response:
                st.session_state.analysis_task_id = submit_response["task_id"]
                st.session_state.analysis_status = "PENDING"
                st.success(f"Analysis task submitted successfully! Task ID: {st.session_state.analysis_task_id}")
                st.info("Status will update below automatically (polling every 10s).")
            else:
                st.error("Failed to submit analysis task.")
                st.session_state.submitted_industry = None  # Clear if submission failed
    else:
        st.warning("Please enter an industry name.")

# Display analysis status and result (using polling)
if st.session_state.analysis_task_id:
    st.subheader(
        f"Analysis Status for '{st.session_state.submitted_industry}' (Task ID: {st.session_state.analysis_task_id})"
    )
    status_placeholder = st.empty()

    # Basic polling loop
    while st.session_state.analysis_status in ["PENDING", "RUNNING"]:
        status_data = get_analysis_status(st.session_state.analysis_task_id)
        if status_data:
            st.session_state.analysis_status = status_data.get("status", "UNKNOWN")
            st.session_state.analysis_result = status_data.get("result")
            status_placeholder.info(f"Status: {st.session_state.analysis_status}")
            if st.session_state.analysis_status not in ["PENDING", "RUNNING"]:
                break  # Exit loop if completed or failed
        else:
            status_placeholder.warning("Could not fetch status from API...")
            # Optional: Add a longer delay or break after multiple fetch failures
        time.sleep(10)  # Poll every 10 seconds

    # Display final status and result
    if st.session_state.analysis_status == "COMPLETED":
        status_placeholder.success(f"Status: {st.session_state.analysis_status}")
        st.markdown("**Analysis Result:**")
        st.markdown(st.session_state.analysis_result)
    elif st.session_state.analysis_status == "FAILED":
        status_placeholder.error(f"Status: {st.session_state.analysis_status}")
        st.markdown("**Error Details:**")
        st.error(st.session_state.analysis_result)
    elif st.session_state.analysis_status not in [
        "PENDING",
        "RUNNING",
    ]:  # Handle UNKNOWN or other states
        status_placeholder.warning(f"Status: {st.session_state.analysis_status}")
