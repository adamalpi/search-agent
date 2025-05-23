import hashlib  # For creating safe filenames from URLs
import logging
import os
import time
from urllib.parse import unquote, urlparse

import requests
from langchain.tools import Tool
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# --- Configuration ---
# Use a persistent cache directory relative to this file's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "pdf_cache")
logger.info(f"PDF cache directory configured: {CACHE_DIR}")

# --- Helper Functions ---


def _is_valid_url(url: str) -> bool:
    """Checks if a string is a valid HTTP/HTTPS URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ["http", "https"], result.netloc])
    except ValueError:
        return False


def _url_to_filename(url: str) -> str:
    """Creates a safe filename from a URL, using a hash for uniqueness."""
    # Use SHA256 hash of the URL to ensure uniqueness and avoid filesystem issues
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # Optionally, try to get a readable name from the URL path for easier debugging
    try:
        path = urlparse(url).path
        raw_name = os.path.basename(unquote(path))

        # Single-pass sanitization: keep only alphanumeric, underscore, hyphen, and dot
        safe_name = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-", "."))

        base_name, ext = os.path.splitext(safe_name)

        # Now check if the extension is '.pdf' (case-insensitive) and if base_name exists
        if ext.lower() == ".pdf" and base_name:
            # Limit length and append hash part for uniqueness
            filename = f"{base_name[:50]}_{url_hash[:10]}{ext}"
            return filename
    except Exception as e:
        logger.warning(f"Exception during filename generation for {url}: {e}. Falling back to hash.")
        pass  # Fallback if any error occurs

    return f"{url_hash}.pdf"


# --- Core Functionality ---


def download_pdf(url: str) -> str:
    """
    Downloads a PDF file from a given URL to a local cache directory,
    checking the cache first.

    Args:
        url: The URL of the PDF file to download.

    Returns:
        The local file path of the (potentially cached) PDF if successful,
        otherwise an error message string.
    """
    if not _is_valid_url(url):
        return f"Error: Invalid URL provided: {url}"
    # Relax the check here, rely on content-type or download error later if not PDF

    # Generate filename and check cache
    filename = _url_to_filename(url)
    local_filepath = os.path.join(CACHE_DIR, filename)

    # Ensure cache directory exists before checking/writing
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(local_filepath):
        logger.info(f"Cache hit: PDF already exists at {local_filepath} for URL {url}")
        return local_filepath

    logger.info(f"Cache miss: Attempting to download PDF from: {url}")
    MAX_DOWNLOAD_RETRIES = 3
    RETRY_DOWNLOAD_DELAY_SECONDS = 10
    DOWNLOAD_TIMEOUT_SECONDS = 60  # Increased timeout

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"  # noqa: E501
    }

    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            logger.info(f"Download attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES} for URL: {url}")
            response = requests.get(
                url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, headers=headers, allow_redirects=True
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            content_type = response.headers.get("content-type", "").lower()
            if (
                "application/pdf" not in content_type and "octet-stream" not in content_type
            ):  # Also allow octet-stream as it's common for direct downloads
                logger.warning(
                    f"Content-Type for {url} is '{content_type}', not 'application/pdf' "
                    "or 'application/octet-stream'. Attempting download anyway."
                )
                # Decide whether to proceed or return error - proceeding for now

            with open(local_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Successfully downloaded PDF to cache: {local_filepath}")
            return local_filepath  # Success

        except requests.exceptions.Timeout as e:
            logger.warning(f"Download attempt {attempt + 1} timed out for {url}: {e}")
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                logger.info(f"Retrying download in {RETRY_DOWNLOAD_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DOWNLOAD_DELAY_SECONDS)
            else:
                logger.error(f"All {MAX_DOWNLOAD_RETRIES} download attempts timed out for {url}.")
                # Clean up potentially incomplete file on final error
                if os.path.exists(local_filepath):
                    try:
                        os.remove(local_filepath)
                    except OSError:
                        pass
                error_message = (
                    f"Error: Failed to download PDF from {url} "
                    f"after {MAX_DOWNLOAD_RETRIES} attempts due to timeout. Reason: {e}"
                )
                return error_message
        except requests.exceptions.RequestException as e:
            logger.error(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                # Don't retry on all RequestExceptions, e.g. 404 might not be worth retrying
                # For now, retry on generic RequestException but this could be refined
                logger.info(f"Retrying download in {RETRY_DOWNLOAD_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DOWNLOAD_DELAY_SECONDS)
            else:
                logger.error(f"All {MAX_DOWNLOAD_RETRIES} download attempts failed for {url}. Reason: {e}")
                if os.path.exists(local_filepath):
                    try:
                        os.remove(local_filepath)
                    except OSError:
                        pass
                return f"Error: Failed to download PDF from {url} after {MAX_DOWNLOAD_RETRIES} attempts. Reason: {e}"
        except Exception as e:  # Catch any other unexpected error during a specific attempt
            logger.error(
                f"An unexpected error occurred during download attempt {attempt + 1} for {url}: {e}", exc_info=True
            )
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                logger.info(f"Retrying download in {RETRY_DOWNLOAD_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DOWNLOAD_DELAY_SECONDS)
            else:
                if os.path.exists(local_filepath):
                    try:
                        os.remove(local_filepath)
                    except OSError:
                        pass
                error_message = (
                    f"Error: An unexpected error occurred during download from {url} "
                    f"after {MAX_DOWNLOAD_RETRIES} attempts. Details: {e}"
                )
                return error_message

    # Fallback if loop finishes without returning (should not happen with current logic but as a safeguard)
    logger.error(f"Download process for {url} completed without success or explicit error return after retries.")
    return f"Error: Failed to download PDF from {url} after {MAX_DOWNLOAD_RETRIES} attempts (unknown reason)."


def extract_text_from_pdf(local_filepath: str) -> str:
    """
    Extracts text content from a local PDF file.

    Args:
        local_filepath: The path to the local PDF file.

    Returns:
        The extracted text content as a single string if successful,
        otherwise an error message string.
    """
    logger.info(f"Attempting to extract text from PDF: {local_filepath}")
    if not os.path.exists(local_filepath):
        return f"Error: PDF file not found at path: {local_filepath}"
    if not local_filepath.lower().endswith(".pdf"):
        return f"Error: File does not appear to be a PDF: {local_filepath}"

    try:
        reader = PdfReader(local_filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text extraction returned something
                text += page_text + "\n"  # Add newline between pages

        if not text:
            logger.warning(
                f"No text could be extracted from PDF: {local_filepath}. It might be image-based or protected."
            )
            return f"Warning: No text could be extracted from PDF: {local_filepath}. File might be image-based."

        logger.info(f"Successfully extracted text from PDF: {local_filepath} (Length: {len(text)})")
        return text

    except Exception as e:
        logger.error(f"Error extracting text from PDF {local_filepath}: {e}")
        return f"Error: Failed to extract text from PDF {local_filepath}. Reason: {e}"


# --- LangChain Tool Definitions ---

download_pdf_tool = Tool(
    name="Download PDF",
    func=download_pdf,
    description="Downloads a PDF file from a given URL to a local cache, returning the local path. Checks cache first. Input MUST be a valid URL potentially pointing to a PDF. Output is the local file path or an error message.",  # noqa: E501
)

extract_pdf_text_tool = Tool(
    name="Extract PDF Text",
    func=extract_text_from_pdf,
    description="Extracts text content from a locally stored PDF file (likely from the cache). Input MUST be a valid local file path to a PDF (usually the output of the 'Download PDF' tool). Output is the extracted text or an error message.",  # noqa: E501
)
