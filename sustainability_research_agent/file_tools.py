import logging
import requests
import os
import hashlib  # For creating safe filenames from URLs
from pypdf import PdfReader
from langchain.tools import Tool
from urllib.parse import urlparse, unquote
# Remove tempfile import
# import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# Use a persistent cache directory relative to this file's location
# __file__ gives the path of the current file (file_tools.py)
# os.path.dirname gets the directory containing it
# os.path.join creates the full path to 'pdf_cache'
CACHE_DIR = os.path.join(os.path.dirname(__file__), "pdf_cache")
# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
logging.info(f"Using PDF cache directory: {CACHE_DIR}")

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
        name = os.path.basename(unquote(path))
        if name and name.lower().endswith(".pdf"):
            # Keep only alphanumeric, underscore, hyphen, dot
            safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))
            # Limit length and append hash part for uniqueness
            return f"{safe_name[:50]}_{url_hash[:10]}.pdf"
    except Exception:
        pass  # Fallback to just hash if parsing fails
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
    # if not url.lower().endswith('.pdf'):
    #     return f"Error: URL does not appear to point to a PDF file: {url}"

    # Generate filename and check cache
    filename = _url_to_filename(url)
    local_filepath = os.path.join(CACHE_DIR, filename)

    if os.path.exists(local_filepath):
        logging.info(f"Cache hit: PDF already exists at {local_filepath} for URL {url}")
        return local_filepath

    logging.info(f"Cache miss: Attempting to download PDF from: {url}")
    try:
        # Use headers common for browsers to help with access
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(
            url, stream=True, timeout=30, headers=headers, allow_redirects=True
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type if possible
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" not in content_type:
            logging.warning(
                f"Content-Type for {url} is '{content_type}', not 'application/pdf'. Attempting download anyway."
            )
            # Decide whether to proceed or return error - proceeding for now
            # return f"Error: URL {url} did not return PDF content (Content-Type: {content_type})"

        with open(local_filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Successfully downloaded PDF to cache: {local_filepath}")
        return local_filepath

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF from {url}: {e}")
        # Clean up potentially incomplete file on error
        if os.path.exists(local_filepath):
            try:
                os.remove(local_filepath)
            except OSError:
                pass
        return f"Error: Failed to download PDF from {url}. Reason: {e}"
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during PDF download from {url}: {e}"
        )
        return f"Error: An unexpected error occurred during download. Details: {e}"


def extract_text_from_pdf(local_filepath: str) -> str:
    """
    Extracts text content from a local PDF file. (No changes needed here for caching)

    Args:
        local_filepath: The path to the local PDF file.

    Returns:
        The extracted text content as a single string if successful,
        otherwise an error message string.
    """
    logging.info(f"Attempting to extract text from PDF: {local_filepath}")
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
            logging.warning(
                f"No text could be extracted from PDF: {local_filepath}. It might be image-based or protected."
            )
            return f"Warning: No text could be extracted from PDF: {local_filepath}. File might be image-based."

        logging.info(
            f"Successfully extracted text from PDF: {local_filepath} (Length: {len(text)})"
        )
        return text

    except Exception as e:
        logging.error(f"Error extracting text from PDF {local_filepath}: {e}")
        return f"Error: Failed to extract text from PDF {local_filepath}. Reason: {e}"


# --- LangChain Tool Definitions ---

download_pdf_tool = Tool(
    name="Download PDF",
    func=download_pdf,
    description="Downloads a PDF file from a given URL to a local cache, returning the local path. Checks cache first. Input MUST be a valid URL potentially pointing to a PDF. Output is the local file path or an error message.",
)

extract_pdf_text_tool = Tool(
    name="Extract PDF Text",
    func=extract_text_from_pdf,
    description="Extracts text content from a locally stored PDF file (likely from the cache). Input MUST be a valid local file path to a PDF (usually the output of the 'Download PDF' tool). Output is the extracted text or an error message.",
)

# --- Example Usage (Moved to tests/test_file_tools.py) ---
