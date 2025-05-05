import os
from unittest.mock import MagicMock, mock_open

import pytest
import requests
from research_agent.file_tools import (
    CACHE_DIR,
    _is_valid_url,
    _url_to_filename,
    download_pdf,
    download_pdf_tool,
    extract_pdf_text_tool,
    extract_text_from_pdf,
)

# --- Test Helper Functions ---


def test_is_valid_url():
    assert _is_valid_url("http://example.com") is True
    assert _is_valid_url("https://example.com/path?query=1") is True
    assert _is_valid_url("ftp://example.com") is False
    assert _is_valid_url("example.com") is False
    assert _is_valid_url("http://") is False
    assert _is_valid_url("") is False


def test_url_to_filename():
    url1 = "http://example.com/report.pdf"
    fname1 = _url_to_filename(url1)
    assert fname1.startswith("report_")
    assert fname1.endswith(".pdf")
    assert len(fname1) > 15

    url2 = "https://example.com/very/long/path/name/that/is/way/too/long/for/a/filename/document.pdf"
    fname2 = _url_to_filename(url2)
    assert fname2.startswith("document_")
    assert fname2.endswith(".pdf")
    assert len(fname2) < 70

    url3 = "http://example.com/no_extension"
    fname3 = _url_to_filename(url3)
    assert not fname3.startswith("no_extension")
    assert fname3.endswith(".pdf")

    url4 = "http://example.com/weird%20chars?.pdf"
    fname4 = _url_to_filename(url4)
    # Check that the function falls back to hash for this problematic URL
    assert "weirdchars" not in fname4, f"Expected fallback, but 'weirdchars' found in filename: {fname4}"
    assert "_" not in fname4, f"Expected fallback (no separator), but '_' found in filename: {fname4}"
    assert fname4.endswith(".pdf")
    assert len(fname4) > 60


# --- Test Core Functions (with Mocks) ---


@pytest.fixture(autouse=True)
def ensure_cache_dir():
    """Ensure cache directory exists before tests and is cleaned after."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    yield
    # Optional: Clean up cache dir contents after tests if needed


def test_download_pdf_cache_hit(mocker):
    """Test that download_pdf returns cached path if file exists."""
    url = "http://example.com/cached.pdf"
    filename = _url_to_filename(url)
    filepath = os.path.join(CACHE_DIR, filename)

    with open(filepath, "w") as f:
        f.write("dummy pdf content")

    # Mock requests.get to ensure it's NOT called
    mock_get = mocker.patch("research_agent.file_tools.requests.get")

    result = download_pdf(url)

    assert result == filepath
    mock_get.assert_not_called()

    os.remove(filepath)


def test_download_pdf_cache_miss_success(mocker):
    """Test successful download when file is not cached."""
    url = "http://example.com/new_report.pdf"
    filename = _url_to_filename(url)
    filepath = os.path.join(CACHE_DIR, filename)

    # Ensure file does not exist initially
    if os.path.exists(filepath):
        os.remove(filepath)

    # Mock requests.get response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.iter_content.return_value = [b"pdf", b" content"]
    mock_get = mocker.patch(
        "research_agent.file_tools.requests.get",
        return_value=mock_response,
    )

    m = mock_open()
    mocker.patch("builtins.open", m)

    result = download_pdf(url)

    assert result == filepath
    mock_get.assert_called_once()
    m.assert_called_once_with(filepath, "wb")
    handle = m()
    handle.write.assert_any_call(b"pdf")
    handle.write.assert_any_call(b" content")

    # Actual file won't be created due to mock_open, so no cleanup needed here


def test_download_pdf_request_error(mocker):
    """Test download failure due to requests error."""
    url = "http://example.com/error.pdf"
    filename = _url_to_filename(url)
    filepath = os.path.join(CACHE_DIR, filename)

    # Mock requests.get to raise an error
    mock_get = mocker.patch("research_agent.file_tools.requests.get")
    mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

    result = download_pdf(url)

    assert result.startswith("Error: Failed to download PDF")
    assert "Connection failed" in result
    assert not os.path.exists(filepath)


def test_extract_text_from_pdf_success(mocker):
    """Test successful text extraction."""
    filepath = os.path.join(CACHE_DIR, "dummy_extract.pdf")
    with open(filepath, "w") as f:
        f.write("not real pdf")

    # Mock PdfReader
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 text."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 text."
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page1, mock_page2]
    mocker.patch(
        "research_agent.file_tools.PdfReader",
        return_value=mock_reader_instance,
    )

    result = extract_text_from_pdf(filepath)

    assert result == "Page 1 text.\nPage 2 text.\n"

    os.remove(filepath)


def test_extract_text_from_pdf_no_text(mocker):
    """Test extraction when PDF yields no text."""
    filepath = os.path.join(CACHE_DIR, "dummy_no_text.pdf")
    with open(filepath, "w") as f:
        f.write("not real pdf")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mocker.patch(
        "research_agent.file_tools.PdfReader",
        return_value=mock_reader_instance,
    )

    result = extract_text_from_pdf(filepath)

    assert result.startswith("Warning: No text could be extracted")

    os.remove(filepath)


def test_extract_text_from_pdf_file_not_found():
    """Test extraction when file doesn't exist."""
    filepath = os.path.join(CACHE_DIR, "non_existent.pdf")
    result = extract_text_from_pdf(filepath)
    assert result.startswith("Error: PDF file not found")


# --- Test LangChain Tools ---


def test_download_pdf_tool_run(mocker):
    """Test running the download tool."""
    # Mock the _run method of the tool instance directly
    mock_tool_run = mocker.patch.object(download_pdf_tool, "_run")
    mock_tool_run.return_value = "/path/to/cached/file.pdf"

    url = "http://example.com/tool_test.pdf"
    result = download_pdf_tool.run(url)

    assert result == "/path/to/cached/file.pdf"
    mock_tool_run.assert_called_once_with(url)


def test_extract_pdf_text_tool_run(mocker):
    """Test running the extraction tool."""
    # Mock the _run method of the tool instance directly
    mock_tool_run = mocker.patch.object(extract_pdf_text_tool, "_run")
    mock_tool_run.return_value = "Extracted text content."

    filepath = "/path/to/local.pdf"
    result = extract_pdf_text_tool.run(filepath)

    assert result == "Extracted text content."
    mock_tool_run.assert_called_once_with(filepath)
