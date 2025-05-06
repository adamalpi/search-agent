import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.file_tools import download_pdf_tool, extract_pdf_text_tool

logger = logging.getLogger(__name__)

CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parent.parent
SUMMARY_CACHE_DIR = PROJECT_ROOT / "summary_cache"
logger.info(f"Research Tools: Summary cache directory configured: {SUMMARY_CACHE_DIR}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=400)
logger.info("Research Tools: Initialized text splitter with chunk_size=8000, chunk_overlap=400.")


def _process_single_report_download_extract(company: str, url: str) -> Tuple[str, str]:
    """Helper function to download and extract text for a single report URL."""
    if url is None or url.startswith("Error"):
        logger.warning(f"Research Tools: Skipping download/extraction for {company} due to missing/error URL: {url}")
        return company, "No valid URL found."

    logger.info(f"Research Tools: Processing PDF for {company} from URL: {url}")
    try:
        local_path = download_pdf_tool.run(url)  # download_pdf_tool uses its own caching
        if local_path.startswith("Error"):
            logger.error(f"Research Tools: Download failed for {company}: {local_path}")
            return company, f"Download failed: {local_path}"

        extracted_text = extract_pdf_text_tool.run(local_path)
        if extracted_text.startswith("Error") or extracted_text.startswith("Warning"):
            logger.warning(f"Research Tools: Text extraction failed/empty for {company}: {extracted_text}")
            return company, extracted_text
        else:
            logger.info(f"Research Tools: Successfully extracted text for {company} (length: {len(extracted_text)}).")
            return company, extracted_text
    except Exception as e:
        logger.error(f"Research Tools: Error during download/extraction for {company}: {e}", exc_info=True)
        return company, f"Error during download/extraction: {e}"


def download_and_extract_reports(report_urls: Dict[str, str]) -> Dict[str, str]:
    """Downloads PDFs from URLs and extracts text in parallel.

    Args:
        report_urls: A dictionary mapping company names to report PDF URLs.

    Returns:
        A dictionary mapping company names to extracted text or error messages.
    """
    extracted_texts: Dict[str, str] = {}
    futures_map = {}
    logger.info("Research Tools: Starting parallel download and extraction of PDFs.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for company, url in report_urls.items():
            future = executor.submit(_process_single_report_download_extract, company, url)
            futures_map[future] = company

        for future in concurrent.futures.as_completed(futures_map):
            company_name = futures_map[future]
            try:
                _, result = future.result()
                extracted_texts[company_name] = result
                logger.info(f"Research Tools: Download/Extract completed for {company_name}.")
            except Exception as e:
                logger.error(
                    f"Research Tools: Exception retrieving result for {company_name} in download/extract: {e}",
                    exc_info=True,
                )
                extracted_texts[company_name] = f"Error retrieving download/extract result: {e}"

    logger.info("Research Tools: Finished parallel download and extraction.")
    return extracted_texts


async def _process_single_summary(company: str, text: str, summarize_chain) -> Tuple[str, str]:
    """Helper function to summarize text for a single company, using cache."""
    if (
        text is None
        or text.startswith("Error")
        or text.startswith("Warning")
        or text == "No valid URL found."
        or not text.strip()
    ):
        logger.warning(f"Research Tools: Skipping summarization for {company} due to previous error or empty text.")
        return company, "Skipped due to previous error or empty text."

    cache_filename = f"{company.replace(' ', '_').lower()}_summary.txt"
    cache_filepath = SUMMARY_CACHE_DIR / cache_filename

    SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_filepath.exists():
        logger.info(f"Research Tools: Cache hit for {company} summary: Loading from {cache_filepath}")
        try:
            with open(cache_filepath, "r") as f:
                summary = f.read()
            return company, summary
        except Exception as e:
            log_message = (
                f"Research Tools: Failed to read cached summary for {company} "
                f"from {cache_filepath}: {e}. Will regenerate."
            )
            logger.warning(log_message)

    logger.info(f"Research Tools: Cache miss for {company}. Generating summary...")
    try:
        docs = text_splitter.create_documents([text])
        logger.info(f"Research Tools: Processing {len(docs)} text chunks for {company} summary.")
        start_time = time.time()
        summary_output = await summarize_chain.ainvoke({"input_documents": docs})
        summary = summary_output.get("output_text", "")
        intermediate_steps = summary_output.get("intermediate_steps", [])
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Research Tools: Finished generating summary for {company} in {duration:.2f} seconds.")
        if intermediate_steps:
            logger.info(
                f"Research Tools: Intermediate steps for {company} "
                f"generated {len(intermediate_steps)} mapped summaries."
            )
            total_intermediate_length = sum(len(step) for step in intermediate_steps)
            logger.info(
                f"Research Tools: Total length of intermediate summaries for {company}: "
                f"{total_intermediate_length} characters."
            )
        else:
            logger.info(f"Research Tools: No intermediate steps returned for {company}.")

        if not summary:
            logger.warning(f"Research Tools: Summarization for {company} resulted in empty output.")
            return company, "Error: Summarization resulted in empty output."

        try:
            with open(cache_filepath, "w") as f:
                f.write(summary)
            logger.info(f"Research Tools: Saved summary for {company} to cache: {cache_filepath}")
        except Exception as e:
            logger.error(f"Research Tools: Failed to save summary to cache for {company} at {cache_filepath}: {e}")
        return company, summary

    except Exception as e:
        logger.error(f"Research Tools: Error summarizing report for {company}: {e}", exc_info=True)
        return company, f"Error during summarization: {e}"


async def summarize_extracted_texts(
    extracted_texts: Dict[str, str], summarize_chain, max_workers: int = 10
) -> Dict[str, str]:
    """Summarizes the extracted text for each report in parallel, using a cache.

    Args:
        extracted_texts: A dictionary mapping company names to extracted text.
        summarize_chain: The pre-configured LangChain summarization chain instance.
        max_workers: The maximum number of parallel workers for summarization.

    Returns:
        A dictionary mapping company names to summaries or error messages.
    """
    if summarize_chain is None:
        logger.error("Research Tools: Summarize chain was not provided. Cannot summarize.")
        return {company: "Error: Summarization chain not provided." for company in extracted_texts}

    individual_summaries: Dict[str, str] = {}
    logger.info("Research Tools: Starting concurrent summarization.")

    tasks = []
    for company, text in extracted_texts.items():
        tasks.append(_process_single_summary(company, text, summarize_chain))

    # Run all summarization tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, company in enumerate(extracted_texts.keys()):
        result_or_exception = results[i]
        if isinstance(result_or_exception, Exception):
            logger.error(
                f"Research Tools: Exception during summarization for {company}: {result_or_exception}",
                exc_info=result_or_exception,
            )
            individual_summaries[company] = f"Error retrieving summarization result: {result_or_exception}"
        else:
            _, summary_text = result_or_exception
            individual_summaries[company] = summary_text
            logger.info(f"Research Tools: Summarization completed for {company}.")

    logger.info("Research Tools: Finished concurrent summarization.")
    return individual_summaries
