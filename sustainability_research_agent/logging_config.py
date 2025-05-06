import logging
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging():
    """
    Configures logging for the application.

    Reads the desired log level from the LOG_LEVEL environment variable,
    defaulting to INFO. Configures the root logger and sets specific
    levels for noisy libraries.
    """
    log_level_name = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Use basicConfig with force=True to ensure it can reconfigure
    # This is useful if any library tries to configure logging before this setup.
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],  # Explicitly log to stdout
        force=True,
    )

    # Set specific levels for noisy libraries
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    root_logger.info(f"Logging configured with level: {logging.getLevelName(root_logger.level)}")


if __name__ == "__main__":
    setup_logging()
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
