import logging
import functools
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

def log_function_call(func: F) -> F:
    """
    Decorator to log function entry and exit with parameters and return value.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func_name} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise
    
    return cast(F, wrapper)

def log_file_processing(func: F) -> F:
    """
    Decorator specifically for file processing functions.
    """
    @functools.wraps(func)
    def wrapper(file_path: str, *args: Any, **kwargs: Any) -> Any:
        file_extension = Path(file_path).suffix.lower()
        logger.info(f"Processing file: {file_path}")
        logger.debug(f"Processing file with extension: {file_extension}")
        
        start_time = time.time()
        try:
            result = func(file_path, *args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Processed file {file_path} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    return cast(F, wrapper)

def log_data_import(func: F) -> F:
    """
    Decorator specifically for data import functions.
    """
    @functools.wraps(func)
    def wrapper(data_source: str, username: str, scope_name: str, *args: Any, **kwargs: Any) -> Any:
        logger.info(f"Starting data import from {data_source} for user {username}, scope {scope_name}")
        
        start_time = time.time()
        try:
            result = func(data_source, username, scope_name, *args, **kwargs)
            elapsed = time.time() - start_time
            
            # Log completion details
            if isinstance(result, list):
                logger.info(f"Sent {len(result)} chunks to Celery for embedding")
            
            logger.info(f"Completed data import from {data_source} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.exception(f"Error during data import from {data_source}: {str(e)}")
            raise
    
    return cast(F, wrapper)
def log_directory_processing(func: F) -> F:
    """
    Decorator for directory processing functions.
    """
    @functools.wraps(func)
    def wrapper(directory: str, *args: Any, **kwargs: Any) -> Any:
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in [
                    ".pdf", ".md", ".docx", ".html", ".txt",
                ]:
                    file_paths.append(file_path)
        
        logger.info(
            f"Found {len(file_paths)} supported files to process in directory {directory}\n"
            f"File types: {', '.join(sorted({Path(fp).suffix.lower() for fp in file_paths}))}"
        )
        
        return func(directory, file_paths, *args, **kwargs)
    
    return cast(F, wrapper)
