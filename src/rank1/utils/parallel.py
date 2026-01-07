"""
Parallel execution utilities using joblib and multiprocessing.
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar

from joblib import Parallel, delayed
from tqdm import tqdm

from rank1.logging import get_logger

logger = get_logger()

T = TypeVar("T")
R = TypeVar("R")


def get_n_workers(n_jobs: int = -1) -> int:
    """
    Get number of workers to use.

    Args:
        n_jobs: Number of jobs (-1 means all but one CPU)

    Returns:
        Number of workers
    """
    n_cpus = os.cpu_count() or 1

    if n_jobs == -1:
        return max(1, n_cpus - 1)
    elif n_jobs < 0:
        return max(1, n_cpus + 1 + n_jobs)
    else:
        return min(n_jobs, n_cpus)


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    n_jobs: int = -1,
    backend: str = "loky",
    desc: Optional[str] = None,
    show_progress: bool = True,
    batch_size: str = "auto",
) -> list[R]:
    """
    Parallel map using joblib with progress bar.

    Args:
        func: Function to apply to each item
        items: Sequence of items to process
        n_jobs: Number of parallel jobs (-1 for all but one CPU)
        backend: Joblib backend ('loky', 'threading', 'multiprocessing')
        desc: Description for progress bar
        show_progress: Whether to show progress bar
        batch_size: Batch size for joblib

    Returns:
        List of results
    """
    n_items = len(items)

    if n_items == 0:
        return []

    n_workers = get_n_workers(n_jobs)

    # For very small workloads, don't bother with parallelism
    if n_items <= 2 or n_workers == 1:
        if show_progress and desc:
            return [func(item) for item in tqdm(items, desc=desc)]
        return [func(item) for item in items]

    logger.debug(f"Running {n_items} tasks with {n_workers} workers")

    if show_progress and desc:
        results = Parallel(n_jobs=n_workers, backend=backend, batch_size=batch_size)(
            delayed(func)(item) for item in tqdm(items, desc=desc)
        )
    else:
        results = Parallel(n_jobs=n_workers, backend=backend, batch_size=batch_size)(
            delayed(func)(item) for item in items
        )

    return list(results)


def parallel_starmap(
    func: Callable[..., R],
    args_list: Sequence[tuple],
    n_jobs: int = -1,
    backend: str = "loky",
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> list[R]:
    """
    Parallel starmap using joblib.

    Args:
        func: Function to call with unpacked arguments
        args_list: List of argument tuples
        n_jobs: Number of parallel jobs
        backend: Joblib backend
        desc: Progress bar description
        show_progress: Whether to show progress bar

    Returns:
        List of results
    """
    n_items = len(args_list)

    if n_items == 0:
        return []

    n_workers = get_n_workers(n_jobs)

    if n_items <= 2 or n_workers == 1:
        if show_progress and desc:
            return [func(*args) for args in tqdm(args_list, desc=desc)]
        return [func(*args) for args in args_list]

    if show_progress and desc:
        results = Parallel(n_jobs=n_workers, backend=backend)(
            delayed(func)(*args) for args in tqdm(args_list, desc=desc)
        )
    else:
        results = Parallel(n_jobs=n_workers, backend=backend)(
            delayed(func)(*args) for args in args_list
        )

    return list(results)


class ParallelExecutor:
    """
    Context manager for parallel execution with automatic cleanup.
    """

    def __init__(
        self,
        n_workers: int = -1,
        executor_type: str = "process",
    ):
        """
        Initialize executor.

        Args:
            n_workers: Number of workers (-1 for auto)
            executor_type: 'process' or 'thread'
        """
        self.n_workers = get_n_workers(n_workers)
        self.executor_type = executor_type
        self._executor: Optional[ProcessPoolExecutor | ThreadPoolExecutor] = None

    def __enter__(self) -> "ParallelExecutor":
        if self.executor_type == "process":
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
        return False

    def map(
        self,
        func: Callable[[T], R],
        items: Sequence[T],
        desc: Optional[str] = None,
    ) -> Iterator[R]:
        """
        Map function over items.

        Args:
            func: Function to apply
            items: Items to process
            desc: Progress bar description

        Yields:
            Results as they complete
        """
        if not self._executor:
            raise RuntimeError("Executor not initialized. Use as context manager.")

        futures = {self._executor.submit(func, item): i for i, item in enumerate(items)}

        results = [None] * len(items)

        if desc:
            pbar = tqdm(total=len(items), desc=desc)
        else:
            pbar = None

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return iter(results)

    def submit(self, func: Callable[..., R], *args, **kwargs):
        """Submit a task to the executor."""
        if not self._executor:
            raise RuntimeError("Executor not initialized. Use as context manager.")
        return self._executor.submit(func, *args, **kwargs)


def get_executor(
    n_workers: int = -1,
    executor_type: str = "process",
) -> ParallelExecutor:
    """
    Get a parallel executor.

    Args:
        n_workers: Number of workers
        executor_type: 'process' or 'thread'

    Returns:
        ParallelExecutor context manager
    """
    return ParallelExecutor(n_workers, executor_type)


def chunked(items: Sequence[T], chunk_size: int) -> Iterator[list[T]]:
    """
    Split sequence into chunks.

    Args:
        items: Sequence to split
        chunk_size: Size of each chunk

    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield list(items[i : i + chunk_size])
