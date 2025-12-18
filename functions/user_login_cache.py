from __future__ import annotations
import json
from pathlib import Path
from threading import Lock
from collections.abc import Iterable
from typing import Dict, List, Set
_CACHE_LOCK = Lock()
_CACHE_FILE = Path(__file__).resolve().parents[1] / "working_login_cache.json"
_MAX_FAILURES = 3


def _empty_cache() -> Dict[str, object]:
    return {"working_user_ids": [], "failure_counts": {}}


def _read_cache() -> Dict[str, object]:
    cache = _empty_cache()
    if not _CACHE_FILE.exists():
        return cache

    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return cache

    working_ids = data.get("working_user_ids", [])
    if isinstance(working_ids, Iterable):
        cache["working_user_ids"] = list({str(uid) for uid in working_ids if uid is not None})

    failure_counts = data.get("failure_counts", {})
    if isinstance(failure_counts, dict):
        parsed_failures: Dict[str, int] = {}
        for key, value in failure_counts.items():
            try:
                parsed_failures[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        cache["failure_counts"] = parsed_failures

    return cache


def _write_cache(cache: Dict[str, object]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def record_login_success(user_id: object) -> None:
    """Mark a login as successful and reset its failure counter."""
    if user_id is None:
        return
    user_id_str = str(user_id)

    with _CACHE_LOCK:
        cache = _read_cache()
        raw_working_ids = cache.get("working_user_ids", [])
        raw_failure_counts = cache.get("failure_counts", {})

        working_ids: List[str]
        if isinstance(raw_working_ids, list):
            working_ids = [str(uid) for uid in raw_working_ids]
        elif isinstance(raw_working_ids, Iterable):
            working_ids = [str(uid) for uid in raw_working_ids]
        else:
            working_ids = []

        if isinstance(raw_failure_counts, dict):
            failure_counts = {str(key): int(value) for key, value in raw_failure_counts.items()}
        else:
            failure_counts = {}

        if user_id_str not in working_ids:
            working_ids.append(user_id_str)
        failure_counts[user_id_str] = 0

        cache["working_user_ids"] = working_ids
        cache["failure_counts"] = failure_counts
        _write_cache(cache)


def record_login_failure(user_id: object, *, max_failures: int = _MAX_FAILURES) -> None:
    """Increment the failure counter for a login and evict if it keeps failing."""
    if user_id is None:
        return
    user_id_str = str(user_id)

    with _CACHE_LOCK:
        cache = _read_cache()
        raw_working_ids = cache.get("working_user_ids", [])
        raw_failure_counts = cache.get("failure_counts", {})

        if isinstance(raw_working_ids, list):
            working_ids = [str(uid) for uid in raw_working_ids]
        elif isinstance(raw_working_ids, Iterable):
            working_ids = [str(uid) for uid in raw_working_ids]
        else:
            working_ids = []

        if isinstance(raw_failure_counts, dict):
            failure_counts = {str(key): int(value) for key, value in raw_failure_counts.items()}
        else:
            failure_counts = {}

        failures = failure_counts.get(user_id_str, 0) + 1
        failure_counts[user_id_str] = failures

        if user_id_str in working_ids and failures >= max_failures:
            working_ids.remove(user_id_str)

        cache["working_user_ids"] = working_ids
        cache["failure_counts"] = failure_counts
        _write_cache(cache)


def get_working_user_ids() -> Set[str]:
    """Return the cached set of working user ids."""
    with _CACHE_LOCK:
        cache = _read_cache()
        working_ids = cache.get("working_user_ids", [])
        if not isinstance(working_ids, Iterable):
            return set()
        return {str(uid) for uid in working_ids}


def filter_working_logins(login_details_list: Iterable[dict]) -> List[dict]:
    """Filter login records to the cached working set, falling back to the original list."""
    login_details = list(login_details_list)
    if not login_details:
        return []

    working_ids = get_working_user_ids()
    if not working_ids:
        return login_details

    filtered = [login for login in login_details if str(login.get("user_id")) in working_ids]
    return filtered or login_details
