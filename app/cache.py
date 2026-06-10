# app/cache.py
# ============================================================
# FinanceIQ Advanced Cache Engine
# ============================================================

import os
import json
import time
import gzip
import pickle
import hashlib
import threading
import tempfile

from collections import OrderedDict
from functools import wraps

# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = os.getenv(
    "FINANCEIQ_CACHE_DIR",
    "cache_store"
)

DEFAULT_TTL = int(
    os.getenv(
        "FINANCEIQ_CACHE_TTL",
        300
    )
)

MAX_MEMORY_ITEMS = int(
    os.getenv(
        "FINANCEIQ_MAX_MEMORY_CACHE",
        500
    )
)

ENABLE_CACHE = os.getenv(
    "FINANCEIQ_ENABLE_CACHE",
    "true"
).lower() == "true"

CACHE_VERSION = "v2"

# ============================================================
# THREAD LOCKS
# ============================================================

_memory_lock = threading.RLock()
_disk_lock = threading.RLock()

# ============================================================
# MEMORY CACHE (LRU)
# ============================================================

_memory_cache = OrderedDict()

# ============================================================
# CACHE STATS
# ============================================================

_cache_stats = {

    "memory_hits": 0,

    "disk_hits": 0,

    "misses": 0,

    "writes": 0,

    "evictions": 0
}

# ============================================================
# ENSURE CACHE DIRECTORY
# ============================================================

def _ensure_cache_dir():

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

# ============================================================
# CLEAN EXPIRED MEMORY ITEMS
# ============================================================

def _cleanup_memory():

    now = time.time()

    expired = []

    for key, value in _memory_cache.items():

        if now > value["expiry"]:

            expired.append(key)

    for key in expired:

        del _memory_cache[key]

# ============================================================
# LRU EVICTION
# ============================================================

def _enforce_memory_limit():

    while len(_memory_cache) > MAX_MEMORY_ITEMS:

        _memory_cache.popitem(last=False)

        _cache_stats["evictions"] += 1

# ============================================================
# STABLE HASHING
# ============================================================

def _stable_serialize(obj):

    try:

        return json.dumps(
            obj,
            sort_keys=True,
            default=str
        )

    except:

        return repr(obj)

# ============================================================
# CACHE KEY GENERATION
# ============================================================

def _make_key(
    namespace,
    *args,
    **kwargs
):

    payload = {

        "version": CACHE_VERSION,

        "namespace": namespace,

        "args": args,

        "kwargs": kwargs
    }

    raw = _stable_serialize(payload)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

# ============================================================
# FILE PATH
# ============================================================

def _get_file_path(key):

    return os.path.join(
        CACHE_DIR,
        f"{key}.cache"
    )

# ============================================================
# SERIALIZATION
# ============================================================

def _serialize(value):

    return gzip.compress(
        pickle.dumps(value)
    )

# ============================================================
# DESERIALIZATION
# ============================================================

def _deserialize(blob):

    return pickle.loads(
        gzip.decompress(blob)
    )

# ============================================================
# MEMORY GET
# ============================================================

def get_memory(key):

    if not ENABLE_CACHE:

        return None

    with _memory_lock:

        _cleanup_memory()

        item = _memory_cache.get(key)

        if not item:

            _cache_stats["misses"] += 1

            return None

        if time.time() > item["expiry"]:

            del _memory_cache[key]

            _cache_stats["misses"] += 1

            return None

        # refresh LRU position
        _memory_cache.move_to_end(key)

        _cache_stats["memory_hits"] += 1

        return item["value"]

# ============================================================
# MEMORY SET
# ============================================================

def set_memory(
    key,
    value,
    ttl=DEFAULT_TTL
):

    if not ENABLE_CACHE:

        return

    with _memory_lock:

        expiry = time.time() + ttl

        _memory_cache[key] = {

            "value": value,

            "expiry": expiry
        }

        _memory_cache.move_to_end(key)

        _cleanup_memory()

        _enforce_memory_limit()

# ============================================================
# DISK GET
# ============================================================

def get_disk(key):

    if not ENABLE_CACHE:

        return None

    _ensure_cache_dir()

    path = _get_file_path(key)

    if not os.path.exists(path):

        return None

    with _disk_lock:

        try:

            with open(path, "rb") as f:

                payload = pickle.load(f)

            expiry = payload["expiry"]

            if time.time() > expiry:

                os.remove(path)

                return None

            value = _deserialize(
                payload["value"]
            )

            remaining_ttl = max(
                expiry - time.time(),
                1
            )

            # restore to memory
            set_memory(
                key,
                value,
                ttl=remaining_ttl
            )

            _cache_stats["disk_hits"] += 1

            return value

        except Exception as e:

            try:
                os.remove(path)
            except:
                pass

            print(
                "⚠️ Corrupted cache file:",
                str(e)
            )

            return None

# ============================================================
# DISK SET
# ============================================================

def set_disk(
    key,
    value,
    ttl=DEFAULT_TTL
):

    if not ENABLE_CACHE:

        return False

    _ensure_cache_dir()

    path = _get_file_path(key)

    expiry = time.time() + ttl

    payload = {

        "expiry": expiry,

        "created_at": time.time(),

        "value": _serialize(value)
    }

    with _disk_lock:

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=CACHE_DIR
            ) as tmp:

                pickle.dump(
                    payload,
                    tmp
                )

                temp_name = tmp.name

            os.replace(
                temp_name,
                path
            )

            _cache_stats["writes"] += 1

            return True

        except Exception as e:

            print(
                "⚠️ Cache write error:",
                str(e)
            )

            return False

# ============================================================
# GET CACHE
# ============================================================

def get_cache(key):

    value = get_memory(key)

    if value is not None:

        return value

    return get_disk(key)

# ============================================================
# SET CACHE
# ============================================================

def set_cache(
    key,
    value,
    ttl=DEFAULT_TTL
):

    set_memory(
        key,
        value,
        ttl
    )

    set_disk(
        key,
        value,
        ttl
    )

# ============================================================
# CACHE DECORATOR
# ============================================================

def cached(
    ttl=DEFAULT_TTL,
    namespace=None,
    enabled=True
):

    def decorator(func):

        func_namespace = (

            namespace

            or

            f"{func.__module__}."
            f"{func.__qualname__}"
        )

        @wraps(func)
        def wrapper(*args, **kwargs):

            if not ENABLE_CACHE or not enabled:

                return func(*args, **kwargs)

            force_refresh = kwargs.pop(
                "_force_refresh",
                False
            )

            dynamic_ttl = kwargs.pop(
                "_cache_ttl",
                ttl
            )

            key = _make_key(
                func_namespace,
                *args,
                **kwargs
            )

            if not force_refresh:

                cached_value = get_cache(key)

                if cached_value is not None:

                    return cached_value

            result = func(
                *args,
                **kwargs
            )

            try:

                set_cache(
                    key,
                    result,
                    dynamic_ttl
                )

            except Exception as e:

                print(
                    "⚠️ Cache store failed:",
                    str(e)
                )

            return result

        return wrapper

    return decorator

# ============================================================
# CACHE CLEANUP
# ============================================================

def cleanup_disk_cache():

    if not os.path.exists(CACHE_DIR):

        return 0

    removed = 0

    now = time.time()

    with _disk_lock:

        for filename in os.listdir(CACHE_DIR):

            path = os.path.join(
                CACHE_DIR,
                filename
            )

            try:

                with open(path, "rb") as f:

                    payload = pickle.load(f)

                expiry = payload.get(
                    "expiry",
                    0
                )

                if now > expiry:

                    os.remove(path)

                    removed += 1

            except:

                try:

                    os.remove(path)

                    removed += 1

                except:
                    pass

    return removed

# ============================================================
# CLEAR CACHE
# ============================================================

def clear_cache(
    clear_disk=True
):

    global _memory_cache

    with _memory_lock:

        _memory_cache = OrderedDict()

    removed = 0

    if clear_disk:

        if os.path.exists(CACHE_DIR):

            with _disk_lock:

                for filename in os.listdir(
                    CACHE_DIR
                ):

                    try:

                        os.remove(
                            os.path.join(
                                CACHE_DIR,
                                filename
                            )
                        )

                        removed += 1

                    except:
                        pass

    return removed

# ============================================================
# CACHE INFO
# ============================================================

def cache_info():

    with _memory_lock:

        memory_items = len(
            _memory_cache
        )

    disk_items = 0

    if os.path.exists(CACHE_DIR):

        disk_items = len(
            os.listdir(CACHE_DIR)
        )

    return {

        "enabled": ENABLE_CACHE,

        "memory_items": memory_items,

        "disk_items": disk_items,

        "max_memory_items":
            MAX_MEMORY_ITEMS,

        "stats": _cache_stats
    }

# ============================================================
# REMOVE SINGLE CACHE KEY
# ============================================================

def delete_cache_key(key):

    removed = False

    with _memory_lock:

        if key in _memory_cache:

            del _memory_cache[key]

            removed = True

    path = _get_file_path(key)

    if os.path.exists(path):

        try:

            os.remove(path)

            removed = True

        except:
            pass

    return removed

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n====================================")
    print("TESTING CACHE ENGINE")
    print("====================================\n")

    @cached(ttl=10)
    def expensive_function(x):

        print("Computing...")

        return x * 100

    print(expensive_function(5))

    print(expensive_function(5))

    print(expensive_function(10))

    print("\nCACHE INFO:")

    print(cache_info())

    print("\nCleaning expired cache...")

    removed = cleanup_disk_cache()

    print("Removed:", removed)