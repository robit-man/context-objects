# safe_repo.py
import threading, queue, time, random
import sqlite3  # or apsw; the retry catches OperationalError generically

class SafeRepo:
    """Serialize writes, retry on SQLITE_BUSY, and keep read ops short."""
    def __init__(self, inner, max_retries=8, base_sleep=0.010):
        self._inner = inner
        self._write_q = queue.Queue()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="RepoWriter")
        self._shutdown = threading.Event()
        self._max_retries = max_retries
        self._base_sleep = base_sleep
        self._writer.start()
        self._rw_lock = threading.RLock()  # protects read-phase against concurrent close

    def close(self):
        self._shutdown.set()
        self._write_q.put(None)
        try:
            self._writer.join(timeout=2)
        except Exception:
            pass
        # try to close underlying repo if it supports it
        try:
            if hasattr(self._inner, "close"):
                self._inner.close()
        except Exception:
            pass

    # ---------- public API (minimal: query & save) ----------
    def query(self, predicate):
        # Keep reads short; retry on locked
        with self._rw_lock:
            def _do():
                return self._inner.query(predicate)
            return self._retry(_do)

    def save(self, obj):
        # Do writes in the single writer thread; block caller until done
        done = queue.Queue(maxsize=1)
        self._write_q.put((obj, done))
        ok, val = done.get()
        if ok:
            return val
        raise val  # re-raise error

    # ---------- internals ----------
    def _writer_loop(self):
        while not self._shutdown.is_set():
            item = self._write_q.get()
            if item is None:
                break
            obj, done = item
            try:
                def _do():
                    return self._inner.save(obj)
                res = self._retry(_do)
                done.put((True, res))
            except Exception as e:
                done.put((False, e))

    def _retry(self, fn):
        delay = self._base_sleep
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:
                msg = str(e).lower()
                # sqlite3.OperationalError: database is locked / busy
                if ("locked" in msg or "busy" in msg):
                    time.sleep(delay + random.random() * delay * 0.5)
                    delay = min(delay * 2, 0.300)  # cap at 300ms
                    continue
                raise
        # last try
        return fn()
