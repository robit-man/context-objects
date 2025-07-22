# event_bus.py
import asyncio, time
from typing import Any, AsyncIterator, Dict, List

class EventBus:
    def __init__(self, maxbuf: int = 2000):
        self._subs: List[asyncio.Queue] = []
        self._history: List[Dict[str, Any]] = []
        self._maxbuf = maxbuf

    def publish(self, evt: Dict[str, Any]):
        evt.setdefault("ts", time.time())
        self._history.append(evt)
        if len(self._history) > self._maxbuf:
            self._history = self._history[-self._maxbuf:]
        for q in list(self._subs):
            if q.full():
                try: q.get_nowait()
                except: pass
            q.put_nowait(evt)

    async def subscribe(self) -> AsyncIterator[Dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subs.append(q)
        try:
            for h in self._history[-200:]:
                await q.put(h)
            while True:
                yield await q.get()
        finally:
            self._subs.remove(q)

BUS = EventBus()
