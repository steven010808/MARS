from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol


class SessionStore(Protocol):
    def get_context(self, user_id: str, session_id: str | None = None) -> dict[str, Any]: ...

    def update_event(self, event: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class InMemorySessionStore:
    recent_n: int = 20
    _user_events: dict[str, deque[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    _session_events: dict[str, deque[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    def get_context(self, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            events = list(self._session_events.get(_session_key(user_id, session_id), []))
        else:
            events = list(self._user_events.get(user_id, []))
        return _events_to_context(events[-self.recent_n :])

    def update_event(self, event: dict[str, Any]) -> dict[str, Any]:
        user_id = str(event.get("user_id", ""))
        session_id = event.get("session_id")
        if user_id:
            self._append(self._user_events[user_id], event)
        if user_id and session_id:
            self._append(self._session_events[_session_key(user_id, str(session_id))], event)
        return self.get_context(user_id, str(session_id) if session_id else None)

    def _append(self, queue: deque[dict[str, Any]], event: dict[str, Any]) -> None:
        queue.append(dict(event))
        while len(queue) > self.recent_n:
            queue.popleft()


class RedisSessionStore:
    def __init__(self, redis_client: Any, recent_n: int = 20) -> None:
        self.redis = redis_client
        self.recent_n = recent_n
        self.fallback = InMemorySessionStore(recent_n=recent_n)

    def get_context(self, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        try:
            key = self._context_key(user_id, session_id)
            raw_events = self.redis.lrange(key, 0, self.recent_n - 1) if key else []
            # Redis lists use LPUSH, so convert newest-first storage back to chronological order.
            events = [_loads(raw) for raw in reversed(raw_events)]
            return _events_to_context(events[-self.recent_n :])
        except Exception:
            return self.fallback.get_context(user_id, session_id)

    def update_event(self, event: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = _dumps(event)
            for key in self._keys(str(event.get("user_id", "")), event.get("session_id")):
                self.redis.lpush(key, payload)
                self.redis.ltrim(key, 0, self.recent_n - 1)
            return self.get_context(str(event.get("user_id", "")), event.get("session_id"))
        except Exception:
            return self.fallback.update_event(event)

    def _keys(self, user_id: str, session_id: str | None = None) -> list[str]:
        keys = []
        if user_id:
            keys.append(f"user:{user_id}:recent_events")
        if user_id and session_id:
            keys.append(f"session:{user_id}:{session_id}:recent_events")
        return keys

    def _context_key(self, user_id: str, session_id: str | None = None) -> str:
        if user_id and session_id:
            return f"session:{user_id}:{session_id}:recent_events"
        if user_id:
            return f"user:{user_id}:recent_events"
        return ""


def _events_to_context(events: list[dict[str, Any]]) -> dict[str, Any]:
    recent_products: list[str] = []
    recent_categories: list[str] = []
    event_counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        event_role = str(metadata.get("event_role", ""))
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        product_id = event.get("product_id")
        category = (
            event.get("category")
            or metadata.get("category")
            or event.get("query_intent_category")
            or metadata.get("query_intent_category")
        )
        if product_id and event_type in {"view", "cart", "purchase"} and event_role != "exposure":
            recent_products.append(str(product_id))
        if category and event_role != "exposure":
            recent_categories.append(str(category))
    return {
        "recent_products": _dedupe_keep_order(reversed(recent_products)),
        "recent_categories": _dedupe_keep_order(reversed(recent_categories)),
        "event_counts": event_counts,
        "num_recent_events": len(events),
    }


def _dedupe_keep_order(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _session_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def _dumps(event: dict[str, Any]) -> str:
    import json

    return json.dumps(event, ensure_ascii=False, default=str)


def _loads(raw: Any) -> dict[str, Any]:
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return dict(json.loads(raw))
