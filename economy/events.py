"""Market events and the delayed-effect queue (spec §10)."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    TECH_BREAKTHROUGH = "TECH_BREAKTHROUGH"
    FAB_SHORTAGE = "FAB_SHORTAGE"
    FAB_RECOVERY = "FAB_RECOVERY"
    CHIP_DELIVERY = "CHIP_DELIVERY"
    CAPACITY_ONLINE = "CAPACITY_ONLINE"
    MODEL_RELEASE = "MODEL_RELEASE"
    PROJECT_GATE_RESULT = "PROJECT_GATE_RESULT"
    ENTERPRISE_ADOPTION = "ENTERPRISE_ADOPTION"
    CUSTOMER_CHURN = "CUSTOMER_CHURN"
    PRICE_UPDATE = "PRICE_UPDATE"
    FIRM_ENTRY = "FIRM_ENTRY"
    BANKRUPTCY = "BANKRUPTCY"
    OPPORTUNITY_DISCOVERED = "OPPORTUNITY_DISCOVERED"
    ILLEGAL_ACTION = "ILLEGAL_ACTION"


# Events whose payload is public (appears in observations / viewer as market news).
PUBLIC_EVENTS = {
    EventType.TECH_BREAKTHROUGH, EventType.MODEL_RELEASE, EventType.FIRM_ENTRY, EventType.BANKRUPTCY,
    EventType.CAPACITY_ONLINE, EventType.OPPORTUNITY_DISCOVERED, EventType.PRICE_UPDATE,
}


@dataclass
class MarketEvent:
    month: int
    kind: EventType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"month": self.month, "kind": self.kind.value, "payload": self.payload}


class EventQueue:
    """Min-heap of scheduled future events, ordered by (month, insertion order)."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, MarketEvent]] = []
        self._counter = 0

    def schedule(self, event: MarketEvent) -> None:
        heapq.heappush(self._heap, (event.month, self._counter, event))
        self._counter += 1

    def pop_due(self, month: int) -> list[MarketEvent]:
        due = []
        while self._heap and self._heap[0][0] <= month:
            due.append(heapq.heappop(self._heap)[2])
        return due

    def pending(self) -> list[MarketEvent]:
        return [e for _, _, e in sorted(self._heap)]

    def __len__(self) -> int:
        return len(self._heap)
