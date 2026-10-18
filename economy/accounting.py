"""Double-entry ledger (spec §13).

Every money movement is a transfer between two named accounts, so the sum of
all balances is always exactly the sum of explicit external injections (zero
by construction, because external accounts may go negative). Only accounts
flagged `external` or `credit` may hold a negative balance; every other
overdraft raises.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


class LedgerError(RuntimeError):
    pass


EXTERNAL = (
    "ext:investors",     # builder seed capital and entrant financing
    "ext:lender",        # credit lines
    "ext:enterprise",    # enterprise AI budgets come from the wider economy
    "ext:labor",         # builder salaries / overhead
    "ext:capex",         # equipment vendors, construction
    "ext:power",         # data-center energy
)


@dataclass
class Txn:
    month: int
    src: str
    dst: str
    amount: float
    memo: str


class Ledger:
    def __init__(self, keep_log: bool = False) -> None:
        self.balances: dict[str, float] = defaultdict(float)
        self.flows: dict[str, float] = defaultdict(float)   # per-month memo totals
        self.keep_log = keep_log
        self.log: list[Txn] = []
        self.month = 0
        for a in EXTERNAL:
            self.balances[a] = 0.0

    @staticmethod
    def may_go_negative(account: str) -> bool:
        return account.startswith("ext:")

    def transfer(self, src: str, dst: str, amount: float, memo: str, allow_partial: bool = False) -> float:
        """Move `amount` from src to dst. Returns amount actually moved."""
        if amount < 0:
            raise LedgerError(f"negative transfer {amount} {src}->{dst} ({memo})")
        if amount == 0:
            return 0.0
        if not self.may_go_negative(src):
            avail = self.balances[src]
            if amount > avail + 1e-9:
                if allow_partial:
                    amount = max(0.0, avail)
                else:
                    raise LedgerError(f"overdraft: {src} has {avail:.3f}, needs {amount:.3f} ({memo})")
        self.balances[src] -= amount
        self.balances[dst] += amount
        self.flows[memo] += amount
        if self.keep_log:
            self.log.append(Txn(self.month, src, dst, amount, memo))
        return amount

    def balance(self, account: str) -> float:
        return self.balances[account]

    def total(self) -> float:
        return float(sum(self.balances.values()))

    def check_invariants(self) -> None:
        if abs(self.total()) > 1e-6 * max(1.0, sum(abs(v) for v in self.balances.values())):
            raise LedgerError(f"money not conserved: total={self.total()}")
        for acct, bal in self.balances.items():
            if bal < -1e-6 and not self.may_go_negative(acct):
                raise LedgerError(f"negative balance {acct}={bal}")

    def start_month(self, month: int) -> None:
        self.month = month
        self.flows = defaultdict(float)
