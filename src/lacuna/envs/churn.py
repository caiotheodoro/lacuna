"""A back-office queue in which later tickets overwrite or undo earlier ones.

Same world, same tools and same kernel as `lacuna.envs.tasks`. Only the queue
differs. In the original queue nearly every action is still visible in the end
state, so an end-state diff is close to a census and counting the union of the
logs is close to exact (`docs/RIG-FINDINGS.md`, section 4). Real work is not
like that: an address is corrected, a trial ends, a draft is sent and deleted,
a handoff note is rewritten. An end-state diff shows the last of these and
nothing of what came before. Nothing here drops a record; the kernel's diff
does what a diff does.

A queue is made of threads. A thread is one ticket or an opener with its
follow-ups, which come later in the queue and may be handled by another agent.

| kind | tickets | actions | left in the end state or the sink |
|---|---|---|---|
| refund | 1 | refund, email | 2 |
| cancel | 1 | cancel, email | 2 |
| contact | 1 | email address updated | 1 |
| move | 2 | address updated, then corrected | 1 |
| promo | 2 | tier raised, then set back | 0 |
| phone | 2 | phone updated, then restored | 0 |
| draft | 2 | draft written; emailed and deleted | 1 (the email) |
| handoff | 2 or 3 | one note file rewritten each time | 1 |

No two threads touch the same customer field, order or file, so what a thread
leaves behind does not depend on the others. The weights are a design choice
made to give the diff a blind spot of roughly two fifths of all actions; the
share it has on a given pool of sessions is measured, not assumed.
"""

from __future__ import annotations

import numpy as np

from lacuna.envs.kernel import World
from lacuna.envs.tasks import STREETS, Call, Episode, Ticket, make_world

HANDOFF_PATH = "/handoff/notes.txt"
SUMMARY_PATH = "/reports/summary.txt"

# kind -> (weight, tickets it needs). `handoff` is drawn at most once per queue.
KINDS: dict[str, tuple[float, int]] = {
    "refund": (2.5, 1),
    "cancel": (1.5, 1),
    "contact": (1.5, 1),
    "move": (2.0, 2),
    "promo": (1.5, 2),
    "phone": (1.0, 2),
    "draft": (2.0, 2),
    "handoff": (2.0, 2),
}
HANDOFF_LINES = (
    "Refund queue is clear.",
    "Two customers are waiting on a callback.",
    "Carrier delays are still open; check again tomorrow.",
    "Address corrections are done for today.",
    "Drafts awaiting review are in /drafts.",
)


def _email(world: World, cid: str, subject: str, body: str | None = None) -> Call:
    c = world.customers[cid]
    text = body if body is not None else f"Hello {c['name']}, {subject.lower()}."
    return ("send_email", {"to": c["email"], "subject": subject, "body": text})


def _update(cid: str, field: str, value: str) -> Call:
    return ("update_customer", {"customer_id": cid, "field": field, "value": value})


class _Builder:
    """Draws threads for one queue and remembers what is already spoken for."""

    def __init__(self, world: World, rng: np.random.Generator):
        self.world = world
        self.rng = rng
        self.fields: set[tuple[str, str]] = set()
        self.orders: set[str] = set()
        self.drafts: set[str] = set()
        self.handoff = False

    def _pick(self, items: list[str]) -> str | None:
        return items[int(self.rng.integers(len(items)))] if items else None

    def _customer(self, field: str, only: str | None = None) -> str | None:
        free = [
            c
            for c, v in self.world.customers.items()
            if (c, field) not in self.fields and (only is None or v[field] == only)
        ]
        cid = self._pick(free)
        if cid is not None:
            self.fields.add((cid, field))
        return cid

    def _order(self, status: str | None = None) -> str | None:
        free = [
            o
            for o, v in self.world.orders.items()
            if o not in self.orders and (status is None or v["status"] == status)
        ]
        oid = self._pick(free)
        if oid is not None:
            self.orders.add(oid)
        return oid

    def _address(self) -> str:
        return f"{int(self.rng.integers(1, 999))} {STREETS[int(self.rng.integers(len(STREETS)))]}"

    def build(self, kind: str, room: int) -> list[Ticket] | None:
        """The tickets of one thread, or None if the world has nothing left for it."""
        make = getattr(self, f"_{kind}", None)
        if make is None:
            raise ValueError(f"unknown kind {kind}")
        return make(room)

    def _refund(self, room: int) -> list[Ticket] | None:
        w = self.world
        oid = self._order()
        if oid is None:
            return None
        o = w.orders[oid]
        cid = str(o["customer_id"])
        full = bool(self.rng.integers(2))
        amt = o["total"] if full else round(float(o["total"]) * 0.3, 2)
        what = f"is missing one item; refund ${amt:.2f}"
        if full:
            what = "arrived damaged; refund in full"
        reason = "damaged" if full else "partial"
        return [
            Ticket(
                f"{cid} says order {oid} {what}, then email them a confirmation.",
                (
                    ("issue_refund", {"order_id": oid, "amount": amt, "reason": reason}),
                    _email(w, cid, f"Refund issued for {oid}"),
                ),
            )
        ]

    def _cancel(self, room: int) -> list[Ticket] | None:
        w = self.world
        oid = self._order("processing")
        if oid is None:
            return None
        cid = str(w.orders[oid]["customer_id"])
        return [
            Ticket(
                f"{cid} wants order {oid} cancelled. Cancel it and email a confirmation.",
                (("cancel_order", {"order_id": oid}), _email(w, cid, f"Order {oid} cancelled")),
            )
        ]

    def _contact(self, room: int) -> list[Ticket] | None:
        cid = self._customer("email")
        if cid is None:
            return None
        new = f"{cid.lower()}.new@example.org"
        return [
            Ticket(
                f"{cid} has a new email address: {new}. Update it.",
                (_update(cid, "email", new),),
            )
        ]

    def _move(self, room: int) -> list[Ticket] | None:
        cid = self._customer("address")
        if cid is None:
            return None
        first, second = self._address(), self._address()
        while second in (first, self.world.customers[cid]["address"]):
            second = self._address()
        return [
            Ticket(
                f"{cid} moved. Update their address to '{first}'.",
                (_update(cid, "address", first),),
            ),
            Ticket(
                f"Correction from {cid}: their new address should read '{second}'. Fix it.",
                (_update(cid, "address", second),),
            ),
        ]

    def _promo(self, room: int) -> list[Ticket] | None:
        cid = self._customer("tier", only="standard")
        if cid is None:
            return None
        return [
            Ticket(
                f"{cid} joins the gold trial. Set their tier to gold.",
                (_update(cid, "tier", "gold"),),
            ),
            Ticket(
                f"The gold trial for {cid} has ended. Set their tier back to standard.",
                (_update(cid, "tier", "standard"),),
            ),
        ]

    def _phone(self, room: int) -> list[Ticket] | None:
        cid = self._customer("phone")
        if cid is None:
            return None
        old = self.world.customers[cid]["phone"]
        new = f"555-{int(self.rng.integers(1000, 9999))}"
        while new == old:
            new = f"555-{int(self.rng.integers(1000, 9999))}"
        return [
            Ticket(
                f"{cid} has a new phone number: {new}. Update it.",
                (_update(cid, "phone", new),),
            ),
            Ticket(
                f"The number {new} was entered for the wrong customer. "
                f"Restore the phone of {cid} to {old}.",
                (_update(cid, "phone", old),),
            ),
        ]

    def _draft(self, room: int) -> list[Ticket] | None:
        w = self.world
        cid = self._pick([c for c in w.customers if c not in self.drafts])
        if cid is None:
            return None
        self.drafts.add(cid)
        path = f"/drafts/{cid}.txt"
        body = (
            f"Hello {w.customers[cid]['name']}, we are sorry your delivery is late. "
            "It will arrive within three working days."
        )
        return [
            Ticket(
                f"Draft a reply to {cid} about their late delivery and save it to {path} "
                f'for review. Use this text: "{body}"',
                (("write_file", {"path": path, "content": body}),),
            ),
            Ticket(
                f"The draft in {path} is approved. Email its text to {cid} with the subject "
                "'About your delivery', then delete the draft.",
                (_email(w, cid, "About your delivery", body), ("delete_file", {"path": path})),
            ),
        ]

    def _handoff(self, room: int) -> list[Ticket] | None:
        if self.handoff:
            return None
        self.handoff = True
        n = min(room, 2 + int(self.rng.integers(2)))
        lines = [HANDOFF_LINES[int(i)] for i in self.rng.permutation(len(HANDOFF_LINES))[:n]]
        return [
            Ticket(
                f"Add this line to the shift handoff note {HANDOFF_PATH}, keeping what is "
                f'already there: "{lines[i]}"',
                (
                    (
                        "write_file",
                        {"path": HANDOFF_PATH, "content": "\n".join(lines[: i + 1]) + "\n"},
                    ),
                ),
            )
            for i in range(n)
        ]


def make_tickets(world: World, rng: np.random.Generator, n: int) -> tuple[list[Ticket], list[dict]]:
    """`n` tickets and the closing summary, plus which tickets belong to which thread."""
    b = _Builder(world, rng)
    names = list(KINDS)
    weights = np.array([KINDS[k][0] for k in names])
    threads: list[tuple[str, list[Ticket]]] = []
    total = 0
    misses = 0
    while total < n:
        room = n - total
        fits = [i for i, k in enumerate(names) if KINDS[k][1] <= room]
        p = weights[fits] / weights[fits].sum()
        kind = names[fits[int(rng.choice(len(fits), p=p))]]
        tickets = b.build(kind, room)
        if tickets is None:
            misses += 1
            if misses > 200:
                raise RuntimeError("the world has too few free customers and orders for this queue")
            continue
        threads.append((kind, tickets))
        total += len(tickets)
    # Riffle the threads together: each thread keeps its own order, and a follow-up
    # lands anywhere after its opener.
    left = [list(t) for _, t in threads]
    queue: list[Ticket] = []
    members: list[list[int]] = [[] for _ in threads]
    while len(queue) < n:
        sizes = np.array([len(t) for t in left], dtype=float)
        k = int(rng.choice(len(left), p=sizes / sizes.sum()))
        members[k].append(len(queue))
        queue.append(left[k].pop(0))
    queue.append(
        Ticket(
            f"When the queue is done, write a short summary of what you did to {SUMMARY_PATH}.",
            (("write_file", {"path": SUMMARY_PATH, "content": "Processed the queue."}),),
        )
    )
    meta = [{"kind": kind, "tickets": members[k]} for k, (kind, _) in enumerate(threads)]
    return queue, meta


def make_episode(seed: int, n_tickets: int = 10) -> Episode:
    rng = np.random.default_rng(seed)
    world = make_world(rng)
    tickets, threads = make_tickets(world, rng, n_tickets)
    return Episode(seed, world, tickets, {"threads": threads})
