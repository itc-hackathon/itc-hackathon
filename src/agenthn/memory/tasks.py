"""Synthetic agentic-memory task.

Unlike NIAH (one needle in a haystack of noise), this models an AGENT working a
long-horizon session: it accumulates many *discrete, independent observations* —
tool outputs, config it set, records it saw — each of which it may need to recall
later. Every entry is a real fact (multi-needle), and every value is randomized
and unguessable so a base model cannot answer from priors (it must have stored
the observation). This is the regime where text-memory blows up the context
window but weight-memory stays ~free.

generate_session(n, seed) -> (entries, probes)
  entries: list[MemoryEntry]  — what the agent observed, in order.
  probes:  list[Probe]        — (question, answer) recall queries, one per entry.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    text: str          # the observation, as the agent would journal it
    key: str           # short tag, for display
    answer: str        # the value a probe should recover


@dataclass
class Probe:
    question: str
    answer: str
    entry_idx: int     # which entry holds the answer


# --- random unguessable value generators ---------------------------------
def _token(r, k=6):
    return "".join(r.choices(string.ascii_lowercase + string.digits, k=k))


def _hexid(r, k=6):
    return "".join(r.choices("0123456789abcdef", k=k))


def _num(r, lo, hi):
    return str(r.randint(lo, hi))


# Entity pools (the *subject* varies run-to-run; the *value* is always random).
_COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella", "Soylent", "Hooli",
              "Vandelay", "Wonka", "Stark Industries", "Wayne Enterprises",
              "Cyberdyne", "Tyrell", "Massive Dynamic", "Pied Piper"]
_SERVICES = ["billing", "auth", "search", "checkout", "inventory", "notifications",
             "analytics", "gateway", "scheduler", "recommender"]
_PEOPLE = ["Dana", "Priya", "Marcus", "Wei", "Sofia", "Omar", "Lena", "Diego",
           "Yuki", "Tomas", "Ingrid", "Raj"]
_PROJECTS = ["the mobile relaunch", "the data migration", "the pricing revamp",
             "the onboarding flow", "the search rewrite", "the API v2 cutover"]
_ARTIFACTS = ["nightly backup", "model checkpoint", "incident report",
              "load-test results", "schema dump", "audit log"]


def _f_account(r):
    c = r.choice(_COMPANIES)
    v = _num(r, 10000, 99999)
    return MemoryEntry(
        text=f"Customer {c} was assigned account number {v}.",
        key=f"account[{c}]",
        answer=v,
    ), (f"What account number was assigned to {c}? Reply with only the number.", v)


def _f_region(r):
    s = r.choice(_SERVICES)
    v = r.choice(["us-east", "us-west", "eu-west", "eu-central", "ap-south",
                  "ap-northeast", "sa-east"])
    return MemoryEntry(
        text=f"I migrated the {s} service to the {v} region.",
        key=f"region[{s}]",
        answer=v,
    ), (f"Which region is the {s} service in now? Reply with only the region.", v)


def _f_team(r):
    s = r.choice(_SERVICES)
    v = r.choice(["Platform", "Growth", "Payments", "Infra", "Trust", "Mobile",
                  "Data", "Reliability"])
    return MemoryEntry(
        text=f"Ownership of the {s} service was handed to the {v} team.",
        key=f"team[{s}]",
        answer=v,
    ), (f"Which team owns the {s} service? Reply with only the team name.", v)


def _f_tier(r):
    c = r.choice(_COMPANIES)
    v = r.choice(["bronze", "silver", "gold", "platinum", "enterprise"])
    return MemoryEntry(
        text=f"Customer {c} was upgraded to the {v} support tier.",
        key=f"tier[{c}]",
        answer=v,
    ), (f"Which support tier is {c} on? Reply with only the tier.", v)


def _f_port(r):
    s = r.choice(_SERVICES)
    v = _num(r, 3000, 9999)
    return MemoryEntry(
        text=f"The {s} service is now bound to port {v}.",
        key=f"port[{s}]",
        answer=v,
    ), (f"Which port is the {s} service bound to? Reply with only the number.", v)


def _f_oncall(r):
    s = r.choice(_SERVICES)
    p = r.choice(_PEOPLE)
    return MemoryEntry(
        text=f"{p} is the on-call engineer for the {s} service this week.",
        key=f"oncall[{s}]",
        answer=p,
    ), (f"Who is on call for the {s} service? Reply with only the name.", p)


def _f_codename(r):
    proj = r.choice(_PROJECTS)
    v = r.choice(["Falcon", "Cobalt", "Mango", "Quartz", "Nimbus", "Basil",
                  "Tundra", "Orchid", "Pelican", "Saffron"]) + "-" + _hexid(r, 3)
    return MemoryEntry(
        text=f"The internal codename for {proj} is {v}.",
        key=f"codename[{proj}]",
        answer=v,
    ), (f"What is the internal codename for {proj}? Reply with only the codename.", v)


def _f_path(r):
    a = r.choice(_ARTIFACTS)
    v = f"/srv/{_token(r,4)}/{a.split()[0]}-{_hexid(r,4)}.tar.gz"
    return MemoryEntry(
        text=f"I stored the latest {a} at {v}.",
        key=f"path[{a}]",
        answer=v,
    ), (f"Where did I store the latest {a}? Reply with only the path.", v)


def _f_ticket(r):
    c = r.choice(_COMPANIES)
    v = "T-" + _num(r, 1000, 9999)
    return MemoryEntry(
        text=f"{c}'s escalation is tracked in ticket {v}.",
        key=f"ticket[{c}]",
        answer=v,
    ), (f"Which ticket tracks {c}'s escalation? Reply with only the ticket id.", v)


# --- "hard recall" factories: long random strings the model CANNOT reconstruct
# from a 2B internalized adapter (verified ~0% recall — pure multi-token noise).
# Excluded from the default mix; kept to demonstrate the recall ceiling.
def _f_token(r):
    s = r.choice(_SERVICES)
    v = "tk-" + _token(r, 8)
    return MemoryEntry(text=f"I rotated the {s} service token to {v}.",
                       key=f"token[{s}]", answer=v), \
        (f"What is the current {s} service token? Reply with only the token.", v)


def _f_commit(r):
    s = r.choice(_SERVICES)
    v = _hexid(r, 7)
    return MemoryEntry(text=f"The {s} race-condition bug was fixed in commit {v}.",
                       key=f"commit[{s}]", answer=v), \
        (f"Which commit fixed the {s} race-condition bug? Reply with only the hash.", v)


# Default mix = short, semantic, unguessable *associations* — the regime an agent
# actually needs to remember, and the one D2L recall handles on a 2B model.
_FACTORIES = [_f_account, _f_oncall, _f_codename, _f_port, _f_region, _f_team,
              _f_tier, _f_ticket]
HARD_FACTORIES = [_f_token, _f_commit, _f_path]


def generate_session(n: int, seed: int = 0) -> tuple[list[MemoryEntry], list[Probe]]:
    """Build a session of n distinct observations + one recall probe per entry.

    Subjects are kept unique so probes are unambiguous (one entry per question).
    """
    r = random.Random(seed)
    entries: list[MemoryEntry] = []
    probes: list[Probe] = []
    seen_keys: set[str] = set()
    attempts = 0
    while len(entries) < n and attempts < n * 50:
        attempts += 1
        entry, (q, a) = r.choice(_FACTORIES)(r)
        if entry.key in seen_keys:
            continue  # avoid two entries with the same subject (ambiguous probe)
        seen_keys.add(entry.key)
        probes.append(Probe(question=q, answer=a, entry_idx=len(entries)))
        entries.append(entry)
    return entries, probes
