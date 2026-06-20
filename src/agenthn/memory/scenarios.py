"""Synthetic NIAH-over-trajectory scenarios for the long-horizon memory demo.

Each scenario is a long agent conversation (a "haystack" of filler turns) with a
handful of planted facts ("needles") inserted EARLY, then queried at the end —
after the early turns have scrolled out of any fixed context window. This is the
setting where a vanilla context model (truncates) and a markdown-notes baseline
(lossy summary / bloated prompt) struggle, and weight memory shines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    turns: list[tuple[str, str]]          # (role, text), streamed in order
    probes: list[tuple[str, str]]         # (question, needle substring)
    needle_positions: list[int] = field(default_factory=list)  # turn idx of each needle


# Generic project chatter used to pad the trajectory into a long haystack. None of
# these collide with needle keywords, so retrieval stays clean.
_FILLER = [
    "Let's keep momentum on the sprint; standup looked good this morning.",
    "I refactored the logging module, nothing user-facing changed.",
    "Can you re-run the linter? I think there was a trailing-whitespace warning.",
    "The flaky integration test passed on retry, I'll keep an eye on it.",
    "Bumped the dev dependencies, CI is green again.",
    "Reminder to update the changelog before the next tag.",
    "The staging dashboard was a little slow but recovered on its own.",
    "I closed three stale issues that were already fixed.",
    "Let's avoid scope creep on this ticket and keep it focused.",
    "Coffee break, back in ten. Ping me if the build breaks.",
    "I rebased onto main, only a trivial conflict in the README.",
    "The metrics look stable week over week, no regressions.",
    "Added a couple of unit tests around the parser edge cases.",
    "Docs preview built fine, the screenshots render correctly.",
    "Nothing blocking from my side, proceeding as planned.",
    "I tidied up the import ordering across the package.",
    "The nightly job finished early today, no errors in the logs.",
    "Let's sync briefly after lunch on the rollout plan.",
    "I archived the old experiment branches we no longer need.",
    "Small typo fix in the onboarding guide, merged already.",
]


def _build(name: str, needle_turns: list[tuple[str, str]], probes, *, depth=3, spacing=4, total=48):
    """Interleave needle turns into a filler haystack at increasing depth.

    needle_turns: (role, text) facts to plant. They are inserted starting at
    `depth`, every `spacing` turns, and the rest is filler up to `total` turns.
    """
    turns: list[tuple[str, str]] = []
    positions: list[int] = []
    fi = 0
    needles = list(needle_turns)
    next_needle_at = depth
    while len(turns) < total:
        if needles and len(turns) >= next_needle_at:
            positions.append(len(turns))
            turns.append(needles.pop(0))
            next_needle_at = len(turns) + spacing
        else:
            role = "assistant" if len(turns) % 2 else "user"
            turns.append((role, _FILLER[fi % len(_FILLER)]))
            fi += 1
    # ensure any leftover needles are placed before the end
    for nt in needles:
        positions.append(len(turns) - 1)
        turns.insert(len(turns) - 1, nt)
    return Scenario(name=name, turns=turns, probes=probes, needle_positions=positions)


def apollo_migration() -> Scenario:
    return _build(
        "apollo_migration",
        [
            ("user", "Decision for the record: the deployment region is set to eu-west-2."),
            ("user", "For the record: the on-call engineer for launch week is Priya."),
            ("user", "Note this down: the release codename is Blue Falcon."),
            ("user", "Important: the launch date was moved to October 19."),
            ("user", "Record this: the feature flag for the new checkout is called smooth_sailing."),
            ("user", "One more: QA sign-off is owned by Marcus."),
        ],
        [
            ("Which region was the deployment set to?", "eu-west-2"),
            ("Who is the on-call engineer for launch week?", "priya"),
            ("What is the release codename?", "blue falcon"),
            ("What date was the launch moved to?", "october 19"),
            ("What is the name of the feature flag for the new checkout?", "smooth_sailing"),
            ("Who owns QA sign-off?", "marcus"),
        ],
    )


def trip_planning() -> Scenario:
    return _build(
        "trip_planning",
        [
            ("user", "For the trip: my flight confirmation code is QX7P2R."),
            ("user", "Remember this: the hotel is the Marlowe on Pine Street, room 412."),
            ("user", "Note: my rental car is a blue Subaru, plate 8KZL990."),
            ("user", "Important: the dinner reservation is at Casa Lupe at 7:30pm on Friday."),
            ("user", "Save this: the museum tickets are under the name Dr. Okafor."),
            ("user", "Also: the return train departs from platform 9 at 6:05am."),
        ],
        [
            ("What is my flight confirmation code?", "qx7p2r"),
            ("Which hotel and room am I staying in?", "412"),
            ("What is the rental car license plate?", "8kzl990"),
            ("Where and when is the dinner reservation?", "casa lupe"),
            ("Whose name are the museum tickets under?", "okafor"),
            ("Which platform does the return train depart from?", "platform 9"),
        ],
    )


def research_assistant() -> Scenario:
    return _build(
        "research_assistant",
        [
            ("user", "Log: the baseline model checkpoint we use is run-4417."),
            ("user", "Record: the best learning rate from the sweep was 3e-4."),
            ("user", "Note: the held-out evaluation set is called gorilla-hard."),
            ("user", "Important: the dataset license is CC-BY-NC, no commercial use."),
            ("user", "Save: the lead reviewer for the paper is Professor Adeyemi."),
            ("user", "Also: the submission deadline is November 7 at 23:59 AoE."),
        ],
        [
            ("Which checkpoint is the baseline model?", "run-4417"),
            ("What was the best learning rate from the sweep?", "3e-4"),
            ("What is the held-out evaluation set called?", "gorilla-hard"),
            ("What license is the dataset under?", "cc-by-nc"),
            ("Who is the lead reviewer for the paper?", "adeyemi"),
            ("When is the submission deadline?", "november 7"),
        ],
    )


def all_scenarios() -> list[Scenario]:
    return [apollo_migration(), trip_planning(), research_assistant()]
