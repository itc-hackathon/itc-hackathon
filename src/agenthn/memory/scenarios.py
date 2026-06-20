"""Synthetic NIAH-over-trajectory scenarios for the long-horizon memory demo.

Each scenario is an agent conversation (a "haystack" of filler turns) with a few
planted facts ("needles") inserted EARLY, then queried at the end — after the
early turns have scrolled past a fixed context window. This is the setting where
a vanilla context model (truncates at the window) and a markdown-notes baseline
(lossy summary / bloated prompt) struggle, and weight memory shines.

Sizes scale only the haystack length (needles stay in the first ~14 turns):
  small   ~  fits well inside gemma's 8k window  (all methods should recall)
  medium  ~  approaches the window               (baselines pay a growing cost)
  large   ~  exceeds 8k                          (vanilla truncates the needles)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# gemma-2-2b-it context window; the vanilla baseline can hold at most this many.
GEMMA_WINDOW = 8192

# total turns per size (needles occupy the early turns, the rest is filler haystack)
# large is sized to push the raw transcript past the 8k window so vanilla truncates.
SIZES = {"small": 28, "medium": 80, "large": 320}

# NapLoRA nap interval (K=4 keeps each needle in its own small segment -> clean
# recall). The markdown baseline's summarization interval is decoupled (MD_K) so
# long runs don't pay a 2B summarization pass every 4 turns.
NAP_K = {"small": 4, "medium": 4, "large": 4}
MD_K = {"small": 4, "medium": 6, "large": 12}


@dataclass
class Scenario:
    name: str
    size: str
    turns: list[tuple[str, str]]          # (role, text), streamed in order
    probes: list[tuple[str, str]]         # (question, needle substring)
    needle_positions: list[int] = field(default_factory=list)


# Generic project chatter — each entry is a COMPLETE, self-contained turn (no
# mid-sentence truncation) at ~18-30 words. None collide with needle keywords, so
# retrieval stays clean and segments don't drown the planted fact.
_FILLER = [
    "Quick standup recap: the sprint board looks healthy, most cards are in review, and nothing is blocked on my end right now.",
    "I refactored the logging module to use structured fields instead of string concatenation, so the dashboards should be much easier to query now.",
    "The flaky integration test passed on retry again; I think it's a timing issue with the test container starting up slowly.",
    "Bumped the dev dependencies and regenerated the lockfile, and CI went green on the first try, which is always a pleasant surprise.",
    "We still owe the changelog an entry before the next tag, so I jotted down the highlights to make that quick.",
    "The staging dashboard was a little sluggish for a few minutes but recovered on its own; CPU and memory both looked normal.",
    "I closed a handful of stale issues that were already fixed a while ago, and linked the duplicates to the canonical thread.",
    "Let's try to avoid scope creep on this ticket and keep it focused on the one behavior we actually need to change.",
    "Taking a short coffee break, back in ten; ping me if the build breaks or the reviewer has questions on the migration.",
    "I rebased onto main and there was only a trivial conflict in the README, which I resolved by keeping both paragraphs.",
    "The weekly metrics look stable with no regressions worth calling out; latency is flat and throughput ticked up slightly after the caching change.",
    "I added a few unit tests around the parser edge cases we kept tripping over, especially the empty-input and trailing-delimiter paths.",
    "The docs preview built fine and the screenshots render correctly on both light and dark themes, so it's just polish before we publish.",
    "I tidied up the import ordering across the package and ran the formatter so the diff in future pull requests stays small.",
    "The nightly job finished early today with no errors in the logs, and I archived the old experiment branches we no longer need.",
    "Let's sync briefly after lunch on the rollout plan and the rollback path, so the cutover stays clean if anything regresses.",
]


_NEEDLE_SETS: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]] = {
    "apollo_migration": (
        [
            ("user", "Decision for the record: the deployment region is set to eu-west-2."),
            ("user", "For the record: the on-call engineer for launch week is Priya."),
            ("user", "Note this down: the release codename is Blue Falcon."),
            ("user", "Important: the launch date was moved to October 19."),
            ("user", "Record this: the feature flag for the new checkout is called smooth_sailing."),
            ("user", "One more: the QA sign-off ticket is QA-8842."),
        ],
        [
            ("Which region was the deployment set to?", "eu-west-2"),
            ("Who is the on-call engineer for launch week?", "priya"),
            ("What is the release codename?", "blue falcon"),
            ("What date was the launch moved to?", "october 19"),
            ("What is the name of the feature flag for the new checkout?", "smooth_sailing"),
            ("What is the QA sign-off ticket number?", "qa-8842"),
        ],
    ),
    "trip_planning": (
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
    ),
    "research_assistant": (
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
    ),
}

SCENARIO_NAMES = list(_NEEDLE_SETS)


def make_scenario(name: str, size: str = "medium", *, depth: int = 2, spacing: int | None = None) -> Scenario:
    """Build a scenario: needles planted early, then filler to size.

    spacing defaults to the nap interval K, so each needle lands in its OWN nap
    segment (one fact per adapter -> clean retrieval + recall).
    """
    needle_turns, probes = _NEEDLE_SETS[name]
    total = SIZES[size]
    if spacing is None:
        spacing = NAP_K.get(size, 4)
    turns: list[tuple[str, str]] = []
    positions: list[int] = []
    needles = list(needle_turns)
    fi = 0
    next_at = depth
    while len(turns) < total:
        if needles and len(turns) >= next_at:
            positions.append(len(turns))
            turns.append(needles.pop(0))
            next_at = len(turns) + spacing
        else:
            role = "assistant" if len(turns) % 2 else "user"
            turns.append((role, _FILLER[fi % len(_FILLER)]))  # complete turn, not truncated
            fi += 1
    for nt in needles:  # ensure all needles are planted even if total is tiny
        turns.append(nt)            # append first, then record the true index
        positions.append(len(turns) - 1)
    return Scenario(name=name, size=size, turns=turns, probes=probes, needle_positions=positions)


def all_scenarios(size: str = "medium") -> list[Scenario]:
    return [make_scenario(n, size) for n in SCENARIO_NAMES]
