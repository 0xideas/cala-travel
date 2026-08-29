# Candidate Proposal System

## Goal

Guide a user through imagined travel activities for a city. The system begins with three pairs of high-signal activities, learns from the user's choices, and then proposes one or two activities at a time until the user finishes or the activity pool is empty.

Activities come from journal-entry files and corresponding image paths.

## Inputs

Each city has a journal file:

```json
{
  "city": "Toulouse",
  "entries": [
    {
      "id": "jacobins",
      "name": "Couvent des Jacobins",
      "category": "culture",
      "description": "We stepped inside from the warm street and looked up at the stone palm..."
    }
  ]
}
```

Image paths are supplied by activity ID:

```json
{
  "jacobins": "/assets/cities/toulouse/jacobins.webp"
}
```

The supported categories are:

```python
from enum import StrEnum


class Category(StrEnum):
    FOOD = "food"
    DRINKS = "drinks"
    PARTY = "party"
    NATURE = "nature"
    CULTURE = "culture"
    LOCAL_LIFE = "local_life"
```

## Public contracts

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    category: Category
    description: str
    image_path: str


@dataclass(frozen=True)
class Proposal:
    id: str
    activities: tuple[Activity, ...]  # One or two activities.


@dataclass(frozen=True)
class InitialPairs:
    session_id: str
    pairs: tuple[Proposal, Proposal, Proposal]
```

The service interface is:

```python
def get_initial_pairs(
    city: str,
    season: str,
    food_preferences: list[str],
) -> InitialPairs:
    ...


def complete_initial_selection(
    session_id: str,
    selections: dict[str, list[str]],
) -> Proposal | None:
    ...


def submit_feedback(
    session_id: str,
    proposal_id: str,
    selected_activity_ids: list[str],
) -> Proposal | None:
    ...


def finish_session(session_id: str) -> None:
    ...
```

`selections` maps each initial proposal ID to its selected activity IDs. For any pair `(A, B)`, the valid choices are `[]`, `[A]`, `[B]`, and `[A, B]`.

The server derives rejected activities from the activities offered in each proposal.

## Session state

```python
@dataclass
class BanditArm:
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class Session:
    id: str
    city: str
    season: str
    food_preferences: tuple[str, ...]
    activities: dict[str, Activity]
    shown_ids: set[str]
    ranked_queues: dict[Category, list[str]]
    bandit: dict[Category, BanditArm]
    current_proposal_id: str | None
    finished: bool = False
```

Store proposals with their activity IDs so feedback can be validated and processed idempotently.

## Step 1: Create the three initial pairs

Load the city's journal entries, join them to image paths, and send the activities to an LLM with the city, season, and food preferences.

The LLM instruction is:

```text
Choose six activities that reveal the user's broad travel preferences.
Arrange them into three contrasting pairs.
Use exactly six unique IDs from the supplied activities.
Represent several different categories and styles of experience.
Take the stated season and food preferences into account.
Return only the requested JSON structure.
```

Structured output:

```json
{
  "pairs": [
    ["jacobins", "canal_du_midi"],
    ["victor_hugo_market", "aeroscopia"],
    ["fronton_tasting", "saint_pierre_evening"]
  ]
}
```

Validate that the response contains three pairs, six unique IDs, and only known activities. Save the three proposals and return them as `InitialPairs`.

A deterministic fallback selects activities by rotating through populated categories and pairs them in that order.

## Step 2: Learn from the initial choices

After the user has responded to all three pairs, initialize one Thompson Sampling arm for every category with remaining activities:

```python
arms = {
    category: BanditArm(alpha=1.0, beta=1.0)
    for category in categories_with_remaining_activities
}
```

Apply initial feedback:

```python
for activity in selected_initial_activities:
    arms[activity.category].alpha += 1.0

for activity in rejected_initial_activities:
    arms[activity.category].beta += 0.25
```

The smaller rejection weight reflects that the user may simply have preferred the other activity in the pair.

## Step 3: Rank the remaining activities

Make one LLM call containing:

- City, season, and food preferences.
- Selected initial activities.
- Rejected initial activities.
- Every remaining activity grouped by category.

The LLM instruction is:

```text
Infer the user's preferences from the selected and rejected activities.
Rank the remaining activities within each existing category.
Include every supplied activity ID exactly once.
Keep every activity in its supplied category.
Return only the requested JSON structure.
```

Structured output:

```json
{
  "rankings": {
    "food": ["cassoulet", "victor_hugo_market", "violet_sweets"],
    "nature": ["garonne_walk", "jardin_japonais", "canal_walk"],
    "culture": ["saint_sernin", "augustins", "aeroscopia"]
  }
}
```

Validate IDs, categories, and duplicates. Append any omitted activity to the end of its category in journal-file order. These lists become the session's ranked queues.

If the LLM call fails, use journal-file order within each category.

## Step 4: Select the next proposal

Use Thompson Sampling to select one or two categories, then take the first remaining activity in each selected category.

```python
import random

PAIR_PROBABILITY = 0.35


def get_next_item(session: Session, rng: random.Random) -> Proposal | None:
    available = [
        category
        for category, queue in session.ranked_queues.items()
        if queue
    ]
    if not available:
        return None

    count = 2 if len(available) >= 2 and rng.random() < PAIR_PROBABILITY else 1

    samples = {
        category: rng.betavariate(
            session.bandit[category].alpha,
            session.bandit[category].beta,
        )
        for category in available
    }
    selected_categories = sorted(
        available,
        key=samples.get,
        reverse=True,
    )[:count]

    activities = tuple(
        session.activities[session.ranked_queues[category].pop(0)]
        for category in selected_categories
    )
    proposal = save_proposal(session.id, activities)
    session.shown_ids.update(activity.id for activity in activities)
    session.current_proposal_id = proposal.id
    save_session(session)
    return proposal
```

When two categories are selected, the proposal contains one activity from each category.

## Step 5: Update from ongoing feedback

Process the user's selected IDs and derive the rejected IDs from the saved proposal:

```python
def update_learning_algorithm(
    session: Session,
    selected: list[Activity],
    rejected: list[Activity],
) -> None:
    for activity in selected:
        session.bandit[activity.category].alpha += 1.0

    for activity in rejected:
        session.bandit[activity.category].beta += 0.5
```

`submit_feedback` performs three actions in one transaction:

1. Save feedback for the current proposal.
2. Update the bandit once.
3. Call `get_next_item` and return its result.

Repeated feedback for the same proposal returns the previously created successor.

## Termination

The session ends when:

- `finish_session` is called by the user.
- Every ranked queue is empty and `get_next_item` returns `None`.

## Suggested files

```text
app/
  models.py
  journal_repository.py
  llm.py
  bandit.py
  proposal_service.py
  repository.py
data/
  cities/<city>/journal_entries.json
  cities/<city>/images.json
```
