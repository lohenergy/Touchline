"""Domain schemas for the Touchline match-intelligence pipeline.

WHY THIS FILE EXISTS ON ITS OWN
-------------------------------
Every object in here eventually crosses two boundaries:

    1. an HTTP boundary   (FastAPI -> browser)
    2. a storage boundary (Cosmos DB)

If the wire format and the stored format ever drift apart you get bugs that
only show up in production, when it is expensive to find them. Defining every
shape exactly once, in one place, means the simulator, the API and the database
all agree *by construction* rather than by discipline.

WHY PYDANTIC AND NOT PLAIN DATACLASSES
--------------------------------------
Pydantic validates values as the object is constructed. When data arrives from
a network socket, a database, or an LLM, the question "is this field really a
number between 0 and 100?" is a *security* question, not a formatting question.
Malformed or hostile input is the normal case, not the exception. Validation at
the trust boundary is the cheapest defence you will ever write, and Pydantic
gives it to us for free. It is also FastAPI's native type system, so the same
class that guards the simulator will later guard the HTTP API.

THE COORDINATE CONVENTION
-------------------------
    x = 0    home team's own goal-line
    x = 100  away team's goal-line
    y = 0    one touchline
    y = 100  the other touchline

Home always attacks toward x=100. Away always attacks toward x=0. This is a
*fixed* frame, which matters because a broadcast overlay must place every dot in
the same physical space on screen. Internally the simulator thinks in a
"attacking-local" frame (where 100 is always the goal you are attacking) and
converts at the boundary -- see simulator.py.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
# We use enums rather than bare strings for two reasons:
#   1. Typos become import-time or validation-time errors instead of silent
#      data corruption ("shot" vs "shots" would quietly split your dataset).
#   2. An enum is self-documenting: the set of legal values is visible in the
#      code rather than buried in a spec somewhere.


class EventType(StrEnum):
    """The vocabulary of things that can happen in a match.

    Every member here is emitted by the simulator. A vocabulary entry that is
    never produced is worse than useless -- it is a silent contract violation
    that a downstream consumer will happily write code against and then fail on
    at runtime. (An early draft declared SUBSTITUTION and CARD and never emitted
    them; they are implemented now.)
    """

    KICKOFF = "kickoff"
    PASS = "pass"
    SHOT = "shot"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    SAVE = "save"
    BLOCK = "block"
    OFFSIDE = "offside"
    CORNER = "corner"
    POSSESSION_CHANGE = "possession_change"
    PRESSURE = "pressure"
    LOOSE_BALL = "loose_ball"
    FOUL = "foul"
    CARD = "card"
    SUBSTITUTION = "substitution"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"


class CardType(StrEnum):
    YELLOW = "yellow"
    RED = "red"


class Outcome(StrEnum):
    """How an event resolved.

    A single enum covers all event types deliberately. The alternative -- one
    enum per event type -- sounds tidier but makes generic processing (storing,
    streaming, charting) painful, because every consumer would need to know
    which enum to import for which event.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INTERCEPTED = "intercepted"
    TACKLED = "tackled"
    BLOCKED = "blocked"
    OFF_TARGET = "off_target"
    WOODWORK = "woodwork"
    ON_TARGET = "on_target"
    SAVED = "saved"
    GOAL = "goal"
    WON = "won"
    LOST = "lost"


class Phase(StrEnum):
    """Character of the match at a given instant (ground truth only)."""

    CONTROLLED = "controlled"
    CONTESTED = "contested"
    CHAOTIC = "chaotic"


class PassType(StrEnum):
    """What kind of pass a pass was.

    DELIBERATELY NOT A FIELD ON MatchEvent. A broadcaster can *see* the ball
    travel, but the *intent* -- was that a cross or a hopeful long ball? -- is
    inferred from the trajectory. So the simulator records the true intent as
    ground truth and Phase 1 builds a geometric classifier that recovers it from
    the coordinates alone, then scores itself against this label. That is a
    miniature of the project's whole thesis, applied to a single event.
    """

    SHORT = "short"
    LONG = "long"
    THROUGH_BALL = "through_ball"
    CROSS = "cross"


class StoppageCause(StrEnum):
    """Why a half's clock ran past regulation time.

    Every cause here corresponds to real, observable events in the stream, so
    the added-time figure is *derivable and explainable* rather than announced
    by fiat. "Seven minutes were added: two goals, three substitutions and a
    card" is a claim the overlay can actually justify.
    """

    GOAL = "goal"
    SUBSTITUTION = "substitution"
    CARD = "card"
    ROUTINE = "routine"  # the baseline allowance for ordinary play stoppages


class StoppageBreakdown(BaseModel):
    """The ledger behind a half's added time.

    `total_s` is the added time in seconds; `by_cause` attributes it to the
    events that caused it. The two are kept together so a consumer cannot show
    the total without also having the explanation available.
    """

    total_s: int = Field(ge=0)
    by_cause: dict[StoppageCause, int] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The observable event -- everything an analyst could actually see
# --------------------------------------------------------------------------- #


class MatchEvent(BaseModel):
    """A single observable match event.

    CRITICAL DESIGN RULE: this object must contain *only* things that a real
    tracking/event provider could genuinely observe. It must never contain
    latent state (see LatentState below). If it did, every downstream model
    would be "predicting" an answer that was handed to it -- a failure known as
    data leakage, and the single most common way a machine-learning project
    produces impressive numbers that mean nothing.

    Field names carry their units (clock_s, speed_kmh) so a reader never has to
    guess whether "clock" means seconds, minutes, or a string.
    """

    event_id: str
    match_id: str

    # --- time -------------------------------------------------------------- #
    # clock_s is a strict, monotonic wall clock: it keeps counting through added
    # time, so event ordering and rolling windows never have to special-case a
    # half boundary. `minute` and `stoppage_minute` are the *display* clock a
    # broadcaster shows -- 45 and, in added time, 45+X -- which is derived from
    # clock_s and the period, not from clock_s alone.
    clock_s: int = Field(ge=0, le=13_200, description="Monotonic match seconds")
    minute: int = Field(ge=0, le=90, description="Display minute (regulation)")
    stoppage_minute: int = Field(
        default=0, ge=0, description="X in '45+X'. Zero during regulation."
    )
    period: int = Field(ge=1, le=2)

    # --- who and what ------------------------------------------------------ #
    type: EventType
    team: str
    player_id: str
    player_name: str = Field(
        description="Human-readable name. The brief requires real-time player "
        "name tagging, and an overlay cannot render an opaque synthetic id."
    )
    player_number: int = Field(
        ge=0,
        le=99,
        description="Shirt number. A broadcast camera can read it off the back, "
        "so it is observable -- and it is what the overlay draws on an actor disc. "
        "Colour alone could not tell two players of the same team apart.",
    )
    outcome: Outcome
    card_type: CardType | None = Field(
        default=None, description="Set on CARD events only."
    )
    player_out_id: str | None = Field(
        default=None, description="Set on SUBSTITUTION events: the player replaced."
    )
    stoppage: StoppageBreakdown | None = Field(
        default=None,
        description="Set on HALF_TIME/FULL_TIME only: why the half ran long.",
    )

    # --- where (fixed frame, see module docstring) ------------------------- #
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    end_x: float | None = Field(default=None, ge=0, le=100)
    end_y: float | None = Field(default=None, ge=0, le=100)
    in_box: bool | None = Field(
        default=None,
        description="SHOT only: was the shot taken from inside the penalty area? "
        "A broadcaster can see that, so it is observable.",
    )

    # --- derived football features ----------------------------------------- #
    # These exist because the hackathon brief explicitly asks for them:
    # "pass distance, pass accuracy and a pass difficulty rating",
    # "ball speed and shot speed".
    pass_distance: float | None = Field(default=None, ge=0, description="Metres")
    pass_difficulty: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="0 = trivial, 100 = near-impossible. Derived from the "
        "completion model, so it is principled rather than arbitrary.",
    )
    pressure_at_location: float | None = Field(default=None, ge=0, le=1)
    shot_quality: float | None = Field(
        default=None, ge=0, le=1, description="xG-like expected goal value"
    )
    ball_speed_kmh: float | None = Field(default=None, ge=0)
    shot_speed_kmh: float | None = Field(default=None, ge=0)


# --------------------------------------------------------------------------- #
# Ground truth -- the thing that makes this project different
# --------------------------------------------------------------------------- #


class LatentState(BaseModel):
    """The hidden truth about who is really controlling the match.

    This is the payoff of building on a simulator instead of real data. Real
    match data has no label for "which team was actually on top, and by how
    much?" -- analysts can only argue about it. Our simulator *knows*, because
    it generated the match from these very variables.

    That gives us something no real-data project can have: a numeric ground
    truth against which we can *score* our explanatory model. "Our explanations
    are 81% correct" is a claim we can prove; the same claim from a real-data
    project would be unfalsifiable marketing.

    This object is used for exactly two purposes:
        1. training the explainability model
        2. scoring it
    It must never be mixed into the agent-facing event stream.
    """

    match_id: str
    clock_s: int

    control_true: float = Field(
        ge=-1, le=1, description="Signed game control. +1 home dominant, -1 away."
    )
    chaos_true: float = Field(
        ge=0, le=1, description="How unstructured the game is. 0 = settled, 1 = end-to-end."
    )
    dominant_team: str
    phase: Phase
    pass_type_true: PassType | None = Field(
        default=None,
        description="On PASS snapshots only: the true kind of the pass. Ground "
        "truth for the Phase 1 geometric pass classifier. Never on the event.",
    )
    big_chance_true: bool | None = Field(
        default=None,
        description="On SHOT snapshots only: the true chance tier. A 'big chance' "
        "is an analyst-coded, subjective stat in real football; here it is decided "
        "on latent quality (noisy relative to the observable xG), so recovering it "
        "from events is a real learning task with a labelled answer.",
    )
    chance_created_true: bool | None = Field(
        default=None,
        description="On the single PASS snapshot that manufactured a big chance. "
        "Ground truth for the 'chances created' stat and the creator attribution.",
    )
    error_true: bool | None = Field(
        default=None,
        description="On PASS snapshots only: an unforced give-away deep in the "
        "possession side's own third that hands the opposition a chance. Another "
        "analyst-coded stat (errors) that we can label because we generated the game.",
    )
    from_error_true: bool | None = Field(
        default=None,
        description="On SHOT snapshots only: this shot was the direct product of "
        "an error. Closes the causal chain error -> giveaway -> chance, so the "
        "error-to-chance attribution can be scored.",
    )


# --------------------------------------------------------------------------- #
# The interpreted match state -- the output of the "Interpret" stage
# --------------------------------------------------------------------------- #


class MatchState(BaseModel):
    """A rolling snapshot of meaning derived from the raw event stream.

    This is what the agents reason over. Notice it contains no raw events and
    no latent truth -- only *interpretation*: quantities computed from
    observable events, which is exactly what a real broadcast analyst would
    have to work from.

    Implemented in Phase 1 (metrics.py). Defined here now so the contract is
    fixed before anything depends on it.
    """

    match_id: str
    clock_s: int
    period: int
    score: dict[str, int]

    possession: dict[str, float] = Field(
        description="Share of CONTROLLED time per team, re-normalised to sum "
        "to 1. Contested time is reported separately below, not smuggled in."
    )
    contested_possession: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Fraction of elapsed time when neither side controlled the "
        "ball. This is the time-based answer to 'who had the ball?' -- pass "
        "count alone cannot express it.",
    )
    control: dict[str, float] = Field(
        description="Per-team share of control, summing to 1, read from a "
        "rolling observable window. Phase 1 ships a deliberately transparent "
        "baseline (current rolling possession share); Phase 2 learns a better "
        "estimator and scores it against control_true. A baseline is not an "
        "answer -- it is the thing you measure improvement against."
    )
    control_trend: float = Field(
        description="Signed rate of change of the control margin (home share "
        "minus away share) over the rolling window, per minute. Large "
        "magnitudes are what make a moment 'matter'."
    )
    chaos: float = Field(
        ge=0,
        le=1,
        description="Observable baseline for how unstructured the game is: the "
        "share of recent events that are turnovers, loose balls, offsides or "
        "incomplete passes. Phase 2 scores it against chaos_true.",
    )
    tempo: float = Field(ge=0, description="Events per minute over the rolling window")

    territory: dict[str, float] = Field(
        description="Field tilt: each team's share of final-third passes, "
        "renormalised to sum to 1. A spatial read that possession alone hides."
    )
    pass_completion: dict[str, float] = Field(
        description="Completed / attempted passes, per team, in 0..1."
    )
    pass_difficulty_mean: dict[str, float] = Field(
        description="Mean pass_difficulty of each team's passes, in 0..100."
    )
    shot_quality_total: dict[str, float] = Field(
        description="Cumulative xG per team (sum of shot_quality)."
    )
    shots: dict[str, int] = Field(description="Shot count per team.")


# --------------------------------------------------------------------------- #
# The full-time summary -- what a highlights package shows on the final whistle
# --------------------------------------------------------------------------- #


class MatchSummary(BaseModel):
    """The full-time, observable-only aggregate behind the summary screen.

    This is the same shape a broadcast app shows on the whistle -- possession,
    shots, xG, discipline -- computed entirely from the event stream.

    It deliberately contains NO latent truth. "Big chances" and "errors" are
    analyst-coded AND live in `LatentState`, so this summary cannot report them
    as fact; only an inferred, scored estimator (Phase 2) may put them on
    screen. A summary that quietly printed the ground truth would look
    authoritative and be exactly the unfalsifiable claim this project exists to
    avoid.
    """

    match_id: str
    home: str
    away: str
    home_name: str
    away_name: str

    score: dict[str, int]

    possession: dict[str, float] = Field(description="Controlled-time share, 0..1.")
    contested_pct: float = Field(ge=0, le=100, description="Loose-ball time, percent.")

    shots: dict[str, int]
    shots_on_target: dict[str, int]
    shots_in_box: dict[str, int]
    woodwork: dict[str, int]
    saves: dict[str, int]
    blocks: dict[str, int]
    xg: dict[str, float]

    passes: dict[str, int]
    pass_completion: dict[str, float]
    pass_difficulty_mean: dict[str, float]

    corners: dict[str, int]
    offsides: dict[str, int]
    fouls: dict[str, int]
    yellows: dict[str, int]
    reds: dict[str, int]
    substitutions: dict[str, int]

    territory: dict[str, float]
    added_time_s: dict[int, int] = Field(default_factory=dict)
    stoppage: dict[int, StoppageBreakdown] = Field(default_factory=dict)
