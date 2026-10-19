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

from pydantic import BaseModel, Field, computed_field


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


# --------------------------------------------------------------------------- #
# The Explain stage -- key moments, attributed to cited observable evidence
# --------------------------------------------------------------------------- #


class InsightKind(StrEnum):
    """The key-moment kinds the Explain stage recognises.

    Every member is *observable*. A goal and a chance are visible on the pitch;
    a deep giveaway is a failed pass in the passer's own third. Nothing here
    names a latent quantity: "big chance" and "error" live in `LatentState`, and
    they are exactly what these observable moments are later *scored* against.
    """

    GOAL = "goal"
    CHANCE = "chance"
    DEEP_GIVEAWAY = "deep_giveaway"


class ReasonKind(StrEnum):
    """What a reason is *claiming*.

    A renderer styles these differently, and -- more importantly -- the scoring
    harness selects them structurally. Tagging the claim means the audit reads
    `reason.kind == CREATOR` rather than `reason.text.startswith("Built from")`,
    which would break the day someone rewords a sentence. Same discipline as the
    rest of the project: address data by structure, never by its English.

    There is deliberately **no CAUSE** member. The tempting claim is "this chance
    came from that give-away", and the synthetic truth lets us grade it: only
    ~4% of deep give-aways are genuine errors, and the tape offers nothing to
    separate them from the rest, so the claim has a precision ceiling near 0.03.
    An analyst who said it aloud would be wrong nineteen times in twenty. The
    give-away is surfaced as its own observable moment instead -- never as the
    cause of a later one. See `score_explanations` for the measurement.
    """

    ORIGIN = "origin"        # where the shot came from: box, distance, xG
    CREATOR = "creator"      # the completed pass that built it
    SCORELINE = "scoreline"  # what a goal did to the score
    LOCATION = "location"    # how deep a give-away was
    PRESSURE = "pressure"    # pressure at the time


class InsightReason(BaseModel):
    """One attributable reason a moment matters, bound to the events that show it.

    `event_ids` is not decoration -- it is the audit trail. A reason is a claim
    about what happened, and a claim with no evidence is exactly the
    unfalsifiable commentary this project exists to avoid. If a sentence cannot
    point at events a viewer can already see, it does not belong here.

    Two citation conventions are relied on (and pinned by `test_explainer.py`):
    a CREATOR reason cites `[anchor, the passing event]`, and a LOCATION reason
    cites `[anchor]`. That is how the scoring harness reaches back from a claim
    to the single pass it is checking.
    """

    kind: ReasonKind
    text: str = Field(description="One short, human sentence.")
    event_ids: list[str] = Field(
        min_length=1,
        description="The observable events this sentence is true because of.",
    )


class Insight(BaseModel):
    """A key moment, plus the cited reasons why it matters.

    Output of the Explain stage, input to the Narrator and the overlay. It
    carries no latent truth and no free-floating prose: every reason points at
    real events, so any claim can be traced back to what actually happened.
    """

    insight_id: str
    match_id: str
    clock_s: int
    period: int
    minute: int = Field(ge=0, le=90)
    kind: InsightKind
    team: str = Field(description="The protagonist: the shooter or the passer.")
    anchor_event_id: str = Field(
        description="The event this moment *is* -- the shot, or the give-away. "
        "Its id doubles as the join key so a renderer can line the insight up "
        "with the event it explains."
    )
    x: float = Field(ge=0, le=100, description="Where the moment happened.")
    y: float = Field(ge=0, le=100, description="Where the moment happened.")
    player_name: str = Field(
        description="Human-readable name of the protagonist, for the overlay."
    )
    headline: str = Field(description="A short label, e.g. 'Chance -- J. Smith'.")
    reasons: list[InsightReason] = Field(min_length=1)

    @computed_field
    @property
    def evidence_event_ids(self) -> list[str]:
        """The flat, de-duplicated set of every event any reason cites.

        Derived rather than stored, so the flat list can never drift out of step
        with the per-reason citations: there is one source of truth, and this is
        a view of it.
        """
        seen: list[str] = []
        for reason in self.reasons:
            for event_id in reason.event_ids:
                if event_id not in seen:
                    seen.append(event_id)
        return seen


# --------------------------------------------------------------------------- #
# The Render stage -- the timed overlay feed the graphics layer consumes
# --------------------------------------------------------------------------- #


class OverlayKind(StrEnum):
    """What a timed overlay payload carries."""

    KEY_MOMENT = "key_moment"        # one cited `Insight`, held for its `ttl_s`
    MATCH_SUMMARY = "match_summary"  # the full-time card, held until dismissed


class OverlayPayload(BaseModel):
    """One timed graphic a broadcast overlay engine would draw.

    The overlay feed *is* the product. This webapp is only its first renderer:
    the architecture puts Touchline upstream of the graphics layer, so a real
    broadcast system consumes exactly these payloads and nothing else changes.
    Keeping the payload self-describing -- when to show it, how long to hold it,
    where to anchor it, and the events it cites -- is what makes that swap
    possible.

    A `MATCH_SUMMARY` payload has `ttl_s == 0`, meaning "hold until dismissed";
    every other payload holds for `ttl_s` match-seconds.
    """

    payload_id: str
    match_id: str
    kind: OverlayKind
    clock_s: int = Field(ge=0, description="When the graphic appears.")
    ttl_s: int = Field(ge=0, description="How long it holds (0 = until dismissed).")
    insight: Insight | None = Field(
        default=None, description="Set when kind is KEY_MOMENT."
    )
    summary: MatchSummary | None = Field(
        default=None, description="Set when kind is MATCH_SUMMARY."
    )


# --------------------------------------------------------------------------- #
# The Narrator stage -- grounded studio-desk commentary
# --------------------------------------------------------------------------- #
# This is the one stage allowed to call a language model, and the schemas here
# are the *fence* around it. A model asked to "write about the match" will
# happily invent a stat, a minute or a name, because fluency and accuracy are
# independent skills. So the model is never asked to be careful: it is asked to
# return this shape, and every claim it makes must carry the ids of the
# observable events the sentence is true because of. The prose is the model's;
# the evidence is not, and a claim whose evidence does not check out is thrown
# away rather than trusted. See `narrator.py` for the three gates.
#
# `provider` and `model` travel with the narrative for the same reason the
# calibration harness prints its reference ranges: a reader should always be
# able to see which claims a model made and which came from the deterministic
# fallback, instead of a confident paragraph of unknown provenance.


class Reveal(StrEnum):
    """Which moment of the match a narrative belongs to.

    Both are *scheduled*: half-time and full-time are fixed points on the clock,
    so a narrative can never be requested mid-flow and can never chain into
    another request. Two reveals per match is the whole budget.
    """

    HALF_TIME = "ht"
    FULL_TIME = "ft"


class PersonaId(StrEnum):
    """The three chairs at the virtual studio desk.

    They exist to give one set of facts three voices, not three sets of facts.
    A partisan reading and a numbers reading of the same event are both useful
    to a viewer precisely because they disagree about *emphasis* while agreeing
    about what happened. If a persona could introduce a fact the others could
    not, it would be a liability rather than a feature.
    """

    ANALYST = "analyst"   # measured, tactical, explains the "why"
    DIEHARD = "diehard"   # partisan foil: emotional, but still cited
    NUMBERS = "numbers"   # xG purist: only what it can point a number at


class NarrativeClaim(BaseModel):
    """One sentence of commentary and the events that make it true.

    `event_ids` may be empty, and that is a deliberate, honest signal: it marks
    a claim grounded in the summary as a whole ("they finished with more xG")
    rather than in a single moment. Such a claim still passes the number audit
    (every figure in it must appear in the facts it was given), it simply has no
    single playhead position to seek to. A renderer shows "Show me" only when
    this list is non-empty -- the user is never offered a jump that would land
    on nothing.
    """

    text: str = Field(min_length=1, description="One short, human sentence.")
    event_ids: list[str] = Field(
        default_factory=list,
        description="Observable events this sentence is true because of. Empty "
        "means 'grounded in the summary as a whole', not 'ungrounded'.",
    )


class PersonaTake(BaseModel):
    """One pundit's reading of a reveal: a headline and a few cited claims."""

    persona: PersonaId
    headline: str = Field(min_length=1)
    claims: list[NarrativeClaim] = Field(min_length=1)


class RevealNarrative(BaseModel):
    """The studio desk's response at one scheduled reveal.

    `validated` is False whenever any persona had to be backfilled from the
    deterministic fallback -- usually because the model's JSON did not parse, a
    citation did not resolve, or a number could not be found in the facts. The
    panel renders either way; the flag travels so a viewer (and the project's
    own honesty standard) can tell model prose from template prose at a glance.
    """

    narrative_id: str
    match_id: str
    reveal: Reveal
    clock_s: int = Field(ge=0, description="When the desk is speaking.")
    personas: list[PersonaTake] = Field(min_length=1)
    provider: str = Field(description="azure | github | local")
    model: str
    validated: bool = Field(
        description="True only if every persona came from the model and passed "
        "every gate."
    )
    dropped_claims: int = Field(
        default=0, ge=0, description="Claims discarded by the citation or number gate."
    )
