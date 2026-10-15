"""Geometric pass classifier: recover pass *intent* from geometry alone.

WHY THIS FILE EXISTS
--------------------
`PassType` is deliberately absent from `MatchEvent`. A broadcast camera can see
the ball *travel*, but not the player's *intent* -- was that a cross, or a
hopeful long ball? The simulator records the true intent as ground truth
(`LatentState.pass_type_true`), and this module is the miniature of the project's
whole thesis: **infer the hidden label from the observables, then score the
inference against the answer.**

THE ONE RULE
------------
This module reads observable `MatchEvent`s and nothing else. It does not import
the simulator, so it has no code path to `truth_series()`. `pass_type_true` is
reachable only from the scoring harness in `__main__`, which is a deliberate,
visible act -- exactly the asymmetry `simulator.observable_stream()` was built to
enforce.

WHY GEOMETRY CAN RECOVER INTENT AT ALL
--------------------------------------
The simulator only ever *labels* a pass as a cross when the ball is actually
delivered from a wide final-third position toward the centre, and only calls a
ball "long" when it really travels far. The label is a fact about the
trajectory, not a private wish -- which is what makes it learnable from the
coordinates. Where the geometry is genuinely ambiguous (a big forward pass from
midfield is hard to separate from a long ball), the confusion shows up honestly
in the report rather than being tuned away.

THE COORDINATE FRAME
--------------------
The event stream is in the fixed broadcast frame (home always attacks x=100).
The simulator's decisions were made in an *attacking-local* frame, so the first
thing we do is rotate back into it: a team's own goal is always x=0.

Run the scored report:
    ./.venv/bin/python pass_classifier.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from schemas import EventType, MatchEvent, PassType

# --------------------------------------------------------------------------- #
# Geometry constants. Defined HERE, not imported from the simulator, for the
# same reason `metrics.py` defines its own FINAL_THIRD_X: a module that claims to
# know nothing about the simulator must not borrow its constants. These are
# football-geometric choices, stated so they can be argued with.
# --------------------------------------------------------------------------- #
PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0

FINAL_THIRD_X = 68.0  # attacking-local x where the final third begins
CROSS_START_X = 72.0  # forward of this, from a wide lane, a ball is a delivery
WIDE_Y = 24.0  # inside this lane (or its mirror) counts as "wide"
WIDE_MAX_Y = 100.0 - WIDE_Y
LONG_MIN_ADVANCE = 22.0  # a ball carried this far FORWARD is a long ball
THROUGH_MIN_ADVANCE = 8.0  # a ball played this far forward past x=68 is a through ball
CROSS_MIN_PULL = 8.0  # a delivery must gain this much ground toward the middle
DEEP_FINAL_X = 90.0  # past here the endpoint clamps and direction is lost
CENTER_Y = 50.0


def _to_local(team_id: str, home_id: str, x: float, y: float) -> tuple[float, float]:
    """Fixed broadcast frame -> attacking-local frame (100 = goal being attacked).

    The mirror is the whole game: once every pass is read with the attacker's own
    goal at x=0, a left-winger crossing for the home side and one crossing for the
    away side look identical, which is the only way one rule can classify both.
    """
    if team_id == home_id:
        return x, y
    return 100.0 - x, 100.0 - y


def pass_distance_m(lx: float, ly: float, tx: float, ty: float) -> float:
    """Straight-line distance in metres, in the attacking-local frame."""
    return math.hypot(
        (tx - lx) / 100.0 * PITCH_LEN_M,
        (ty - ly) / 100.0 * PITCH_WID_M,
    )


def classify_pass(event: MatchEvent, home_id: str) -> PassType:
    """Recover the kind of a pass from its observable geometry.

    Decision order matters and mirrors how a viewer reads a pass:
      1. delivered forward, from wide in the final third, toward the middle -> CROSS
      2. carried a long way FORWARD                                      -> LONG
      3. played forward into the final third                             -> THROUGH_BALL
      4. deep and central, where the geometry has run out                -> THROUGH_BALL
      5. otherwise                                                       -> SHORT

    Every threshold below was *measured* on a disjoint block of seeds (see
    TUNE_SEEDS) before this classifier was ever scored on the held-out block
    (EVAL_SEEDS). Four of the choices are worth stating because they were made
    from the data, not from taste:

    * LONG is tested by forward `advance`, not euclidean distance. A long ball
      is a ball hit down the pitch; lateral drift inflates a euclidean distance
      and made ordinary 20 m diagonals look "long". Advance has no such failure.
    * A CROSS must gain real ground toward the middle (`CROSS_MIN_PULL`), not
      merely be aimed vaguely infield. True deliveries cluster at a large
      inward pull; safe circulation sits near zero, so this is a clean cut.
    * A THROUGH_BALL must be played a genuine distance forward
      (`THROUGH_MIN_ADVANCE`) to count as penetrating.
    * Past `DEEP_FINAL_X` the two endpoints are so close that the simulator's
      own endpoint clamp erases the direction of travel entirely. Geometry
      cannot decide there, so we fall back on the one thing that survives:
      position. A central ball in that band is more often a through ball than a
      safe short one, so that is the guess.

    `home_id` is the only non-event input: the fixed frame pins home to the
    attacking direction, so the frame conversion needs it.
    """
    if event.type != EventType.PASS or event.end_x is None or event.end_y is None:
        raise ValueError("classify_pass expects a pass with an endpoint")

    lx, ly = _to_local(event.team, home_id, event.x, event.y)
    tx, ty = _to_local(event.team, home_id, event.end_x, event.end_y)

    advance = tx - lx
    pull = abs(ly - CENTER_Y) - abs(ty - CENTER_Y)  # >0 == moved toward the middle
    start_final = lx > CROSS_START_X
    start_wide = ly < WIDE_Y or ly > WIDE_MAX_Y

    if start_final and start_wide and pull >= CROSS_MIN_PULL and advance > 0.0:
        return PassType.CROSS
    if advance >= LONG_MIN_ADVANCE:
        return PassType.LONG
    if tx > FINAL_THIRD_X and advance > THROUGH_MIN_ADVANCE:
        return PassType.THROUGH_BALL
    if start_final and not start_wide and lx > DEEP_FINAL_X:
        return PassType.THROUGH_BALL
    return PassType.SHORT


# --------------------------------------------------------------------------- #
# Scoring -- pairs the observable stream with the truth, and keeps score.
# --------------------------------------------------------------------------- #


@dataclass
class PassTypeReport:
    """A confusion matrix plus the rates derived from it.

    The matrix is the honest artefact: one overall accuracy hides *which* kinds
    get confused. We report per-class precision/recall/F1 so a weak class cannot
    be buried under a strong one.
    """

    classes: tuple[PassType, ...] = tuple(PassType)
    matrix: dict[tuple[PassType, PassType], int] = field(default_factory=dict)
    total: int = 0

    def update(self, truth: PassType, pred: PassType) -> None:
        self.matrix[(truth, pred)] = self.matrix.get((truth, pred), 0) + 1
        self.total += 1

    def correct(self) -> int:
        return sum(n for (t, p), n in self.matrix.items() if t == p)

    def accuracy(self) -> float:
        return self.correct() / self.total if self.total else 0.0

    def accuracy_ci(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for the accuracy -- the error bar.

        A bare "90.9%" invites "how sure are you?", and the honest answer moves
        with sample size. Wilson is used rather than the textbook normal
        interval because it stays inside [0, 1] and behaves sanely for rare
        classes, which is exactly our CROSS case.
        """
        n = self.total
        if n == 0:
            return 0.0, 0.0
        p = self.accuracy()
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, center - half), min(1.0, center + half)

    def _row(self, truth: PassType) -> int:
        return sum(n for (t, _), n in self.matrix.items() if t == truth)

    def _col(self, pred: PassType) -> int:
        return sum(n for (_, p), n in self.matrix.items() if p == pred)

    def precision(self, cls: PassType) -> float:
        col = self._col(cls)
        return self.matrix.get((cls, cls), 0) / col if col else 0.0

    def recall(self, cls: PassType) -> float:
        row = self._row(cls)
        return self.matrix.get((cls, cls), 0) / row if row else 0.0

    def f1(self, cls: PassType) -> float:
        p, r = self.precision(cls), self.recall(cls)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def render(self) -> str:
        lo, hi = self.accuracy_ci()
        lines = [f"passes scored        {self.total}"]
        lines.append(
            f"overall accuracy     {100 * self.accuracy():.1f}%  "
            f"(95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"
        )

        # Baseline: always guess the most common true class. If we cannot beat
        # this, we have learned nothing that a constant could not do.
        counts = {c: self._row(c) for c in self.classes}
        majority = max(counts, key=counts.get)
        baseline = counts[majority] / self.total if self.total else 0.0
        lines.append(
            f"majority baseline    {100 * baseline:.1f}%  "
            f"(always '{majority.value}')"
        )
        lines.append("")

        width = max(len(c.value) for c in self.classes)
        header = "truth \\ pred".ljust(width + 12) + "".join(
            c.value.rjust(width + 3) for c in self.classes
        )
        lines.append(header)
        for t in self.classes:
            cells = "".join(
                str(self.matrix.get((t, p), 0)).rjust(width + 3) for p in self.classes
            )
            lines.append(t.value.ljust(width + 12) + cells)
        lines.append("")
        lines.append(
            "class".ljust(width + 4)
            + "prec".rjust(7)
            + "recall".rjust(8)
            + "f1".rjust(7)
            + "n".rjust(7)
        )
        for c in self.classes:
            lines.append(
                c.value.ljust(width + 4)
                + f"{self.precision(c):>7.2f}"
                + f"{self.recall(c):>8.2f}"
                + f"{self.f1(c):>7.2f}"
                + f"{self._row(c):>7}"
            )
        return "\n".join(lines)


def score_seeds(seeds: range) -> PassTypeReport:
    """Score the classifier over many matches using the paired truth channel.

    `observable_stream()` and `truth_series()` are both projections of the same
    `snapshots` list, so index i of one lines up with index i of the other. We
    walk them together and only look at the truth where a pass actually happened.
    """
    # Imported here, not at module scope: the classifier itself must stay unable
    # to see the truth. This harness is the one place allowed to.
    from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

    report = PassTypeReport()
    home_id = DEMO_HOME.team_id
    for seed in seeds:
        sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id=f"P{seed:04d}", seed=seed)
        sim.simulate(live=True)
        for event, truth in zip(sim.observable_stream(), sim.truth_series(), strict=True):
            if truth.pass_type_true is None:
                continue
            report.update(truth.pass_type_true, classify_pass(event, home_id))
    return report


# The thresholds above were selected on TUNE_SEEDS; the headline number comes
# from EVAL_SEEDS, which never touched the selection. Two disjoint blocks keep
# the reported accuracy from being the number we optimised against.
TUNE_SEEDS = range(1000, 1040)
EVAL_SEEDS = range(2000, 2040)


if __name__ == "__main__":
    for label, seeds in (("tuning split", TUNE_SEEDS), ("HELD-OUT split", EVAL_SEEDS)):
        print(f"Geometric pass classifier -- {label} "
              f"(seeds {seeds.start}-{seeds.stop - 1})")
        print("=" * 62)
        print(score_seeds(seeds).render())
        print()
