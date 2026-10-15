"""Render stage: turn insights into a timed overlay feed a graphics layer draws.

WHY THIS FILE EXISTS
--------------------
The Explain stage produces `Insight`s -- key moments with cited reasons. That is
*meaning*, not *graphics*. A broadcast overlay engine does not want a list of
insights; it wants a timeline: show this card at this match-second, hold it for
this long, anchor it here. This module makes that translation and nothing else.

It is deliberately the smallest stage in the pipeline, and deliberately dumb. It
decides *when* and *how long for*, never *what is true*. Every word it emits was
already produced and cited upstream, so an overlay bug can never invent a claim;
the worst it can do is show a real one at the wrong moment, which is a timing
bug and easy to see.

WHY IT IS A FEED, NOT A PAGE
----------------------------
The architecture puts Touchline *upstream* of the graphics layer: the overlay
JSON is the product, and any renderer -- our page, OBS, a broadcast box -- is
just a consumer. So the payload has to be self-describing and renderer-agnostic:
a kind, a start time, a duration, an anchor, and the cited events. Nothing in
here knows what a browser is.

THE TIMING RULES
----------------
Hold times are a broadcast judgement, not a derived quantity: a goal earns
longer on screen than a give-away, because the eye needs longer to take in
"2-1" than "they lost it 30 m out". They are constants here so they are easy to
argue about and easy to change in one place.

Run a spot check (prints one match's overlay timeline):
    ./.venv/bin/python overlay.py
"""

from __future__ import annotations

from schemas import (
    Insight,
    InsightKind,
    MatchSummary,
    OverlayKind,
    OverlayPayload,
)

# How long each graphic holds, in match-seconds. A summary holds until the
# viewer dismisses it, which is what ttl 0 means.
TTL_S: dict[InsightKind, int] = {
    InsightKind.GOAL: 8,
    InsightKind.CHANCE: 6,
    InsightKind.DEEP_GIVEAWAY: 4,
}


def payload_for(insight: Insight) -> OverlayPayload:
    """Wrap one insight as a timed overlay payload."""
    return OverlayPayload(
        payload_id=f"{insight.match_id}-o-{insight.insight_id}",
        match_id=insight.match_id,
        kind=OverlayKind.KEY_MOMENT,
        clock_s=insight.clock_s,
        ttl_s=TTL_S[insight.kind],
        insight=insight,
    )


def build_overlay(
    insights: list[Insight], summary: MatchSummary | None = None
) -> list[OverlayPayload]:
    """The whole match's overlay timeline, in clock order.

    The summary, when given, is appended at the final whistle and holds until
    dismissed. Everything else is one key-moment payload per insight.
    """
    payloads = [payload_for(insight) for insight in insights]
    payloads.sort(key=lambda p: (p.clock_s, p.payload_id))

    if summary is not None:
        payloads.append(
            OverlayPayload(
                payload_id=f"{summary.match_id}-o-summary",
                match_id=summary.match_id,
                kind=OverlayKind.MATCH_SUMMARY,
                clock_s=max((p.clock_s for p in payloads), default=0),
                ttl_s=0,  # hold until dismissed
                summary=summary,
            )
        )
    return payloads


if __name__ == "__main__":
    from explainer import MatchExplainer
    from simulator import DEMO_AWAY, DEMO_HOME, MatchSimulator

    sim = MatchSimulator(DEMO_HOME, DEMO_AWAY, match_id="X0001", seed=148)
    sim.simulate(live=True)
    explainer = MatchExplainer(
        sim.match_id, DEMO_HOME.team_id, DEMO_AWAY.team_id,
        home_name=DEMO_HOME.name, away_name=DEMO_AWAY.name,
    )
    explainer.extend(list(sim.observable_stream()))

    payloads = build_overlay(explainer.insights(), explainer.summary())
    n_moments = sum(1 for p in payloads if p.kind == OverlayKind.KEY_MOMENT)
    print(f"{len(payloads)} overlay payloads ({n_moments} key moments + summary)")
    print("=" * 66)
    for payload in payloads[:12]:
        if payload.kind == OverlayKind.MATCH_SUMMARY:
            print(f"[{payload.clock_s:>5}s] summary   hold until dismissed")
            continue
        insight = payload.insight
        print(f"[{payload.clock_s:>5}s] {payload.ttl_s}s  {insight.headline}")
        print(f"            at ({insight.x:.0f}, {insight.y:.0f})  "
              f"cites {insight.evidence_event_ids}")
    print("...")
