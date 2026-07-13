from nba_highlights.models import (
    ShotScore, Highlight, SignalBreakdown, Evidence,
)


def _build_highlight(hid: int, group: list[ShotScore]) -> Highlight:
    peak = max(group, key=lambda s: s.fused)
    keywords: list[str] = []
    for s in group:
        for k in s.matched_keywords:
            if k not in keywords:
                keywords.append(k)
    transcript = " ".join(s.transcript for s in group if s.transcript).strip()
    return Highlight(
        id=hid,
        start_s=min(s.start_s for s in group),
        end_s=max(s.end_s for s in group),
        score=peak.fused,
        signals=SignalBreakdown(audio=peak.audio, action=peak.action),
        evidence=Evidence(peak_loudness_s=peak.peak_loudness_s,
                          keywords=keywords, transcript=transcript),
        member_scenes=[s.scene_id for s in group],
    )


def merge_shots(shot_scores: list[ShotScore], threshold: float,
                max_gap_s: float) -> list[Highlight]:
    qualifying = sorted((s for s in shot_scores if s.fused >= threshold),
                        key=lambda s: s.start_s)
    highlights: list[Highlight] = []
    group: list[ShotScore] = []
    for s in qualifying:
        if group and (s.start_s - group[-1].end_s) <= max_gap_s:
            group.append(s)
        else:
            if group:
                highlights.append(_build_highlight(len(highlights), group))
            group = [s]
    if group:
        highlights.append(_build_highlight(len(highlights), group))
    return highlights
