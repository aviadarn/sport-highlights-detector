from nba_highlights.models import (
    VideoInfo, AudioSignals, ActionSignals, ShotScore,
    SignalBreakdown, Evidence, Highlight, Report,
)


def test_shot_score_defaults():
    s = ShotScore(scene_id=3, start_s=10.0, end_s=12.0,
                  audio=AudioSignals(loudness=0.9), action=ActionSignals())
    assert s.fused == 0.0
    assert s.matched_keywords == []
    assert s.peak_loudness_s is None


def test_report_round_trip():
    r = Report(
        video=VideoInfo(source="u", title="t", duration_s=100.0),
        config={"threshold": 0.5},
        generated_at="2026-07-13T00:00:00Z",
        highlights=[Highlight(
            id=0, start_s=10.0, end_s=13.0, score=0.8,
            signals=SignalBreakdown(audio=AudioSignals(loudness=0.9),
                                    action=ActionSignals(score=0.7, label="dunking basketball")),
            evidence=Evidence(peak_loudness_s=11.0, keywords=["dunk"], transcript="oh my"),
            member_scenes=[4, 5],
        )],
    )
    dumped = r.model_dump_json()
    back = Report.model_validate_json(dumped)
    assert back.highlights[0].signals.action.label == "dunking basketball"
    assert back.highlights[0].member_scenes == [4, 5]
    assert back.video.duration_s == 100.0
