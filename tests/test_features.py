from nba_highlights.features import ShotFeatures, FEATURE_NAMES, to_vector


def test_feature_names_frozen_order():
    assert FEATURE_NAMES == ["loudness_n", "prosody_n", "keywords_n",
                             "keyword_count", "action_score", "shot_duration_s",
                             "rel_position"]


def test_to_vector_follows_feature_names():
    f = ShotFeatures(scene_id=1, start_s=0.0, end_s=2.0, loudness_n=0.9,
                     prosody_n=0.8, keywords_n=0.6, keyword_count=2.0,
                     action_score=0.7, shot_duration_s=2.0, rel_position=0.5)
    assert to_vector(f) == [0.9, 0.8, 0.6, 2.0, 0.7, 2.0, 0.5]


def test_defaults_zero():
    f = ShotFeatures(scene_id=0, start_s=0.0, end_s=1.0)
    assert to_vector(f) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
