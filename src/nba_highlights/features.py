from pydantic import BaseModel

FEATURE_NAMES = [
    "loudness_n", "prosody_n", "keywords_n", "keyword_count",
    "action_score", "shot_duration_s", "rel_position",
]


class ShotFeatures(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    loudness_n: float = 0.0
    prosody_n: float = 0.0
    keywords_n: float = 0.0
    keyword_count: float = 0.0
    action_score: float = 0.0
    shot_duration_s: float = 0.0
    rel_position: float = 0.0


def to_vector(feats: ShotFeatures) -> list[float]:
    return [getattr(feats, name) for name in FEATURE_NAMES]
