from nba_highlights.scenes import detect_scenes
from nba_highlights.models import Shot


def test_detect_scenes_maps_boundaries_to_shots():
    shots = detect_scenes("v.mp4", 30.0,
                          detector_fn=lambda p: [(0.0, 5.0), (5.0, 12.0)])
    assert shots == [Shot(scene_id=0, start_s=0.0, end_s=5.0),
                     Shot(scene_id=1, start_s=5.0, end_s=12.0)]


def test_detect_scenes_empty_is_single_shot_over_duration():
    shots = detect_scenes("v.mp4", 42.0, detector_fn=lambda p: [])
    assert shots == [Shot(scene_id=0, start_s=0.0, end_s=42.0)]
