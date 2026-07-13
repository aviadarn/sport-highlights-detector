from nba_highlights.models import Shot


def _default_detector(video_path: str) -> list[tuple[float, float]]:
    from scenedetect import detect, AdaptiveDetector
    scene_list = detect(video_path, AdaptiveDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def detect_scenes(video_path: str, duration_s: float,
                  detector_fn=_default_detector) -> list[Shot]:
    boundaries = detector_fn(video_path)
    if not boundaries:
        boundaries = [(0.0, duration_s)]
    return [Shot(scene_id=i, start_s=s, end_s=e)
            for i, (s, e) in enumerate(boundaries)]
