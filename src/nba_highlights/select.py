from nba_highlights.models import Highlight


def select_highlights(highlights: list[Highlight], threshold: float) -> list[Highlight]:
    kept = sorted((h for h in highlights if h.score >= threshold),
                  key=lambda h: h.start_s)
    return [h.model_copy(update={"id": i}) for i, h in enumerate(kept)]
