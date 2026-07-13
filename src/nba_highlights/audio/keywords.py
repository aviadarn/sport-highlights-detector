from nba_highlights.interfaces import Transcript


def shot_keywords(transcript: Transcript, start_s: float, end_s: float,
                  lexicon: list[str]) -> tuple[float, list[str]]:
    text = transcript.text_in(start_s, end_s).lower()
    if not text:
        return 0.0, []
    matched = [phrase for phrase in lexicon if phrase.lower() in text]
    return float(len(matched)), matched
