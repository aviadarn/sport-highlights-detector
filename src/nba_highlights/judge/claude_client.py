import base64

from nba_highlights.judge.base import JudgeInput, JudgeVerdict

SYSTEM_PROMPT = (
    "You are labeling NBA broadcast clips for a highlight reel. Given a few "
    "keyframes and the commentary transcript for one clip, decide whether it "
    "shows a highlight-worthy PLAY: a made basket, dunk, three, layup, block, "
    "steal, or a notable defensive play. It is NOT a highlight if it is a "
    "timeout, replay of nothing, free-throw routine, stoppage, coach's "
    "challenge, or pure commentary/analysis with no live play. Return your "
    "verdict as structured output: is_highlight (bool), confidence (0-1), "
    "play_type (short label, e.g. 'dunk', 'three', 'block', or '' if none), "
    "and a one-sentence rationale."
)


class ClaudeHighlightJudge:
    def __init__(self, model: str = "claude-opus-4-8", client=None) -> None:
        self._model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _user_content(self, clip: JudgeInput) -> list[dict]:
        content: list[dict] = []
        for path in clip.keyframe_paths:
            with open(path, "rb") as fh:
                data = base64.standard_b64encode(fh.read()).decode("utf-8")
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": data}})
        content.append({"type": "text", "text":
            f"Clip {clip.scene_id} spans {clip.start_s:.1f}-{clip.end_s:.1f}s. "
            f"Commentary transcript: \"{clip.transcript}\". "
            f"Is this clip a highlight-worthy play?"})
        return content

    def judge(self, clip: JudgeInput) -> JudgeVerdict:
        client = self._get_client()
        response = client.messages.parse(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._user_content(clip)}],
            output_format=JudgeVerdict,
        )
        return response.parsed_output
