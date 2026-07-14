import base64
from nba_highlights.judge.base import JudgeInput, JudgeVerdict
from nba_highlights.judge.claude_client import ClaudeHighlightJudge
from nba_highlights.judge.factory import load_judge
from nba_highlights.judge.stub import StubHighlightJudge


class _FakeMessages:
    def __init__(self, recorder):
        self._rec = recorder

    def parse(self, **kwargs):
        self._rec.update(kwargs)

        class _Resp:
            parsed_output = JudgeVerdict(is_highlight=True, confidence=0.77,
                                         play_type="dunk", rationale="slam")
        return _Resp()


class _FakeClient:
    def __init__(self, recorder):
        self.messages = _FakeMessages(recorder)


def test_claude_judge_builds_request_and_maps_verdict(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # minimal bytes; content is base64-encoded verbatim
    clip = JudgeInput(scene_id=0, start_s=1.0, end_s=3.0,
                      keyframe_paths=[str(img)], transcript="what a dunk")
    rec: dict = {}
    verdict = ClaudeHighlightJudge(client=_FakeClient(rec)).judge(clip)

    assert verdict.is_highlight is True and verdict.confidence == 0.77
    assert rec["model"] == "claude-opus-4-8"
    assert rec["output_format"] is JudgeVerdict
    content = rec["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    text_blocks = [b for b in content if b["type"] == "text"]
    assert len(image_blocks) == 1 and len(text_blocks) == 1
    assert image_blocks[0]["source"]["data"] == base64.standard_b64encode(b"\xff\xd8\xff").decode()
    assert "what a dunk" in text_blocks[0]["text"]
    # no sampling params / budget_tokens (400 on Claude 4.x)
    for banned in ("temperature", "top_p", "top_k", "thinking", "budget_tokens"):
        assert banned not in rec


def test_load_judge_backends():
    assert isinstance(load_judge("stub", "claude-opus-4-8"), StubHighlightJudge)
    assert isinstance(load_judge("claude", "claude-opus-4-8"), ClaudeHighlightJudge)
