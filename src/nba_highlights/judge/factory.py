from nba_highlights.judge.stub import StubHighlightJudge


def load_judge(backend: str, model: str):
    if backend == "claude":
        from nba_highlights.judge.claude_client import ClaudeHighlightJudge
        return ClaudeHighlightJudge(model)
    return StubHighlightJudge()
