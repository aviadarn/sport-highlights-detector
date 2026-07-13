import nba_highlights
from nba_highlights.errors import ConfigError, IngestError


def test_version_is_string():
    assert isinstance(nba_highlights.__version__, str)


def test_errors_are_exceptions():
    assert issubclass(IngestError, Exception)
    assert issubclass(ConfigError, Exception)
