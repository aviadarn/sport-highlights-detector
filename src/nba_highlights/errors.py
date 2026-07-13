class IngestError(Exception):
    """Raised when downloading or extracting media fails."""


class ConfigError(Exception):
    """Raised on invalid configuration (bad weights, unknown backend)."""
