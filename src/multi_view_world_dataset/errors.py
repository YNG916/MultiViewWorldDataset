class MVWDError(RuntimeError):
    """Base error for generation failures."""


class ConfigurationError(MVWDError):
    """Tracked dataset or machine configuration is invalid."""


class GeometryError(MVWDError):
    """A coordinate transform or calibration invariant failed."""


class SimulatorUnavailableError(MVWDError):
    """The external simulator stack could not be resolved."""


class SampleRejected(MVWDError):
    """A stochastic candidate failed QA and should be recorded as rejected."""

    def __init__(self, reason: str, details: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}

