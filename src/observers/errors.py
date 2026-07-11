"""Observer engine exceptions."""


class ObserverEngineError(Exception):
    """Base exception for observer engine failures."""


class DuplicateObserverError(ObserverEngineError):
    """Raised when two observers use the same stable name."""


class InvalidObserverOutputError(ObserverEngineError):
    """Raised when an observer returns data outside the observer contract."""
