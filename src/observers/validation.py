"""Validation for observer outputs."""

from core import Observation, ObserverResult
from observers.errors import InvalidObserverOutputError


class ObserverResultValidator:
    """Validate observer outputs before the engine accepts them."""

    def validate(self, observer_name: str, result: object) -> ObserverResult:
        """Return a typed result or raise when output is invalid."""

        if not isinstance(result, ObserverResult):
            raise InvalidObserverOutputError(
                f"{observer_name} returned {type(result).__name__}, "
                "expected ObserverResult."
            )

        if not result.observer.strip():
            raise InvalidObserverOutputError(
                f"{observer_name} returned an ObserverResult with no observer name."
            )

        if not isinstance(result.observations, list):
            raise InvalidObserverOutputError(
                f"{observer_name} returned observations that are not a list."
            )

        for observation in result.observations:
            if not isinstance(observation, Observation):
                raise InvalidObserverOutputError(
                    f"{observer_name} returned a non-Observation item."
                )

        return result
