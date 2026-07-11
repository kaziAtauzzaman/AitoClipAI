"""Base observer interface."""

from abc import ABC, abstractmethod

from core import ObserverResult
from observers.contracts import ObserverContext


class Observer(ABC):
    """Base interface implemented by every observer."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable unique observer name."""

    @property
    def order(self) -> int:
        """Execution order; lower values run first."""

        return 1000

    def setup(self, context: ObserverContext) -> None:
        """Prepare the observer before execution."""

    @abstractmethod
    def observe(self, context: ObserverContext) -> ObserverResult:
        """Run the observer and return structured observations."""

    def teardown(self, context: ObserverContext) -> None:
        """Release observer resources after execution."""
