"""Observer registration and discovery."""

from typing import Iterable, Protocol

from observers.base import Observer
from observers.errors import DuplicateObserverError


class ObserverProvider(Protocol):
    """Provider capable of returning observers for discovery."""

    def get_observers(self) -> Iterable[Observer]:
        """Return observer instances available from this provider."""


class ObserverRegistry:
    """Stores directly registered observers and discovery providers."""

    def __init__(
        self,
        observers: Iterable[Observer] | None = None,
        providers: Iterable[ObserverProvider] | None = None,
    ) -> None:
        self._observers: list[Observer] = []
        self._providers: list[ObserverProvider] = []

        for observer in observers or []:
            self.register(observer)

        for provider in providers or []:
            self.register_provider(provider)

    def register(self, observer: Observer) -> None:
        """Register one observer instance."""

        self._observers.append(observer)

    def register_provider(self, provider: ObserverProvider) -> None:
        """Register a provider for lazy observer discovery."""

        self._providers.append(provider)

    def discover(self) -> list[Observer]:
        """Return all registered and provider-discovered observers."""

        observers = [*self._observers]
        for provider in self._providers:
            observers.extend(provider.get_observers())

        self._ensure_unique_names(observers)
        return observers

    def _ensure_unique_names(self, observers: Iterable[Observer]) -> None:
        seen: set[str] = set()

        for observer in observers:
            if observer.name in seen:
                raise DuplicateObserverError(
                    f"Observer name must be unique: {observer.name}"
                )
            seen.add(observer.name)
