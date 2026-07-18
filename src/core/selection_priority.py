"""Finite deterministic priority shared by scoring and selection."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import math


CANDIDATE_SCORE_DECIMAL_PLACES = 6


@dataclass(frozen=True, slots=True)
class SelectionPriorityContract:
    """Normalize selection scores onto one finite, documented alphabet.

    Candidate scoring publishes values in ``[0, 1]`` rounded to six decimal
    places.  Selection may deliberately use fewer of those decimal places, but
    may never claim precision that scoring does not produce. Equal normalized
    ranks retain input order; no candidate payload field can improve an equal
    rank later in a stream.
    """

    score_decimal_places: int = CANDIDATE_SCORE_DECIMAL_PLACES

    def __post_init__(self) -> None:
        value = self.score_decimal_places
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Selection priority precision must be an integer.")
        if not 0 <= value <= CANDIDATE_SCORE_DECIMAL_PLACES:
            raise ValueError(
                "Selection priority precision must be between zero and the "
                f"scorer's {CANDIDATE_SCORE_DECIMAL_PLACES}-decimal output "
                "precision."
            )

    @property
    def scale(self) -> int:
        """Return integer units per score point."""

        return 10**self.score_decimal_places

    @property
    def rank_count(self) -> int:
        """Return the complete finite alphabet size, including zero and one."""

        return self.scale + 1

    @property
    def maximum_strictly_improving_chain_length(self) -> int:
        """Maximum number of ranks in a strict forward improvement path."""

        return self.rank_count

    @property
    def identity(self) -> tuple[str, int, int, int, str, str]:
        """Return the immutable compatibility identity for this contract."""

        return (
            "selection-priority-v1",
            self.score_decimal_places,
            0,
            self.scale,
            "decimal-half-even",
            "stable-input-order",
        )

    def normalize(self, score: float) -> int:
        """Return the finite rank for one public score payload."""

        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(float(score))
        ):
            raise ValueError("Selection priorities require a finite numeric score.")
        numeric = Decimal(str(score))
        if numeric < 0 or numeric > 1:
            raise ValueError("Selection priority scores must be between zero and one.")
        return int(
            (numeric * self.scale).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
        )

    def ordering_key(self, score: float) -> tuple[int]:
        """Return the ascending sort key; higher normalized ranks sort first."""

        return (-self.normalize(score),)


DEFAULT_SELECTION_PRIORITY_CONTRACT = SelectionPriorityContract()
