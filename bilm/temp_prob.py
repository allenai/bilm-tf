from typing import Tuple, AbstractSet, Mapping, List, TextIO

from attr import attrs, attrib
from immutablecollections import ImmutableDict, ImmutableSet
from vistautils.attrutils import attrib_immutable
from vistautils.preconditions import check_arg

WordPair = Tuple[str, str]


@attrs(kw_only=True, auto_attribs=True)
class TempProb:
    _raw_before_counts: Mapping[WordPair, int] = attrib_immutable(ImmutableDict)
    _raw_after_counts: Mapping[WordPair, int] = attrib_immutable(ImmutableDict)
    _smoothing_pseudo_count: float = 0
    seen_as_before: AbstractSet[WordPair] = attrib(init=False, kw_only=True)
    seen_as_after: AbstractSet[WordPair] = attrib(init=False, kw_only=True)

    @staticmethod
    def from_file(f: TextIO, *, smoothing_pseudo_count: float=0.0) -> 'TempProb':
        before_counts: List[Tuple[WordPair, int]] = []
        after_counts: List[Tuple[WordPair, int]] = []

        for line in f:
            fields = line.split("\t")
            direction = fields[2]
            if direction == 'before':
                before_counts.append(((fields[0], fields[1]), int(fields[3])))
            elif direction == 'after':
                after_counts.append(((fields[0], fields[1]), int(fields[3])))

        return TempProb(raw_before_counts=before_counts,
                        raw_after_counts=after_counts,
                        smoothing_pseudo_count=smoothing_pseudo_count)

    # TODO: accomodate other values than before and after
    def smoothed_fraction_before(self, w1: str, w2: str) -> float:
        numerator = self._raw_before_counts.get((w1, w2), 0.0) + self._smoothing_pseudo_count
        return numerator / (numerator
                            + self._raw_after_counts.get((w1, w2), 0.0) +
                            self._smoothing_pseudo_count)

    def __attrs_post_init__(self) -> 'None':
        check_arg(self._smoothing_pseudo_count >= 0, "Smoothing pseudo-count must be non-negative")

    @seen_as_before.default
    def seen_as_before_init(self) -> AbstractSet[WordPair]:
        return ImmutableSet.of(self._raw_before_counts.keys())

    @seen_as_after.default
    def seen_as_after_init(self) -> AbstractSet[WordPair]:
        return ImmutableSet.of(self._raw_after_counts.keys())
