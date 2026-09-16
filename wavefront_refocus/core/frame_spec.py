"""Parse a user-typed list of frame indices.

Accepts comma- (or whitespace-) separated indices and inclusive ranges, e.g.
``"1,38,39,100"`` or ``"0-9, 25, 40-45"``. Used by the interpolation panel to
restrict which frames get (re)computed, so already-good planes are left alone.

Ranges are written ``lo-hi`` (also ``lo..hi`` or ``lo:hi``) with no spaces
around the operator, since whitespace separates entries.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

_SEP = re.compile(r"[,;\s]+")
# Whitespace is a separator, so a range must be written without spaces around
# the operator ("10-20", "10..20", "10:20") to survive the split above.
_RANGE = re.compile(r"^(\d+)(?:-|\.\.|:)(\d+)$")


def parse_frame_spec(text: str, valid: Optional[Iterable[int]] = None) -> List[int]:
    """Return the sorted, de-duplicated frame indices named by ``text``.

    ``valid``, when given, is the set of existing frame indices; anything
    outside it raises :class:`ValueError` so the user hears about a typo
    instead of silently interpolating nothing.

    An empty/blank string yields an empty list, which callers read as
    "no restriction".
    """
    text = (text or "").strip()
    if not text:
        return []

    valid_set: Optional[Set[int]] = set(valid) if valid is not None else None
    out: Set[int] = set()

    for token in _SEP.split(text):
        if not token:
            continue
        m = _RANGE.match(token)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            span = range(lo, hi + 1)
        else:
            try:
                span = [int(token)]
            except ValueError:
                raise ValueError(f"cannot read {token!r} as a frame index or range")
        for i in span:
            if valid_set is not None and i not in valid_set:
                raise ValueError(f"frame {i} is outside the loaded sequence")
            out.add(i)

    return sorted(out)
