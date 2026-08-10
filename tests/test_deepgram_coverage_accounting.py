"""The documented Deepgram coverage matches the code.

ADR 0010 and the README state how much of Deepgram's pre-recorded contract is
implemented. Those numbers were wrong once already, and prose cannot be
type-checked — so the counts are asserted against the vendor SDK's own
signature and against the refusal list the endpoint actually uses.

When a `deepgram-sdk` bump adds a parameter these fail, which is the point:
the new parameter has to be triaged into honoured, refused, or ignored, and
the coverage docs updated to match.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from deepgram.listen.v1.media.client import MediaClient

from coro.api.v1.listen import _UNSUPPORTED_PARAMS

_ADR = Path(__file__).resolve().parents[1] / "docs/adr/0010-vendor-native-endpoints.md"

# `request` is the audio body itself, not a query parameter.
VENDOR_PARAMS = {
    name
    for name in inspect.signature(MediaClient.transcribe_file).parameters
    if name not in ("self", "request_options", "request")
}
REFUSED = set(_UNSUPPORTED_PARAMS) | {"multichannel"}
HONOURED = {"diarize", "utterances", "language"}
IGNORED = VENDOR_PARAMS - REFUSED - HONOURED


class TestRefusalListIsCoherent:
    def test_every_refused_parameter_is_a_real_vendor_parameter(self):
        # Catches a typo that would silently never match a request.
        assert REFUSED - VENDOR_PARAMS == set()

    def test_refused_and_honoured_do_not_overlap(self):
        assert REFUSED & HONOURED == set()

    def test_every_vendor_parameter_is_accounted_for(self):
        assert REFUSED | HONOURED | IGNORED == VENDOR_PARAMS

    def test_feature_modifiers_are_covered_by_their_head_parameter(self):
        # `custom_topic_mode` alone is meaningless; refusing `custom_topic`
        # is what makes ignoring the modifier safe.
        for head, modifier in [
            ("custom_topic", "custom_topic_mode"),
            ("custom_intent", "custom_intent_mode"),
        ]:
            assert head in REFUSED
            assert modifier in IGNORED


class TestDocumentedCountsMatchTheCode:
    @pytest.mark.parametrize(
        "label,expected",
        [("honoured", 3), ("refused", 16), ("ignored", 18)],
    )
    def test_computed_counts(self, label: str, expected: int):
        computed = {"honoured": HONOURED, "refused": REFUSED, "ignored": IGNORED}[label]
        assert len(computed) == expected

    def test_adr_states_the_computed_totals(self):
        text = _ADR.read_text(encoding="utf-8")
        match = re.search(
            r"(\d+) pre-recorded parameters \|[^|]*\| "
            r"(\d+) honoured, (\d+) refused, (\d+) accepted",
            text,
        )
        assert match is not None, "ADR 0010 coverage table row not found"
        total, honoured, refused, ignored = (int(g) for g in match.groups())
        assert (total, honoured, refused, ignored) == (
            len(VENDOR_PARAMS),
            len(HONOURED),
            len(REFUSED),
            len(IGNORED),
        )

    def test_adr_lists_every_refused_parameter_by_name(self):
        text = _ADR.read_text(encoding="utf-8")
        refused_section = text.split("**Refused (")[1].split("**Accepted")[0]
        missing = {name for name in REFUSED if f"`{name}`" not in refused_section}
        assert missing == set(), f"undocumented refusals: {sorted(missing)}"
