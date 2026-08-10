"""The documented Deepgram coverage matches the code.

ADR 0015 and the README state how much of Deepgram's pre-recorded contract is
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

from coro.api.deepgram.listen import _UNSUPPORTED_PARAMS

_ADR = Path(__file__).resolve().parents[1] / "docs/adr/0015-vendor-native-endpoints.md"

# `request` is the audio body itself, not a query parameter.
VENDOR_PARAMS = {
    name
    for name in inspect.signature(MediaClient.transcribe_file).parameters
    if name not in ("self", "request_options", "request")
}
REFUSED = set(_UNSUPPORTED_PARAMS)
HONOURED = {"diarize", "utterances", "language"}
IGNORED = VENDOR_PARAMS - REFUSED - HONOURED


class TestRefusalListIsCoherent:
    def test_every_refused_parameter_is_a_real_vendor_parameter(self):
        # Catches a typo that would silently never match a request.
        assert set() == REFUSED - VENDOR_PARAMS

    def test_refused_and_honoured_do_not_overlap(self):
        assert set() == REFUSED & HONOURED

    def test_every_vendor_parameter_is_accounted_for(self):
        assert REFUSED | HONOURED | IGNORED == VENDOR_PARAMS

    def test_only_silently_harmful_parameters_are_refused(self):
        # Everything else is accepted and ignored so a client's standard
        # parameter bundle still works. These two cannot be ignored safely:
        # `redact` would return unredacted text under a redaction request, and
        # `callback` would leave a client awaiting a webhook that never fires.
        assert {"redact", "callback", "callback_method"} == REFUSED


class TestDocumentedCountsMatchTheCode:
    @pytest.mark.parametrize(
        "label,expected",
        [("honoured", 3), ("refused", 3), ("ignored", 31)],
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
        assert match is not None, "ADR 0015 coverage table row not found"
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
