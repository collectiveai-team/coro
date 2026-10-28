"""Markers for the tests that need a socket while pytest runs with ``--disable-socket``."""

import pytest

LOOPBACK_HOSTS = ["127.0.0.1", "::1", "localhost"]

# An integration test that talks only to a server it starts on the loopback
# interface. Use it as ``pytestmark = LOOPBACK`` in a module or a class.
LOOPBACK = [pytest.mark.integration, pytest.mark.allow_hosts(LOOPBACK_HOSTS)]
