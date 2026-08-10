"""Vendor-native boundary schemas and response projections.

Each vendor gets its own endpoint implementing that vendor's own contract, so
the OpenAI-compatible surface is never extended with values OpenAI does not
define. Each module owns one vendor's boundary models and the projection that
fills them. See ADR 0010 for the fidelity policy these are held to.

Import the submodule you need; this package deliberately re-exports nothing.
"""
