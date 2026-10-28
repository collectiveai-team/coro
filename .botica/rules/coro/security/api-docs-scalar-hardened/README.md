## Directive

Each call to `get_scalar_api_reference()` passes `scalar_js_url`, `telemetry=False` and `agent`
(the hardened call disables it with `AgentScalarConfig(disabled=True)`). A call that leaves one of
the three to its default, or sets `telemetry=True`, is an error.

## Why

The defaults of `scalar_fastapi` call out from the `/docs` page. `scalar_js_url` is an unversioned
jsDelivr URL, resolved again on every load and with no SRI. `telemetry` is `True`. `agent` is the
hosted AI chat of Scalar: it is on by default on localhost and uploads the served OpenAPI document
to `api.scalar.com` on the first message. ADR 0013 records why the page contacts its own origin and
nothing else; `coro/api/docs.py` holds the hardened call.

## Do

```python
get_scalar_api_reference(
    openapi_url="/openapi.json",
    scalar_js_url="/docs/static/scalar-api-reference.js",
    telemetry=False,
    agent=AgentScalarConfig(disabled=True),
)
```

## Don't

```python
get_scalar_api_reference(openapi_url="/openapi.json")
get_scalar_api_reference(scalar_js_url=URL, telemetry=True, agent=AGENT)
```

## Suppression

A call that passes `**kwargs` is not reported, because the rule cannot see what the mapping holds.
`# botica: ignore[coro/api-docs-scalar-hardened]` on the line of the call silences one finding,
with the reason in the comment.
