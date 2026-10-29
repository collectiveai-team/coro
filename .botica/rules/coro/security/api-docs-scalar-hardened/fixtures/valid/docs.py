import scalar_fastapi
from fastapi.openapi.docs import get_redoc_html
from scalar_fastapi import AgentScalarConfig, get_scalar_api_reference

BUNDLE = "/docs/static/scalar-api-reference.js"


def hardened():
    return get_scalar_api_reference(
        openapi_url="/openapi.json",
        scalar_js_url=BUNDLE,
        telemetry=False,
        agent=AgentScalarConfig(disabled=True),
    )


def qualified():
    return scalar_fastapi.get_scalar_api_reference(
        scalar_js_url=BUNDLE, telemetry=False, agent=AgentScalarConfig(disabled=True)
    )


def splat(settings):
    return get_scalar_api_reference(**settings)


def other_reference():
    return get_redoc_html(openapi_url="/openapi.json", title="API")
