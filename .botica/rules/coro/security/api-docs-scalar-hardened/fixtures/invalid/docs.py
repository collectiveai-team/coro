import scalar_fastapi
from scalar_fastapi import AgentScalarConfig, get_scalar_api_reference

BUNDLE = "/docs/static/scalar-api-reference.js"


def defaults():
    return get_scalar_api_reference(openapi_url="/openapi.json")


def telemetry_on():
    return get_scalar_api_reference(
        scalar_js_url=BUNDLE, telemetry=True, agent=AgentScalarConfig(disabled=True)
    )


def agent_left_on():
    return scalar_fastapi.get_scalar_api_reference(scalar_js_url=BUNDLE, telemetry=False)
