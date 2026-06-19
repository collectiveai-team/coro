# syntax=docker/dockerfile:1.7
#
# Slim runtime image for the coro ASR server.
#
# Built in two flavours via build args, mirroring the cpu/cuda extras in
# pyproject.toml (which are mutually exclusive — pick exactly one):
#
#   CPU:  --build-arg CORE_IMAGE=ubuntu:noble \
#         --build-arg EXTRA=cpu
#   GPU:  --build-arg CORE_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 \
#         --build-arg EXTRA=cuda
#
# Unlike .devcontainer/Dockerfile this ships no dev tooling (node, zsh,
# opencode, docker-in-docker); only the Python runtime, ffmpeg and the app.

ARG UV_VERSION=0.11.12
ARG CORE_IMAGE=ubuntu:noble

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-source

FROM ${CORE_IMAGE} AS runtime

ARG PYTHON_VERSION=3.12
ARG EXTRA=cpu
# hatch-vcs derives the version from git, which isn't in the build context.
# CI passes the resolved version here; defaults to 0.0.0 for plain `docker build`.
ARG CORO_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${CORO_VERSION} \
    HATCH_VCS_PRETEND_VERSION=${CORO_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_PREFERENCE=system \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/usr/local/cuda/bin:/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Runtime OS deps only: Python (deadsnakes for a consistent 3.12 across bases),
# ffmpeg for audio decoding, ca-certificates for model downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        ffmpeg \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-source /uv /uvx /bin/

WORKDIR /app

# Resolve dependencies first (cached) using only the lock + project metadata,
# then copy the source and install the project itself. --no-dev keeps the dev
# dependency-group out of the image; --extra selects cpu or cuda wheels.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ${EXTRA}

COPY coro ./coro

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ${EXTRA}

EXPOSE 8000

# Defaults bind 0.0.0.0:8000 (coro/settings.py). Override behaviour with
# CORO_* env vars or CLI flags appended after the entrypoint.
ENTRYPOINT ["coro"]
