# syntax=docker/dockerfile:1.7
# check=skip=UndefinedVar
#
# Slim runtime image for the coro ASR server.
#
# Multi-stage build: a heavy `builder` stage resolves and installs the project
# venv (needs uv + python headers + build tooling); the final `runtime` stage
# carries only the Python interpreter, ffmpeg and the prebuilt venv — no uv, no
# compilers, no source tree — so it stays as light as possible.
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

# ---------------------------------------------------------------------------
# base — shared Python runtime layer for both builder and runtime stages.
# Keeping this common ensures the venv copied into runtime resolves against the
# exact same interpreter it was built against.
# ---------------------------------------------------------------------------
FROM ${CORE_IMAGE} AS base

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/usr/local/cuda/bin:/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Python (deadsnakes for a consistent 3.12 across bases) and ca-certificates
# for model downloads. ffmpeg is intentionally added only in the runtime stage.
#
# The python/python3 symlinks are created explicitly *after* autoremove: the
# distro python3 package (pulled transitively by software-properties-common)
# owns /usr/bin/python3, so autoremove deletes that link. Pointing both names
# straight at the deadsnakes binary keeps the interpreter path stable and
# identical across the builder and runtime stages (the copied venv depends on
# /usr/bin/python3 resolving in the final image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# builder — resolve dependencies and install the project into /app/.venv.
# Carries uv and python headers; none of this leaks into the runtime image.
# ---------------------------------------------------------------------------
FROM base AS builder

ARG PYTHON_VERSION=3.12
ARG EXTRA=cpu
# hatch-vcs derives the version from git, which isn't in the build context.
# CI passes the resolved version here; defaults to 0.0.0 for plain `docker build`.
ARG CORO_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${CORO_VERSION} \
    HATCH_VCS_PRETEND_VERSION=${CORO_VERSION} \
    UV_PYTHON_PREFERENCE=system \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Python headers for packages that build native extensions during resolution.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-source /uv /uvx /bin/

WORKDIR /app

# Resolve dependencies first (cached) using only the lock + project metadata,
# then copy the source and install the project itself. --no-dev keeps the dev
# dependency-group out; --extra selects cpu or cuda wheels; --no-editable bakes
# the project into site-packages so runtime needs no source tree on disk.
COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable --extra ${EXTRA}

COPY coro ./coro

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ${EXTRA}

# ---------------------------------------------------------------------------
# runtime — the shipped image: interpreter + ffmpeg + the prebuilt venv only.
# ---------------------------------------------------------------------------
FROM base AS runtime

# ffmpeg for audio decoding; everything else is provided by the copied venv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The venv is self-contained and installed at the same path it was built at,
# so its console scripts and interpreter symlink resolve unchanged.
COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

# Defaults bind 0.0.0.0:8000 (coro/settings.py). Override behaviour with
# CORO_* env vars or CLI flags appended after the entrypoint.
ENTRYPOINT ["coro"]
