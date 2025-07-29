
FROM docker.cnb.cool/ecoslab/rtl2gds/ieda:latest AS builder
RUN bash build.sh

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

COPY --from=builder /opt/iEDA /app/iEDA

ENV iEDA=/app/iEDA/bin/iEDA
ENV MCP_SERVER_TYPE=stdio
ENV MCP_SERVER_URL=0.0.0.0
ENV MCP_SERVER_PORT=3002

ENV WORKSPACE=/app/iEDA/scripts/design/sky130_gcd

# Place executables in the environment at the front of the path
ENV PATH="/app/iEDA/bin:/app/.venv/bin:$PATH"

# CMD ["iEDA", "-v"]
ENTRYPOINT ["mcp-iEDA"]