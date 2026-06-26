FROM ghcr.io/oro-ai/oro/search-server-base:latest

WORKDIR /app

# Bundle async-profiler so we can JVMTI-attach to the embedded Lucene JVM
# (pyserini) at runtime without restarting the container. ~5 MiB; opt-in
# only — never invoked unless an operator runs `asprof` manually inside
# the container.
ARG ASPROF_VERSION=3.0
RUN set -eux; \
    arch="$(uname -m)"; case "$arch" in \
        x86_64) asprof_arch=x64 ;; \
        aarch64|arm64) asprof_arch=arm64 ;; \
        *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates; \
    mkdir -p /opt/async-profiler; \
    curl -fsSL "https://github.com/async-profiler/async-profiler/releases/download/v${ASPROF_VERSION}/async-profiler-${ASPROF_VERSION}-linux-${asprof_arch}.tar.gz" \
        | tar xz --strip-components=1 -C /opt/async-profiler; \
    ln -s /opt/async-profiler/bin/asprof /usr/local/bin/asprof; \
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# Only the code layer — changes on every release
COPY src/search_engine /app/src/search_engine
COPY src/__init__.py /app/src/

# Copy entrypoint script
COPY docker/search-server/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port (default 5632, configurable via ENV)
EXPOSE 5632

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=5632
# Pyserini embeds a JVM via JNI (Lucene). The JVM reads _JAVA_OPTIONS
# directly at startup; JAVA_OPTS is non-standard and was being ignored.
# 2g cap fits comfortably for the prod workload (sustained ≥17 rps at 60
# concurrent) and saves ~2 GiB on memory-constrained hosts.
ENV _JAVA_OPTIONS="-Xmx2g -Xms1g"

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
