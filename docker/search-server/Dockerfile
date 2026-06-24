FROM ghcr.io/oro-ai/oro/search-server-base:latest

WORKDIR /app

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
