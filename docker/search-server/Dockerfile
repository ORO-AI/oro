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
ENV JAVA_OPTS="-Xmx4g -Xms2g"

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
