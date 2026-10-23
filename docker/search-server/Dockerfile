# The world search server: the /internal/worlds routes, with no baked catalog index. The validator
# installs each world pack's Lucene indexes at runtime over PUT /internal/worlds/<sha>.
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:21.0.12_8-jdk@sha256:92a2a4d7a928d057e7bd999c418d66c26a34eb9a0442f3ab67721c3f88110b2d $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

COPY docker/search-server/requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.2 && \
    python -m pip install --no-cache-dir --no-deps pyserini==1.0.0 && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY docker/search-server/patch_pyserini.py /tmp/patch_pyserini.py
RUN python /tmp/patch_pyserini.py && \
    rm /tmp/patch_pyserini.py && \
    python -c 'from pyserini.search.lucene import LuceneSearcher; print(LuceneSearcher.__name__)'

COPY src/search_engine /app/src/search_engine
COPY src/__init__.py /app/src/

EXPOSE 5632
ENV PORT=5632 WORLDS_DIR=/tmp/worlds _JAVA_OPTIONS="-Xmx1g -Xms256m"

# Sync workers: one JVM per process. Workers share installed
# worlds through WORLDS_DIR.
CMD ["sh", "-c", "exec gunicorn --workers \"${SEARCH_WORKERS:-2}\" --worker-class sync --bind \"0.0.0.0:${PORT}\" --chdir /app --timeout 120 --error-logfile - 'src.search_engine.worlds:app()'"]
