FROM python:3.10-slim

# Copy JDK 21 from eclipse-temurin (handles multi-arch automatically)
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:21-jdk $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install pyserini for index building
RUN pip install --no-cache-dir pyserini

WORKDIR /data

# Default command builds indexes from /data/resources to /data/indexes
ENTRYPOINT ["python", "-m", "pyserini.index.lucene", \
    "--collection", "JsonCollection", \
    "--input", "/data/resources", \
    "--index", "/data/indexes", \
    "--generator", "DefaultLuceneDocumentGenerator", \
    "--threads", "1", \
    "--storePositions", "--storeDocvectors", "--storeRaw"]
