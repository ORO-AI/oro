FROM python:3.10-slim

# Copy JDK 21 from eclipse-temurin (multi-arch index digest). Pinned because an index
# manifest records its java_version and the search image must match it exactly;
# the floating 21-jdk tag moved from 21.0.12 to 21.0.12.1.
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:21.0.12_8-jdk@sha256:92a2a4d7a928d057e7bd999c418d66c26a34eb9a0442f3ab67721c3f88110b2d $JAVA_HOME $JAVA_HOME
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
    "--fields", "catalog_brand", "catalog_candidate", "catalog_category", \
    "catalog_price", "catalog_source", "catalog_source_variant", \
    "--storePositions", "--storeDocvectors", "--storeRaw"]
