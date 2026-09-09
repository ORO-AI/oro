FROM python:3.11-slim

RUN pip install --no-cache-dir fastapi==0.136.0 'uvicorn[standard]==0.45.0'

WORKDIR /app

COPY subnet/__init__.py /app/subnet/__init__.py
COPY subnet/validator/__init__.py /app/subnet/validator/__init__.py
COPY subnet/validator/session_errors.py /app/subnet/validator/session_errors.py
COPY subnet/validator/session_service.py /app/subnet/validator/session_service.py
COPY tests/__init__.py /app/tests/__init__.py
COPY tests/integration/__init__.py /app/tests/integration/__init__.py
COPY tests/integration/session_runtime.py /app/tests/integration/session_runtime.py

ENV PYTHONPATH=/app

CMD ["python", "-m", "tests.integration.session_runtime"]
