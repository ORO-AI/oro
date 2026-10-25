FROM python:3.11-slim

RUN python -m pip install --no-cache-dir --no-deps 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
RUN python -m pip install --no-cache-dir fastapi==0.136.0 'uvicorn[standard]==0.45.0' requests==2.33.0 'urllib3>=2.8.0' 'setuptools>=84.1.0' 'certifi>=2026.10.19'

WORKDIR /app

COPY subnet/__init__.py /app/subnet/__init__.py
COPY subnet/validator/__init__.py /app/subnet/validator/__init__.py
COPY subnet/validator/session_errors.py /app/subnet/validator/session_errors.py
COPY subnet/validator/session_service.py /app/subnet/validator/session_service.py
COPY src/__init__.py /app/src/__init__.py
COPY src/agent/proxy_client.py /app/src/agent/proxy_client.py
COPY tests/__init__.py /app/tests/__init__.py
COPY tests/integration/__init__.py /app/tests/integration/__init__.py
COPY tests/integration/session_runtime.py /app/tests/integration/session_runtime.py

ENV PYTHONPATH=/app

CMD ["python", "-m", "tests.integration.session_runtime"]
