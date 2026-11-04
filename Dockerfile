FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home sentinel
COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit
ENV SENTINELML_ROOT=/app SENTINELML_ARTIFACTS=/app/artifacts
RUN mkdir -p /app/artifacts /app/data/raw && chown -R sentinel:sentinel /app
USER sentinel
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
