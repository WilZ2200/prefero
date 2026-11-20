FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 prefero

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . && pip check

COPY --chown=prefero:prefero app/ app/
COPY --chown=prefero:prefero data/test_dce.xlsx data/test_dce.xlsx
COPY .streamlit/config.toml .streamlit/config.toml
RUN mkdir -p data/raw data/processed data/generated \
    && chown prefero:prefero data/raw data/processed data/generated

ENV HOME=/home/prefero
USER prefero
EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1
CMD ["streamlit", "run", "app/Home.py", "--server.port=7860", "--server.address=0.0.0.0"]
