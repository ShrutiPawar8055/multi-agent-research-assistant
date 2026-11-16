FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY sample_docs/ ./sample_docs/
RUN pip install --no-cache-dir ".[api,gemini]"
ENV PORT=8080
CMD ["sh", "-c", "uvicorn researchcrew.api:app --host 0.0.0.0 --port ${PORT}"]
