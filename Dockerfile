FROM python:3.13-bookworm

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip3 install -U -r requirements.txt

# Copy application code
COPY qwen3-asr-server/ ./qwen3-asr-server/

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs').read()" || exit 1

# Run the FastAPI server with uvicorn
CMD ["python3", "-u", "qwen3-asr-server/run.py"]
