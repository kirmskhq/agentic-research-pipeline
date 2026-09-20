# One-shot container for the research pipeline.
# The script is bind-mounted at /app, so editing it needs no rebuild.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir     "openai==2.38.0"     "ddgs==9.14.4"

CMD ["python", "-c", "import openai, ddgs; print('image ok, openai', openai.__version__)"]
