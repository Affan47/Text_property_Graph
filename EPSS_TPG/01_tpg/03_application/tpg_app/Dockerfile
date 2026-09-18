# TPG Platform — containerised deployment
# Build (from the EPSS_TPG project root):
#   docker build -f 01_tpg/03_application/tpg_app/Dockerfile -t tpg-platform .
# Run:
#   docker run -p 8742:8742 -v tpg_data:/data tpg-platform
FROM python:3.12-slim

WORKDIR /app

COPY 01_tpg/03_application/tpg_app/requirements.txt /app/tpg_app/requirements.txt
RUN pip install --no-cache-dir -r /app/tpg_app/requirements.txt \
    && python -m spacy download en_core_web_sm

COPY .tpg-project-root /app/.tpg-project-root
COPY 01_tpg/pyproject.toml /app/01_tpg/pyproject.toml
COPY 01_tpg/01_core/tpg /app/01_tpg/01_core/tpg
COPY 01_tpg/03_application/tpg_app /app/01_tpg/03_application/tpg_app
COPY 01_tpg/04_chatbot/tpg_chatbot /app/01_tpg/04_chatbot/tpg_chatbot
RUN pip install --no-deps -e /app/01_tpg

ENV TPG_DB=/data/tpg_workspace.db \
    TPG_UPLOADS=/data/uploads \
    TPG_HOST=0.0.0.0 \
    TPG_PORT=8742 \
    TPG_DOMAIN=general

VOLUME /data
EXPOSE 8742

CMD ["python", "-m", "tpg_app.server"]
