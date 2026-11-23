FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY label_lookup ./label_lookup
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-m", "label_lookup"]
