FROM python:3.11

WORKDIR /app

COPY . /app/

RUN mkdir -p /app/hf_cache /app/models
RUN chmod -R 777 /app/hf_cache /app/models

ENV HF_HOME=/app/hf_cache

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
