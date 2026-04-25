FROM python:3.13-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt ruff

COPY . /workspace

CMD ["sh", "-c", "python -m ruff check . && python -m py_compile app/config.py app/entrypoint.py scripts/import_from_mongodb.py scripts/check_providers.py app/models/provider_clients.py app/models/provider_factory.py app/models/embedding_client.py app/models/llm_client.py app/vector_store/faiss_store.py app/vector_store/metadata_store.py app/rag/service.py app/web_service/app.py app/web_service/handlers.py app/web_service/schemas.py && python -m unittest discover -s tests -p 'test_*.py' -v"]
