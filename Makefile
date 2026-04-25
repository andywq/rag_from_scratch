PYTHON ?= python3

.PHONY: install lint build unittest test

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install ruff

lint:
	$(PYTHON) -m ruff check .

build:
	$(PYTHON) -m py_compile app/config.py app/entrypoint.py scripts/import_from_mongodb.py scripts/check_providers.py
	$(PYTHON) -m py_compile app/models/provider_clients.py app/models/provider_factory.py app/models/embedding_client.py app/models/llm_client.py
	$(PYTHON) -m py_compile app/vector_store/faiss_store.py app/vector_store/metadata_store.py app/rag/service.py
	$(PYTHON) -m py_compile app/web_service/app.py app/web_service/handlers.py app/web_service/schemas.py

unittest:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

test: lint build unittest
