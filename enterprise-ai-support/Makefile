# Developer convenience commands.
.PHONY: install dev-backend dev-frontend test lint compose-up compose-down seed

install:
	pip install -r requirements.txt

dev-backend:
	uvicorn app.main:app --reload --port 8000

dev-frontend:
	streamlit run frontend/streamlit_app.py

test:
	pytest

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v
