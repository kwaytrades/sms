# Makefile - Place in root directory (optional but helpful)

.PHONY: help install dev test lint format clean docker-up docker-down deploy

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean cache and temp files"
	@echo "  docker-up   - Start Docker services"
	@echo "  docker-down - Stop Docker services"
	@echo "  deploy      - Deploy to production"

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install pytest pytest-asyncio black flake8 mypy

# Run development server
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run with Docker
docker-up:
	docker-compose up -d

# Stop Docker services
docker-down:
	docker-compose down

# View Docker logs
docker-logs:
	docker-compose logs -f app

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	black . --line-length=127
	isort . --profile black

# Type checking
typecheck:
	mypy . --ignore-missing-imports

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".coverage" -delete
	find . -name "*.cover" -delete
	find . -name "coverage.xml" -delete
	find . -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Setup development environment
setup-dev: install-dev
	cp .env.example .env
	@echo "✅ Development environment setup complete!"
	@echo "📝 Edit .env file with your API keys"
	@echo "🚀 Run 'make docker-up' to start local services"
	@echo "💻 Run 'make dev' to start the development server"

# Health check
health:
	curl -s http://localhost:8000/health | python -m json.tool

# Deploy to production (requires RENDER_API_KEY)
deploy:
	@echo "🚀 Deploying to production..."
	git push origin main
	@echo "✅ Deployment initiated via GitHub Actions"

# Generate requirements.txt from environment
freeze:
	pip freeze > requirements.txt

# Database commands
db-reset:
	docker-compose exec mongo mongo sms_trading_bot --eval "db.dropDatabase()"
	@echo "🗑️ Database reset complete"

# Cache commands  
cache-clear:
	docker-compose exec redis redis-cli FLUSHALL
	@echo "🗑️ Cache cleared"

# View logs
logs:
	tail -f logs/*.log

# Backup data
backup:
	@echo "📦 Creating backup..."
	docker-compose exec mongo mongodump --db sms_trading_bot --out /tmp/backup
	docker-compose cp mongo:/tmp/backup ./backup/$(shell date +%Y%m%d_%H%M%S)
	@echo "✅ Backup created in ./backup/"
