# SDKWork-TTS Makefile
# Simplifies common operations for development and deployment

.PHONY: help build test run clean docker docker-run install release

# Default target
.DEFAULT_GOAL := help

# Variables
APP_NAME := sdkwork-tts
VERSION := $(shell grep '^version =' Cargo.toml | cut -d'"' -f2)
DOCKER_IMAGE := sdkwork-tts
DOCKER_TAG := latest

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Help target
help: ## Display this help message
	@echo "$(BLUE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║           SDKWork-TTS Makefile Help                       ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC) make [target]"
	@echo ""
	@echo "$(YELLOW)Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Build targets
build: ## Build in release mode
	@echo "$(YELLOW)Building $(APP_NAME) in release mode...$(NC)"
	cargo build --release --no-default-features --features cpu

build-gpu: ## Build with CUDA support
	@echo "$(YELLOW)Building $(APP_NAME) with CUDA support...$(NC)"
	CUDA_COMPUTE_CAP=89 cargo build --release --features cuda

build-debug: ## Build in debug mode
	@echo "$(YELLOW)Building $(APP_NAME) in debug mode...$(NC)"
	cargo build

# Test targets
test: ## Run tests
	@echo "$(YELLOW)Running tests...$(NC)"
	cargo test --lib --no-default-features --features cpu

test-all: ## Run all tests including integration tests
	@echo "$(YELLOW)Running all tests...$(NC)"
	cargo test --no-default-features --features cpu

check: ## Run clippy and format check
	@echo "$(YELLOW)Running clippy...$(NC)"
	cargo clippy --no-default-features --features cpu -- -D warnings
	@echo "$(YELLOW)Checking format...$(NC)"
	cargo fmt -- --check

# Run targets
run: ## Run server in local mode
	@echo "$(YELLOW)Starting server in local mode...$(NC)"
	cargo run --release --no-default-features --features cpu -- server --mode local

run-cloud: ## Run server in cloud mode
	@echo "$(YELLOW)Starting server in cloud mode...$(NC)"
	cargo run --release --no-default-features --features cpu -- server --mode cloud

run-hybrid: ## Run server in hybrid mode
	@echo "$(YELLOW)Starting server in hybrid mode...$(NC)"
	cargo run --release --no-default-features --features cpu -- server --mode hybrid

# Docker targets
docker: ## Build Docker image (CPU)
	@echo "$(YELLOW)Building Docker image (CPU)...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu --target runtime .

docker-gpu: ## Build Docker image (GPU)
	@echo "$(YELLOW)Building Docker image (GPU)...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu -f Dockerfile.gpu .

docker-run: ## Run Docker container
	@echo "$(YELLOW)Running Docker container...$(NC)"
	docker run -d -p 8080:8080 \
		-v $(PWD)/checkpoints:/app/checkpoints:ro \
		-v $(PWD)/speaker_library:/app/speaker_library \
		--name $(APP_NAME) \
		$(DOCKER_IMAGE):$(DOCKER_TAG)-cpu

docker-run-gpu: ## Run Docker container with GPU
	@echo "$(YELLOW)Running Docker container with GPU...$(NC)"
	docker run -d -p 8080:8080 \
		--gpus all \
		-v $(PWD)/checkpoints:/app/checkpoints:ro \
		-v $(PWD)/speaker_library:/app/speaker_library \
		--name $(APP_NAME)-gpu \
		$(DOCKER_IMAGE):$(DOCKER_TAG)-gpu

docker-compose-up: ## Start with docker-compose
	@echo "$(YELLOW)Starting docker-compose...$(NC)"
	docker compose --profile cpu up -d

docker-compose-down: ## Stop docker-compose
	@echo "$(YELLOW)Stopping docker-compose...$(NC)"
	docker compose down

docker-logs: ## View Docker logs
	docker logs -f $(APP_NAME)

docker-stop: ## Stop Docker container
	@echo "$(YELLOW)Stopping Docker container...$(NC)"
	docker stop $(APP_NAME)
	docker rm $(APP_NAME)

# Installation targets
install: ## Install to local system
	@echo "$(YELLOW)Installing $(APP_NAME)...$(NC)"
	./scripts/install.sh

install-windows: ## Install to Windows system
	@echo "$(YELLOW)Installing $(APP_NAME) on Windows...$(NC)"
	powershell -ExecutionPolicy Bypass -File scripts/install.ps1

uninstall: ## Uninstall from local system
	@echo "$(YELLOW)Uninstalling $(APP_NAME)...$(NC)"
	rm -rf ~/.sdkwork-tts
	@echo "$(GREEN)✓ Uninstallation complete$(NC)"

# Release targets
release: ## Create a release build
	@echo "$(YELLOW)Creating release build v$(VERSION)...$(NC)"
	cargo build --release --no-default-features --features cpu
	@echo "$(GREEN)✓ Release build complete$(NC)"

release-artifacts: ## Create release artifacts
	@echo "$(YELLOW)Creating release artifacts...$(NC)"
	mkdir -p release
	cp target/release/$(APP_NAME) release/$(APP_NAME)-linux
	@echo "$(GREEN)✓ Release artifacts created$(NC)"

# Utility targets
clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	cargo clean
	rm -rf release/

clean-all: ## Clean all including Docker
	@echo "$(YELLOW)Cleaning all artifacts...$(NC)"
	cargo clean
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu 2>/dev/null || true
	rm -rf release/
	@echo "$(GREEN)✓ Clean complete$(NC)"

logs: ## View server logs
	tail -f logs/server.log

shell: ## Open shell in development container
	docker run -it --rm \
		-v $(PWD):/app \
		-w /app \
		rust:1.75-slim-bookworm \
		bash

version: ## Show version information
	@echo "$(BLUE)$(APP_NAME) version $(VERSION)$(NC)"
	cargo --version
	rustc --version

# Development targets
dev: ## Start development server with hot reload
	@echo "$(YELLOW)Starting development server...$(NC)"
	cargo watch -x 'run -- server --mode local'

watch: ## Watch for changes and rebuild
	@echo "$(YELLOW)Watching for changes...$(NC)"
	cargo watch -x 'build --release'

# Benchmark targets
bench: ## Run benchmarks
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	cargo bench

# Documentation targets
doc: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	cargo doc --no-deps --open

doc-check: ## Check documentation
	@echo "$(YELLOW)Checking documentation...$(NC)"
	cargo doc --no-deps

# Format targets
fmt: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	cargo fmt

fix: ## Automatically fix linting issues
	@echo "$(YELLOW)Fixing linting issues...$(NC)"
	cargo clippy --fix --no-default-features --features cpu --allow-dirty

# Security targets
audit: ## Run security audit
	@echo "$(YELLOW)Running security audit...$(NC)"
	cargo audit

# CI targets
ci: check test ## Run CI checks
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

ci-full: check test-all ## Run full CI checks
	@echo "$(GREEN)✓ All CI checks passed$(NC)"
