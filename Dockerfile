# TTS Server Docker Image
# Multi-stage build for optimized size

# ============================================
# Stage 1: Builder
# ============================================
FROM rust:1.75-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source files
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build in release mode (CPU version)
RUN cargo build --release --no-default-features --features cpu

# Build CUDA version (optional, uncomment if needed)
# ENV CUDA_COMPUTE_CAP=89
# RUN cargo build --release --features cuda

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r tts && useradd -r -g tts tts

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/sdkwork-tts /usr/local/bin/

# Create directories for data
RUN mkdir -p /app/checkpoints /app/speaker_library /app/config && \
    chown -R tts:tts /app

# Switch to non-root user
USER tts

# Expose port
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info
ENV MODE=local
ENV PORT=8080
ENV HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entry point
ENTRYPOINT ["sdkwork-tts", "server"]

# Default arguments
CMD ["--mode", "local", "--port", "8080"]
