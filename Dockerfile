# Build Stage
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/oopllama
COPY . .

# Build with CUDA support
RUN cargo build --release

# Runtime Stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from:builder /usr/src/oopllama/target/release/oopllama .
COPY --from:builder /usr/src/oopllama/models ./models

# Set runtime environment
ENV RUST_LOG=info
# Disabled auto-launch to free up GPU for Nova Prime training
CMD ["tail", "-f", "/dev/null"]
