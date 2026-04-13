# LLM Inference API

A production-ready REST API for self-hosted large language model inference, built with FastAPI and HuggingFace Transformers. Designed to sit behind a downstream proxy gateway with shared secret authentication.

## Overview

This project implements a clean, modular inference pipeline around Mistral 7B — a state-of-the-art open source language model. The API handles tokenization, 4-bit quantized GPU inference, and detokenization, returning natural language responses over a simple HTTP interface.

Built with extensibility in mind: per-user context, conversation history, and multi-model support are all natural next steps without restructuring the codebase.

## Architecture

```
downstream proxy
      │
      ▼
X-Internal-Gateway-Auth header verification  (middleware)
      │
      ▼
POST /chat  (FastAPI route)
      │
      ▼
inference queue  (asyncio lock — serialized GPU access)
      │
      ▼
tokenize → generate → detokenize  (Mistral 7B 4-bit quantized)
      │
      ▼
JSON response
```

## Tech Stack

- **FastAPI** — async HTTP framework with automatic request validation
- **HuggingFace Transformers** — model loading and inference pipeline
- **Mistral 7B Instruct v0.2** — instruction-tuned LLM for conversational responses
- **bitsandbytes** — 4-bit quantization, reducing VRAM requirement from ~14GB to ~4GB
- **Docker + CUDA 12.4** — containerized GPU inference
- **Pydantic Settings** — typed configuration from environment variables

## Features

- **Gateway authentication** — shared secret header verification on every request
- **Serialized inference** — asyncio mutex prevents concurrent GPU access and OOM errors
- **4-bit quantization** — Mistral 7B runs on consumer GPUs with as little as 4GB VRAM
- **Persistent model cache** — Docker volume prevents re-downloading 14GB of weights on every container start
- **Inference timeout** — requests that exceed 300 seconds return a 504 instead of hanging
- **Input validation** — messages over 4000 characters are rejected before touching the model
- **Configurable** — model, temperature, top_p, and token limits all controlled via environment variables

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with 4GB+ VRAM
- HuggingFace account with access to Mistral 7B

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/llm-inference-api
cd llm-inference-api
```

**2. Create your `.env` file**
```bash
cp .env.example .env
```

Fill in your values:
```
GATEWAY_SECRET=your_shared_secret
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
HUGGINGFACE_TOKEN=hf_your_token_here
```

**3. Build and run**
```bash
docker compose up --build
```

First run will download Mistral 7B (~14GB) into a Docker volume. Subsequent starts load from the volume instantly.

## Usage

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-Internal-Gateway-Auth: your_shared_secret" \
  -H "Content-Type: application/json" \
  -d '{"message": "explain what a transformer model is"}'
```

**Response:**
```json
{
  "response": "A transformer model is a type of neural network architecture..."
}
```

**PowerShell:**
```powershell
$response = Invoke-RestMethod -Method POST "http://localhost:8000/chat" `
  -Headers @{
    "X-Internal-Gateway-Auth" = "your_shared_secret"
    "Content-Type" = "application/json"
  } `
  -Body '{"message": "explain what a transformer model is"}'

$response.response
```

## Error Responses

| Status | Reason |
|--------|--------|
| 401 | Missing or invalid `X-Internal-Gateway-Auth` header |
| 422 | Request body missing `message` field or message exceeds 4000 characters |
| 504 | Inference exceeded 300 second timeout |
| 500 | Internal model error |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_SECRET` | required | Shared secret for gateway auth |
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model identifier |
| `MAX_NEW_TOKENS` | `512` | Maximum tokens to generate per response |
| `TEMPERATURE` | `0.7` | Sampling temperature (higher = more random) |
| `TOP_P` | `0.9` | Nucleus sampling threshold |
| `HUGGINGFACE_TOKEN` | required | HuggingFace access token for gated models |

## Project Structure

```
├── main.py               # App entrypoint, middleware and router registration
├── config.py             # Typed settings from environment variables
├── middleware/
│   └── auth.py           # Gateway shared secret verification
├── routes/
│   └── chat.py           # POST /chat handler
├── core/
│   └── inference.py      # Tokenization, generation, decoding
├── docker-compose.yml    # Service definition with GPU passthrough and volume mount
├── Dockerfile            # CUDA 12.4 + Python 3.11 + dependencies
├── requirements.txt      # Pinned dependencies with PyTorch CUDA index
└── .env.example          # Environment variable template
```

## Planned

- Per-user conversation history and context via database
- User identity injection from gateway (`X-User-Email` header)
- Streaming responses
- Support for swappable model backends