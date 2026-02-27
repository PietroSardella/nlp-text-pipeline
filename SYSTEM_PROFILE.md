# SYSTEM PROFILE – Mini RAG Project

## Hardware

Model: Samsung 550XED  
CPU: Intel 12th Gen (~1.3 GHz base)  
RAM: 12 GB  
Disk: 256 GB SSD  

## OS

Windows 11 x64

## Python

Python 3.12

## Constraints

- CPU only
- No GPU
- Moderate RAM (12 GB)
- Prefer lightweight models (<150MB)
- Avoid heavy local LLMs

## Strategic Decisions

- Use MiniLM embeddings (384 dims)
- Limit top-k to 3–5
- Use small chunk sizes (300–500 tokens)
- Prefer API-based LLM instead of local heavy model
