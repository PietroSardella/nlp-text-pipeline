# NLP Text Pipeline — Modular RAG Foundation for AI Agents

Base modular em Python para construção de Agentes de IA com suporte a RAG (Retrieval-Augmented Generation), otimizada para execução em CPU.

Este projeto estabelece uma arquitetura desacoplada entre:

- Processamento de linguagem natural (NLP)
- Geração de embeddings
- Recuperação vetorial (retrieval)
- Orquestração do pipeline

O objetivo é servir como fundação técnica para evolução futura para agentes autônomos e integração com frameworks como LangChain.

---

## 🎯 Objetivo

Construir uma base clara, organizada e escalável para:

- Processamento de texto
- Geração de embeddings com Sentence Transformers
- Busca semântica com similaridade cosseno
- Implementação de estratégias de ranking (threshold + gap)
- Evolução futura para agentes de IA

Projeto projetado para:

- Execução em CPU
- Ambientes leves
- Desenvolvimento incremental
- Arquitetura modular

---

## 🏗 Arquitetura

src/
├── core/
│ ├── loader.py
│ ├── cleaner.py
│ ├── tokenizer.py
│ └── vectorizer.py
│
├── rag/
│ ├── embedder.py
│ └── retriever.py
│
└── main.py


### 🔹 core/
Responsável por NLP base:
- carregamento de dados
- limpeza
- tokenização
- vetorização clássica

### 🔹 rag/
Responsável por:
- geração de embeddings (SentenceTransformers)
- busca semântica com cosine similarity
- controle de threshold e gap

### 🔹 main.py
Orquestra o pipeline completo.

---

## ⚙️ Tecnologias Utilizadas

- Python 3.12
- sentence-transformers
- scikit-learn
- NumPy

Modelo utilizado:
- `all-MiniLM-L6-v2` (leve e eficiente para CPU)

---

## 🚀 Como Executar

### 1. Clonar o repositório
git clone https://github.com/PietroSardella/nlp-text-pipeline.git
cd nlp-text-pipeline

### 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate # Windows

### 3. Instalar dependências
pip install -r requirements.txt

### 4. Executar como módulo
python -m src.main


---

## 🔎 Estratégia de Retrieval

O módulo `retriever.py` implementa:

- Similaridade cosseno
- Threshold mínimo de relevância
- Estratégia de gap (diferença entre primeiro e segundo score)
- Controle de Top-k

Isso evita retornos irrelevantes e melhora precisão da busca.

---

## 🧠 Próximos Passos

- Persistência de índice vetorial
- RAG híbrido (TF-IDF + embeddings)
- Debug estruturado de similaridade
- Integração com LangChain
- Evolução para agente ReAct

---

## 📌 Filosofia do Projeto

- Simplicidade antes de complexidade
- Modularidade antes de frameworks pesados
- CPU-friendly por padrão
- Entendimento do pipeline antes da abstração

---

## 📄 Licença

Projeto para fins educacionais e desenvolvimento técnico.