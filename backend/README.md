# Research Agent Backend

FastAPI backend for source-assisted research. It accepts text and PDF files, retrieves relevant source passages, optionally searches the web, and streams a Markdown response that matches the provided frontend contract.

## API contract

- `POST /api/sources` accepts multipart uploads under the repeated `files` field.
- `POST /api/research` accepts `{ "request": "..." }` and returns a raw streamed Markdown body.

The response is intentionally not Server-Sent Events (SSE): `src/api.ts` reads UTF-8 response bytes directly and appends them to the answer panel.

## Architecture

### Source ingestion

1. Validate file type, size, and content.
2. Extract UTF-8 text from `.txt` and `.md` files or the text layer from digital PDFs.
3. Split content into bounded, structure-aware chunks of up to 800 characters with an 80-character (10%) overlap.
4. Create embeddings in-process with FastEmbed and the quantized `nomic-ai/nomic-embed-text-v1.5-Q` model.
5. Store chunk text, metadata, and vectors in a process-local cosine-similarity index.

Files in the same upload batch are embedded concurrently through the bounded worker queue. The previous corpus remains available during processing and is replaced atomically only after every file succeeds. The upload request waits for that commit, so a subsequent research request can immediately retrieve the complete new source set.

### Research requests

1. Anthropic Claude Sonnet 4.6 plans and produces the response.
2. The agent can call `search_uploads` for source retrieval and `web_search` for current external information.
3. Independent tool calls can run concurrently.
4. The eight most relevant source chunks are returned in full with filename, chunk index, and relevance metadata.
5. Only the final Markdown answer is streamed to the browser; internal tool payloads are not exposed.
6. If Anthropic fails or returns an empty answer, the same tool workflow can fall back to the local Qwen model in LM Studio when that optional service is configured.

Brave Search is called through its HTTP API. An MCP integration configured inside the LM Studio chat interface is not automatically available to an external FastAPI process.

## Design decisions

- **Retrieval instead of full-document prompting:** scales beyond short documents and keeps context focused.
- **Overlapping chunks:** reduces information loss at chunk boundaries while keeping embedding inputs bounded.
- **Full retrieved passages:** avoids truncating a relevant fact after retrieval.
- **In-memory vector index:** appropriate for the single-user take-home contract and simple to run locally.
- **In-process FastEmbed retrieval:** keeps document and query embedding self-contained and uses the model's query-specific encoding path.
- **Anthropic primary, optional local fallback:** provides a reliable hosted default while preserving an optional offline generation path.
- **Raw HTTP streaming:** exactly matches the existing frontend implementation.

### Embedding provider evolution

The initial implementation generated Nomic embeddings through LM Studio. That worked in the development environment, but it required every reviewer or deployment to install LM Studio, load a separate embedding model, configure its server and token, and keep that service running before document retrieval could work.

The final implementation runs the quantized Nomic model directly inside the backend with FastEmbed. This keeps the same retrieval architecture and comparable result quality while removing LM Studio from the required upload and search path. It also makes the application easier to run from a clean checkout or Docker environment. LM Studio remains an optional Qwen generation fallback and is no longer used for embeddings.

## Setup

Requirements:

- Python 3.12
- Node.js 22 for the frontend
- Anthropic and Brave API credentials
- Internet access on first upload so FastEmbed can download and cache its approximately 130 MB model
- Optional: LM Studio on port `1234` with Qwen loaded for fallback generation

Optionally load the local fallback model:

```sh
lms load qwen3.8-27b-uncensored-mlx
lms ps
```

Copy `backend/.env.example` to `backend/.env.local` and configure `ANTHROPIC_API_KEY` and `BRAVE_API_KEY`. Configure `LM_STUDIO_API_TOKEN` only if the Qwen fallback is wanted. LM Studio is not used for document or query embeddings.

Start the backend:

```sh
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Start the frontend in another terminal:

```sh
npm install
npm run dev
```

Open `http://localhost:5173`, upload a supported source, and submit a research question.

## Example queries and expected outputs

Exact wording can vary between model runs. These examples define the facts, source usage, and citations that a correct response must contain.

### Uploaded source only

Upload `harbor.txt` containing:

```text
Project Harbor launches on 12 March 2027. The project lead is Lina Ortiz.
```

Query:

```text
According to the uploaded source, when does Project Harbor launch? Cite the filename.
```

Expected output: the answer states **12 March 2027**, cites `harbor.txt`, and does not introduce unrelated web claims.

### External sources only

Query:

```text
Search the public web for BoWatt GmbH. Briefly explain what it does and cite the URLs used.
```

Expected output: a short current description supported by clickable Brave Search result URLs. If web search is unavailable, the response reports a user-safe failure instead of inventing sources.

### Uploaded and external sources

Upload a product brief, then query:

```text
Compare the product claims in the uploaded brief with current public information. Clearly attribute each claim to the file or a web URL.
```

Expected output: uploaded claims cite the filename, external claims cite URLs, and any disagreement is presented explicitly rather than silently merged.

## Verification

```sh
cd backend
source .venv/bin/activate
pytest
```

The tests cover:

- overlapping chunk boundaries and complete source coverage
- atomic corpus replacement and concurrent multi-file embedding
- full retrieved passage delivery without post-retrieval truncation
- cosine-similarity retrieval
- separate FastEmbed document and query encoding paths
- digital PDF ingestion and HTTP validation
- hidden-reasoning tag filtering for the local fallback
- Anthropic-first generation
- fallback after provider errors or empty output
- sanitized pre-stream and mid-stream failure handling
- streamed research responses

Recommended evaluation cases:

1. A fact available only in an uploaded file.
2. A current fact that requires web search.
3. A question requiring both uploaded and web sources.
4. Conflicting source and web claims that must remain clearly attributed.

A production evaluation suite should measure retrieval recall, answer faithfulness, citation accuracy, time to first token, and total latency against a versioned set of expected results.

## Docker

`docker compose up --build` starts the frontend and API. FastEmbed downloads its model into the backend container on the first upload. If the optional Qwen fallback is enabled, LM Studio remains on the host and uses `LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1`. Credentials must be supplied at runtime and must not be baked into an image.

## Current scope and production evolution

The implementation is a professional single-user prototype, not a complete multi-tenant deployment. Current limitations are explicit:

- source data is process-local and is cleared on restart
- uploads are shared by all callers because the provided contract has no user or session identifier
- scanned PDFs require OCR and are rejected when no text layer is present
- vector search is exact and has no lexical-search or reranking stage
- observability, rate limiting, durable job processing, and provider health metrics are not included

For a multi-user production service, add authenticated workspaces, object storage for original files, a persistent vector database such as pgvector or Qdrant, background ingestion jobs, hybrid retrieval with reranking, OCR for scanned documents, structured citations, request tracing, quotas, and automated evaluation gates.
