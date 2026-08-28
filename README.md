# Research Agent Frontend

React and TypeScript interface for a source-assisted research service, bootstrapped with Vite.

The app lets a user:

- enter a research request
- upload source files
- view a streamed markdown response

## Setup

Install dependencies:

```sh
npm install
```

Run the frontend locally:

```sh
npm run dev
```

Vite will print the local URL, usually `http://localhost:5173/`.

## Backend API

By default, the frontend calls:

```txt
http://localhost:8787
```

Set a different backend URL with:

```sh
VITE_API_BASE_URL=http://localhost:8787 npm run dev
```

Expected endpoints:

- `POST /api/research` — accepts JSON `{ "request": "..." }` and returns a streamed markdown response.
- `POST /api/sources` — accepts multipart form uploads under the repeated field name `files`.

## Backend

The FastAPI server lives in `backend/`. It listens on `http://localhost:8787` and uses Anthropic Claude Sonnet 4.6 for primary generation, LM Studio for local embeddings and fallback generation, and Brave Search for web results.

See [backend/README.md](backend/README.md) for setup, architecture, verification, and production considerations.
