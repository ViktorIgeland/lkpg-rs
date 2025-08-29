## Linköping Municipality News Semantic Search

Minimal pipeline to scrape Linköping Municipality news, embed with OpenAI, index in Pinecone, and search via FastAPI.

### Setup

1. Create and activate a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and set:
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- (optional) `PINECONE_INDEX_NAME` (default: linkoping)
- (optional) `PINECONE_CLOUD` (default: aws)
- (optional) `PINECONE_REGION` (default: eu-west-1)

### Run: Scrape and Index

```bash
python -m src.pipeline
```
This will:
- Scrape `https://www.linkoping.se/nyheter/`
- Extract title, date, url, and detail page content
- Normalize/clean text and dates
- Generate embeddings with `text-embedding-3-small`
- Create Pinecone index if missing and upsert vectors
- Save raw data to `data/news.json`

### Run: FastAPI server

```bash
uvicorn src.api:app --reload
```

### Search example

POST to `/search` with a JSON body:
```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "Skolkort"}' | jq
```

Try queries like:
- "Skolkort"
- "Drottninggatan"

### Notes

- Scraper is heuristic-based to be resilient to small HTML changes.
- Dates are normalized to ISO (YYYY-MM-DD) when possible. If a date cannot be parsed, the field may be empty.

