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

## Svenska

### Problem & målbild

Bygg en MVP för en pipeline som hämtar nyheter från Linköpings kommun, skapar vektorinbäddningar och gör det sökbara via ett API, så att användare snabbt kan hitta relevanta nyheter utifrån innebörd och inte bara ett nyckelord.

### Kort beskrivning av flödet

- Hämtar nyheter från `https://www.linkoping.se/nyheter/`
- Rensar och normaliserar text och datum
- Skapar embeddings med OpenAI
- Lagrar vektorer i Pinecone
- Exponerar sök via FastAPI: `/search` tar en text/sökord, embed:ar den och matchar i Pinecone

### Val av verktyg/databas – och varför

- Cursor: snabbt sätt att komma igång och bygga en prototyp
- BeautifulSoup: enkel, beprövad HTML-parsning för skrapning
- OpenAI Embeddings: hög kvalitet på API:et och enkelt att använda då jag har tidigare kunskap av det
- Pinecone (vektordatabas): skalbar och snabb att komma igång med, eftersom det är en vektordatabas kan man enkelt göra "likhetssökning" och hitta relevant info
- FastAPI + Uvicorn: snabbt och enkelt att bygga ett litet, responsivt API

### Kör-/testinstruktioner
- Se instruktioner ovan

### Reflektion: nästa steg och skalning
Just nu är det en väldigt enkel och minimal produkt, men några idéer: 
- Schemalägg skrapning så att det sker automatiskt med jämna mellanrum, utan behov att göra det manuellt. Även lägga till mer felhantering, t.ex. för att undvika kopior eller om något går fel under skrapning etc.
- Lägg till filtrering och kategorisering, t.ex. analysera nyheten med chatGPT innan den laddas upp till Pinecone och då märk den med olika kategorier och filter.
- Mer specific databehandling etc.
- Bygg ett frontend som gör det enkelt att använda.

Eftersom vi kör Pinecone så skalar det väldigt bra och ju mer artiklar vi får desto bättre och mer relevant information kan man hitta via sökning.

### Vart användes AI?
- AI användes först för att planera och ta fram en strategi på hur projektet kunde genomföras (chatGPT)
- Denna planering användes för att skapa en prompt, som sedan gavs till Cursor för att bygga själva MVP:n.
- Buggfixar och setup av t.ex. databas etc gjordes manuellt.
