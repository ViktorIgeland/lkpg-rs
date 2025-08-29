"""FastAPI app to query Pinecone with OpenAI embeddings."""

import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "linkoping")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

app = FastAPI(title="Linköpings Kommun News Search")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 1


class SearchResult(BaseModel):
    title: str
    date: str
    url: str
    content: str
    score: float


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    emb = client.embeddings.create(model="text-embedding-3-small", input=req.query)
    vector = emb.data[0].embedding
    res = index.query(vector=vector, top_k=req.top_k, include_metadata=True)

    results: List[SearchResult] = []
    for match in res.matches or []:
        md = match.metadata or {}
        results.append(SearchResult(
            title=md.get("title", ""),
            date=md.get("date", ""),
            url=md.get("url", ""),
            content=md.get("content", ""),
            score=float(match.score) if match.score is not None else 0.0,
        ))
    return results


