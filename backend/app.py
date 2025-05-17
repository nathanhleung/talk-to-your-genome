import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

import anthropic
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SNPediaRequest(BaseModel):
    question: str
    snps: List[str]


class ChatResponse(BaseModel):
    text: str
    sources: List[str]


@app.post("/snpedia", response_model=ChatResponse)
async def search_snpedia(request: SNPediaRequest):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    SYSTEM_PROMPT = """
    You are a helpful assistant that provides information about SNPs and their effects on health. 
    You will be given a list of SNPs the user has. They will ask questions and you should provide personalized health advice based on the SNPs provided.
    You should also provide sources where the user can get more information if needed. Omit any SNPs that are not relevant to the specific user question.
    Use simple language with direct answers.
    Responses should be a maximum of one paragraph.
    """

    message = client.beta.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here are my SNPs: {', '.join(request.snps)}. What can you tell me about them?",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is my question: {request.question}",
                    }
                ],
            },
        ],
        tools=[
            {
                "name": "web_search",
                "type": "web_search_20250305",
                "allowed_domains": ["snpedia.com"],
            }
        ],
        betas=["web-search-2025-03-05"],
    )

    text, sources = extract_text_with_citations(message.content)

    return ChatResponse(text=text, sources=sources)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def extract_text_with_citations(api_response):
    user_texts = []
    citations = []

    for block in api_response:
        block = dict(block)
        if block["type"] == "text":
            text = block.get("text", "").strip()
            if text:
                user_texts.append(text)
        if block.get("citations"):
            for cite in block["citations"]:
                cite = dict(cite)
                url = cite.get("url")
                title = cite.get("title")
                if url and title:
                    citations.append(f"{title}: {url}")

    full_text = "\n\n".join(user_texts)
    return full_text, citations


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
