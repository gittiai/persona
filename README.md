# AI Persona — Aditya

RAG-grounded chat + voice agent over my resume and GitHub repos.

## Stack

- **Backend:** FastAPI (single file, `api/index.py`)
- **LLM:** Groq Llama 3.3 70B (`llama-3.3-70b-versatile`)
- **Voice infra:** Vapi (phone, STT, TTS, barge-in) → calls `/vapi/llm` on this backend
- **Calendar:** Cal.com link returned by the LLM when the user asks to book
- **Hosting:** Vercel (serverless Python)
- **Grounding:** resume PDF + GitHub READMEs loaded once at module init, stuffed into the system prompt (no vector DB — corpus is small)

## Architecture

```
            ┌───────────────────────────────────────────────────┐
            │                                                     │
  User chat ──► / (Jinja) ──► POST /chat ──►                      │
                                            ├─► Groq Llama 3.3   │
  Phone call ──► Vapi (STT/TTS) ──► /vapi/llm ──►                │
                                                                  │
            │                ▲                                     │
            │                │                                     │
            │       Corpus (resume + READMEs)                      │
            │                │                                     │
            │   data/resume.pdf  +  GitHub API (READMEs)           │
            └───────────────────────────────────────────────────┘
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/health` | Configuration + corpus status |
| GET | `/corpus` | Preview of what the LLM has access to |
| POST | `/chat` | `{message, history}` → `{reply}` |
| POST | `/vapi/llm` | OpenAI-compatible chat completions for Vapi's "Custom LLM" |

## Setup (local)

```bash
cd persona
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
# drop resume PDF at: data/resume.pdf
uvicorn api.index:app --reload
```

Visit `http://localhost:8000`.

## Deploy (Vercel)

```bash
vercel link
vercel env add GROQ_API_KEY production
vercel env add GITHUB_USERNAME production
vercel env add CAL_BOOKING_URL production
# (add the rest from .env.example)
vercel --prod
```

The resume PDF lives in `data/` and is bundled with the deploy. GitHub READMEs are fetched at cold-start.

## Voice (Vapi setup)

1. Create a Vapi assistant
2. Model → **Custom LLM**
3. URL → `https://<your-vercel-url>/vapi/llm`
4. Transcriber → Deepgram or Groq Whisper
5. Voice → any (Vapi defaults are fine)
6. Buy / attach a phone number to the assistant

## Costs (per session)

| Component | Per chat (~10 turns) | Per voice call (~3 min) |
|---|---|---|
| Groq inference | ~$0 (free tier) | ~$0 |
| Vapi voice infra | n/a | ~$0.30 |
| Vercel | $0 (free tier) | $0 |
| Embeddings | $0 (none used) | $0 |
| **Total** | **~$0** | **~$0.30** |

## Design tradeoff

**No vector DB.** Stuffing the full resume + 10 repo READMEs (~40KB) into Llama 3.3 70B's 128K-token context window is simpler and **more reliable** than chunked retrieval for a corpus this size. No risk of the retriever missing the right chunk. Tradeoff: latency cost of prompt-processing the full corpus on every request (~200ms on Groq). Acceptable.

If the corpus grew past ~50 repos / 200KB, I'd switch to pgvector + retrieval.

## What I'd build with 2 more weeks

- Real-time calendar booking via Cal.com API function call (not just a link)
- Streaming responses to chat (currently buffered)
- Vapi function-call for "transfer to Aditya" if the caller is a recruiter who wants to skip the AI
- Eval harness that re-runs the golden Q&A set on every deploy via GitHub Actions
- Voice-tuned system prompt that handles barge-in more elegantly (currently relies on Vapi's defaults)
