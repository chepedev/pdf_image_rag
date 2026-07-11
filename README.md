# KTM-RAG — A Multimodal RAG Chatbot for Your Documents

> Drop PDFs and images into a folder, and chat with them. Every answer comes back
> with the **exact source**: the file name, the page number, and a **picture of that
> page** so you can verify it with your own eyes.

This README is written for someone who **started in AI today**. It explains every
concept, every file, every command, and every error we already hit — top to bottom.
Read it once end-to-end; then use the Table of Contents to jump back.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Concepts you need first (the 10-minute AI primer)](#2-concepts-you-need-first-the-10-minute-ai-primer)
3. [How the whole thing fits together (architecture)](#3-how-the-whole-thing-fits-together-architecture)
4. [The tech stack — what each piece is and why](#4-the-tech-stack--what-each-piece-is-and-why)
5. [Project structure — file by file](#5-project-structure--file-by-file)
6. [Prerequisites](#6-prerequisites)
7. [Setup, step by step](#7-setup-step-by-step)
8. [Running it](#8-running-it)
9. [Using it (the day-to-day loop)](#9-using-it-the-day-to-day-loop)
10. [How it works internally (deep dive)](#10-how-it-works-internally-deep-dive)
11. [Configuration reference (`.env`)](#11-configuration-reference-env)
12. [Persistence — do I re-index every run?](#12-persistence--do-i-re-index-every-run)
13. [Cost & quotas](#13-cost--quotas)
14. [Troubleshooting (real errors we hit)](#14-troubleshooting-real-errors-we-hit)
15. [Glossary](#15-glossary)
16. [Where to go next](#16-where-to-go-next)

---

## 1. What is this project?

It's a **chatbot that only knows about your documents**. You put files (PDF manuals,
scanned images, text notes) into a folder called `corpus/`. The system reads them,
"memorizes" them in a special database, and then lets you ask questions in plain
language. When it answers, it shows you **where the answer came from** — including a
screenshot of the source page.

The name comes from the example use case: a **KTM motorcycle user manual** (lots of
diagrams and part numbers), where "just the text" isn't enough — you want to *see*
the page with the exploded diagram.

**Why is this useful?** A normal chatbot (like a raw LLM) makes things up when it
doesn't know — this is called *hallucination*. By forcing the bot to answer **only
from your documents** and to **cite the page**, you get answers you can trust and
verify. That technique has a name: **RAG**.

---

## 2. Concepts you need first (the 10-minute AI primer)

If these terms are new, read this section slowly. Everything else depends on it.

### LLM (Large Language Model)
A model like Google's **Gemini** (or Claude, GPT, etc.) that takes text (and, for
"multimodal" ones, images) and generates text back. It's very smart but it only knows
what it was trained on, and it will confidently invent answers for things it doesn't
know. We use an LLM here **only to write the final answer** from context we give it.

### Embedding
The core trick of this whole system. An **embedding** is a list of numbers (a
*vector*) that represents the *meaning* of a piece of content.

- The sentence *"How do I change the oil?"* becomes something like
  `[0.021, -0.874, 0.335, ... ]` (in our case, **3072 numbers long**).
- Two pieces of content with **similar meaning** get **similar vectors** — they land
  close together in this 3072-dimensional space.
- *"How do I change the oil?"* and *"Oil change procedure"* end up near each other,
  even though they share almost no words.

This is what lets us search by **meaning**, not by keywords. The model that turns
content into vectors is called an **embedding model**. We use Google's
**`gemini-embedding-2`**.

### Multimodal embedding (why this project is special)
Most embedding models only understand **text**. `gemini-embedding-2` is **multimodal**:
it can turn **text AND images** into vectors *in the same space*. So a photo of a
motorcycle chain and the words "drive chain tension" can land near each other. This is
exactly what we need to retrieve the right *page image* for a question.

### Vector database (Qdrant)
Once every chunk of your documents is an embedding, you need somewhere to store
millions of these vectors and, given a new vector (your question), instantly find the
**closest** ones. That's a **vector database**. We use **Qdrant**. The operation
"find the nearest vectors" is called a **similarity search** (we use *cosine
similarity* — think of it as "how close are these two arrows pointing?").

### Chunking
You don't embed a whole 200-page PDF as one vector — the meaning gets blurry and you
can't point to a specific page. Instead you split it into small pieces (**chunks**).
Here, **one chunk = one PDF page** (text + a picture of that page). Each chunk gets
its own vector and its own citation.

### RAG (Retrieval-Augmented Generation)
The pattern that ties it all together. Three steps every time you ask a question:

1. **Retrieve** — turn your question into a vector, search Qdrant, get the top few
   most-relevant chunks.
2. **Augment** — paste those chunks (their text + page images) into a prompt.
3. **Generate** — ask the LLM to answer the question *using only that context*, and to
   cite sources.

"Augmented" because we *augment* the LLM's prompt with your private knowledge that it
was never trained on. That's how we stop hallucinations and get citations.

### top-k
When we search, we don't want *every* match — we want the **k** best ones (e.g.
top 5). That number is `top_k`. Higher = more context (better recall, more tokens,
slower, pricier). Lower = tighter, cheaper, faster.

---

## 3. How the whole thing fits together (architecture)

There are two flows: **ingestion** (getting documents into the brain) and **chat**
(asking questions). Here's the whole picture:

```
                      ┌──────────────────────────────────────────────┐
                      │                YOU (browser)                  │
                      │           http://localhost:8501               │
                      └───────────────┬───────────────▲──────────────┘
                                      │ question       │ answer + source images
                                      ▼                │
          ┌───────────────────────────────────────────────────────────┐
          │                    STREAMLIT APP (app.py)                  │
          │                                                            │
          │   INGESTION (auto, every 3s)        CHAT (on each question)│
          │   ┌───────────────────────┐         ┌─────────────────────┐│
          │   │ 1. watch corpus/ folder│        │ 1. embed question    ││
          │   │ 2. PDF→page images+text│        │ 2. search Qdrant     ││
          │   │ 3. embed (text+image)  │        │ 3. build prompt w/   ││
          │   │ 4. store in Qdrant     │        │    top-k chunks+imgs ││
          │   └───────────┬───────────┘         │ 4. LLM writes answer ││
          │               │                     └─────────┬───────────┘│
          └───────────────┼───────────────────────────────┼───────────┘
                          │                               │
          embeddings +    │                        query  │ + generate
          images to embed │                        vector │
                          ▼                               ▼
   ┌──────────────────────────────┐        ┌──────────────────────────────┐
   │   GOOGLE GEMINI API (cloud)   │        │        QDRANT (Docker)        │
   │  • gemini-embedding-2 (vecs)  │        │  vector DB, stores 3072-dim   │
   │  • gemini-3.5-flash (answers) │        │  vectors + payload (metadata) │
   └──────────────────────────────┘        └───────────────┬──────────────┘
                                                            │ persists to disk
                                                            ▼
                                                   ./qdrant_storage/  (host)

   Your files live on your machine:  ./corpus/ (input)  ./storage/ (rendered pages)
```

**Plain-English version:** The Streamlit app is the brain-stem. It sends your content
to Google to get vectors, stores those vectors in Qdrant, and when you ask something it
searches Qdrant and asks Google to write the answer. Everything except the two Google
API calls runs on your own machine.

---

## 4. The tech stack — what each piece is and why

| Piece | What it is | Why we use it |
|---|---|---|
| **Python 3.12** | Programming language | Everything is written in it |
| **Streamlit** | Turns a Python script into a web app | Gives us a chat UI + a page to show source images, with almost no frontend code |
| **Google Gemini API** | Cloud AI service | Provides the embedding model and the answer-writing model |
| **`gemini-embedding-2`** | Multimodal embedding model | Turns text *and images* into 3072-number vectors in one shared space |
| **`gemini-3.5-flash`** | Fast multimodal LLM | Reads the retrieved pages (text + images) and writes the answer |
| **Qdrant** | Vector database | Stores the vectors and does the "find nearest" search |
| **PyMuPDF (`fitz`)** | PDF library | Renders each PDF page to an image and extracts its text |
| **Pillow (`PIL`)** | Image library | Handles image files |
| **Docker / Docker Compose** | Containers | Packages the app + Qdrant so it runs the same everywhere, with one command |

---

## 5. Project structure — file by file

```
ktmrag/
├── app.py                 ← THE WHOLE APP: ingestion + chat + web UI (one file)
├── requirements.txt       ← Python libraries to install
├── .env.example           ← Template for your secrets/config (copy to .env)
├── .env                   ← YOUR real secrets/config  (you create this; never commit)
├── Dockerfile             ← Recipe to build the app's container image
├── docker-compose.yml     ← Defines the 2 services (app + qdrant) and how they connect
├── .dockerignore          ← Files Docker should NOT copy into the image
├── README.md              ← This file
│
├── corpus/                ← 📥 YOU DROP DOCUMENTS HERE (PDF, PNG, JPG, TXT, MD)
├── storage/               ← 🤖 auto-generated: rendered page images + manifest.json
│   ├── pages/             ←    one PNG per PDF page (shown as the source image)
│   └── manifest.json      ←    "which files are already indexed" (skip-list)
└── qdrant_storage/        ← 🤖 Qdrant's data on disk (your vectors live here)
```

The only files **you** ever touch are `.env` (once) and whatever you drop into
`corpus/`. Everything under `storage/` and `qdrant_storage/` is managed automatically.

### What's inside `app.py` (the map)

`app.py` is one file, organized top-to-bottom in these sections:

1. **Config** — reads settings from `.env` (model names, dimensions, folders).
2. **`get_clients()`** — connects to Gemini + Qdrant, and creates the Qdrant
   "collection" (like a table) the first time.
3. **`embed()`** — sends content to Gemini and returns one vector. Wrapped in
   `_with_retry()` so a temporary Google outage doesn't crash us.
4. **Ingestion** — `points_for_file()` turns one file into vectors; `index_corpus()`
   loops over the folder; `corpus_signature()` cheaply detects changes.
5. **Retrieval + answer** — `retrieve()` searches Qdrant; `answer()` asks the LLM.
6. **UI** — the sidebar with the auto-watcher, and the chat area that renders answers
   plus the source images.

---

## 6. Prerequisites

You need three things on your Mac:

1. **Docker Desktop** — installed and **running** (the whale icon in the menu bar).
   This is the engine that runs the containers. If Qdrant already runs on your machine
   via Docker, you have this.
2. **A Google Gemini API key** — free to start. Get it at
   <https://aistudio.google.com/apikey>. This is the *only* credential you need; it is
   a simple API key, **not** a service-account JSON file (that's only for Google Vertex
   AI, which we are not using).
3. **(Optional) Python 3.12** — only if you want to run the app *without* Docker.
   Docker already includes Python, so you can skip this.

---

## 7. Setup, step by step

### Step 7.1 — Get your API key
Go to <https://aistudio.google.com/apikey>, sign in with a Google account, click
**Create API key**, and copy the string. Treat it like a password.

### Step 7.2 — Create your `.env` file
The repo ships a template called `.env.example`. Copy it to a real `.env` and paste
your key in:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```
GEMINI_API_KEY=paste_your_key_here
```

Leave the rest at the defaults (explained in
[section 11](#11-configuration-reference-env)). The file already has "max quality"
settings: `EMBED_DIM=3072` and the current models.

> **Why a `.env` file?** So your secret key lives in one place, is easy to change, and
> is kept out of the code and out of Git. Our `.dockerignore` and (recommended)
> `.gitignore` exclude it so you never accidentally publish your key.

### Step 7.3 — Make sure the `docker` command works in your terminal
Docker Desktop sometimes installs the `docker` command in `~/.docker/bin` **without
adding that folder to your shell's PATH**. If you type `docker --version` and get
`command not found`, that's the cause. Fix it once:

```bash
echo '' >> ~/.zshrc                                      # ensure a clean new line first
echo 'export PATH="$HOME/.docker/bin:$PATH"' >> ~/.zshrc # add Docker's CLI to PATH
source ~/.zshrc                                          # reload your current shell
docker --version                                         # should now print a version
```

> ⚠️ **The newline gotcha (we hit this).** If your `~/.zshrc` doesn't already end in a
> blank line, a plain `echo '...' >> ~/.zshrc` will glue the new line onto the last
> existing line and silently break both. That's the `echo '' >>` first line above —
> it guarantees a line break. If `docker` still isn't found, open `~/.zshrc` and check
> that the `export PATH=...` line sits on its own line.

---

## 8. Running it

You have two ways to run. **Docker is the recommended one** — it starts both the app
and Qdrant together with a single command.

### Option A — Docker (recommended)

```bash
# 1) Free the port. The compose file runs its OWN Qdrant, so stop any standalone
#    Qdrant container that's already using port 6333.
docker stop qdrant && docker rm qdrant

# 2) Build the app image and start both services in the background (-d = detached)
docker compose up -d --build

# 3) Check both are "Up"
docker compose ps

# 4) (optional) Follow the app's logs. Ctrl+C stops watching logs; it does NOT stop the app.
docker compose logs -f app
```

Now open **<http://localhost:8501>**.

To stop everything: `docker compose down`. Your data (vectors, rendered pages) stays on
disk — see [Persistence](#12-persistence--do-i-re-index-every-run).

> ⚠️ **Port conflict.** If step 2 fails with "port is already allocated", another
> container still holds `6333`. Run `docker ps` to find it and `docker stop <name>`.
> Only stop the Qdrant container for this project — leave any unrelated containers
> running.

#### How the containers talk to each other
Inside Docker, `localhost` means "this container", **not** your Mac. So the app cannot
reach Qdrant at `http://localhost:6333`. The `docker-compose.yml` fixes this by setting
`QDRANT_URL=http://qdrant:6333` for the app — `qdrant` is the *service name*, which
Docker resolves to the Qdrant container automatically. This override wins over whatever
is in your `.env` (Docker's `environment:` beats `env_file:`).

### Option B — Run locally without Docker (for development)

This runs the Python app directly on your Mac and talks to a Qdrant you started
separately. Useful when you're editing `app.py` a lot.

```bash
# Make sure a Qdrant is running on localhost:6333 (your existing Docker one is fine).
python3 -m venv .venv            # create an isolated Python environment
source .venv/bin/activate        # activate it (your prompt shows "(.venv)")
pip install -r requirements.txt  # install the libraries
pip install watchdog             # optional: smoother file-watching on macOS
streamlit run app.py             # start the app
```

In this mode the app reads `QDRANT_URL` from your `.env` (default
`http://localhost:6333`), which is correct because the app is *not* in a container.

---

## 9. Using it (the day-to-day loop)

1. **Open the app** at <http://localhost:8501>.
2. **Drop documents** into the `corpus/` folder on your Mac. Supported types:
   `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.txt`, `.md`.
3. **Wait ~3 seconds.** The sidebar has an auto-watcher that scans `corpus/` every 3
   seconds. When it sees a new file it indexes it automatically and pops a toast:
   *"🆕 N fragments indexed"*. No button to press.
4. **Ask a question** in the chat box at the bottom.
5. **Read the answer.** Below the text you'll see the **Sources**: for each source, a
   thumbnail of the page, with the file name, the page number, and a relevance score.
   That thumbnail is the actual rendered page — this is the "show me the picture of the
   manual page" requirement.

**Sidebar controls:**
- **Files in corpus** — a live count of indexable files detected.
- **🗑️ Reindex everything (reset)** — deletes the Qdrant collection and the manifest,
  then re-embeds every file from scratch. Use this only if you changed `EMBED_DIM` or
  want a clean slate. It costs API calls.
- **Sources per answer (top-k)** — slider, 1–10, how many chunks to retrieve per
  question.

---

## 10. How it works internally (deep dive)

This section walks the actual code paths. Reference: `app.py`.

### 10.1 Ingestion — turning files into searchable vectors

Triggered automatically by the watcher (see 10.4) or the reset button.

**`corpus_signature()`** builds a cheap fingerprint of the folder — for every file it
records `path:size:modified_time`. This is fast (no reading file contents). The watcher
compares this fingerprint to the previous one; only when it changes do we do real work.
This is what stops us from re-hashing and re-embedding on every 3-second tick.

**`index_corpus()`** lists indexable files, and — using the **manifest**
(`storage/manifest.json`) — skips any file whose content hash is already recorded. For
each *new* file it calls `points_for_file()` and upserts the results into Qdrant in
batches of 64.

**`points_for_file()`** handles three kinds of files:

- **PDF** (via PyMuPDF/`fitz`): for each page it
  1. renders the page to a PNG at `PAGE_DPI` (default 150) and saves it to
     `storage/pages/<hash>_p<N>.png`,
  2. extracts the page's text,
  3. creates **one multimodal embedding** from `[text, page_image]` — this is the key
     step; the vector captures both the words and the visual layout/diagrams,
  4. stores a Qdrant "point" = the vector + a **payload** (metadata):
     `{source, type:"pdf", page:N, text, image_path}`.
- **Image files** (PNG/JPG/…): embeds the image itself; payload points `image_path` at
  the original file so the chat can show it.
- **Text files** (TXT/MD): splits into ~1200-character chunks (with 150 overlap so we
  don't cut sentences awkwardly) and embeds each chunk (text-only).

**Deterministic IDs.** Each point's ID is `uuid5(namespace, "<filehash>:<page>")`.
Because the ID is derived from the file's content, re-indexing the same file writes to
the *same* ID (an "upsert") instead of creating duplicates. Idempotent by design.

**`embed()`** is the single choke-point that calls Gemini:

```python
res = gg.models.embed_content(
    model="gemini-embedding-2",
    contents=parts,                       # e.g. ["task: search document | <text>", <image Part>]
    config={"output_dimensionality": 3072},
)
return _norm(res.embeddings[0].values)    # one joint vector, L2-normalized
```

Two things to notice:
- **No `task_type` parameter.** The newer `gemini-embedding-2` model dropped it; you
  instead prefix the task as text, e.g. `"task: search document | ..."` for documents
  and `"task: question answering | query: ..."` for questions. This nudges the model
  to produce vectors optimized for retrieval.
- **`embeddings[0]`.** When you pass `[text, image]`, the model returns **one** combined
  vector (we verified this live). `[0]` is that joint vector.

**`_with_retry()`** wraps the API calls and retries on `503`/`500` (Google temporarily
unavailable) and `429` (rate limit) with exponential backoff (1s, 2s, 4s, 8s). This is
why a transient Google outage no longer crashes ingestion.

### 10.2 Retrieval — finding the right chunks

**`retrieve(query, k)`**:
1. Embeds the question with the query instruction:
   `"task: question answering | query: <your question>"`.
2. Calls `qc.query_points(collection, query=vector, limit=k)` — Qdrant returns the `k`
   nearest points by cosine similarity, each with its payload and a `score`.

### 10.3 Generation — writing the grounded answer

**`answer(query, hits)`**:
1. Builds a text **context** block listing each hit: `[n] <file> (page N): <text>`.
2. Attaches the **actual page images** of the top hits (up to 4) as image parts, so the
   LLM can *read the diagrams*, not just the extracted text.
3. Sends a prompt to `gemini-3.5-flash` that says, in Spanish: *answer ONLY from the
   context; if it's not there, say so; cite sources as `[file, page N]`.*
4. Returns the model's text. The UI then renders the answer **plus** the source
   thumbnails from the payloads.

This is the "grounding" that prevents hallucination: the model is told to use only what
we retrieved, and we show you the receipts.

### 10.4 The auto-watcher

The sidebar function is decorated with `@st.fragment(run_every="3s")`. A Streamlit
*fragment* re-runs on its own timer **without** reloading the whole page — so your chat
history stays intact while the folder is watched. Each tick it computes
`corpus_signature()`; if it differs from the stored one, it runs `index_corpus()` and
shows a toast. If nothing changed, it does essentially nothing (cheap `stat()` calls
only). That's how "drop a file and it's ready in ~3s" works with zero button clicks.

---

## 11. Configuration reference (`.env`)

Every setting, what it does, and when to change it:

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | *(none)* | **Required.** Your Google AI Studio key. |
| `QDRANT_URL` | `http://localhost:6333` | Where Qdrant is. In Docker this is overridden to `http://qdrant:6333` by `docker-compose.yml`. |
| `QDRANT_COLLECTION` | `ktm_rag` | Name of the Qdrant collection (like a table name). |
| `EMBED_MODEL` | `gemini-embedding-2` | The multimodal embedding model. |
| `GEN_MODEL` | `gemini-3.5-flash` | The model that writes answers. Use `gemini-flash-latest` if you want it to auto-track the newest flash. |
| `EMBED_DIM` | `3072` | Vector size. `3072` = max quality. `1536`/`768` = smaller & cheaper storage, tiny quality loss. **Changing this after indexing requires a Reset** (the collection's size is fixed at creation). |
| `CORPUS_DIR` | `./corpus` | Folder you drop documents into. |
| `STORAGE_DIR` | `./storage` | Where rendered pages + the manifest live. |
| `TOP_K` | `5` | Default number of sources retrieved per question. |
| `PAGE_DPI` | `150` | Resolution for rendered PDF pages. Higher = sharper images, bigger files, slower. |

> **Important rule about `EMBED_DIM`:** Qdrant fixes a collection's vector size when the
> collection is first created. If you later change `EMBED_DIM`, new vectors won't match
> the old collection and you'll get a dimension error. The fix is the **Reset** button
> (or delete the collection), which rebuilds it at the new size.

---

## 12. Persistence — do I re-index every run?

**No. You index each file once; it survives restarts.** Two independent layers:

1. **Vectors persist on disk.** Qdrant stores everything in `./qdrant_storage`, which is
   mounted from your Mac into the container (a "bind mount"). Restarts, reboots, and
   even `docker compose down` keep the data. It's only gone if you delete that folder by
   hand. (Note: `docker compose down -v` removes *named volumes* but **not** bind mounts,
   so your data is safe even then.)
2. **The app skips already-indexed files.** `storage/manifest.json` records which files
   are done (by content hash). On startup the watcher sees them and skips — **zero** API
   calls, instant readiness. Only new or changed files get embedded.

Because both `qdrant_storage/` and `storage/` are bind-mounted to your host, the
manifest and the vectors stay in sync across restarts. **Don't delete those two folders**
unless you want to re-embed everything (which costs API calls).

Even in the worst case (you delete the manifest but keep Qdrant), the deterministic IDs
prevent *duplicate* points — but it would still re-call the API to recompute embeddings.
So: keep `storage/` around.

---

## 13. Cost & quotas

- **Embeddings and answers are billed by Google**, per token/character and per image.
  Retrieval/search inside Qdrant is free (runs on your machine).
- The **free tier** of the Gemini API is enough to experiment. Heavy indexing of big
  PDFs will consume quota.
- **Cost controls in your hands:**
  - You embed each file **once** (thanks to the manifest) — re-runs are free.
  - Lower `TOP_K` and the number of attached images to reduce answer cost.
  - Lower `EMBED_DIM` (e.g. `1536`) to shrink storage; it barely affects quality.
  - Avoid pressing **Reset** casually — it re-embeds everything.

---

## 14. Troubleshooting (real errors we hit)

### `zsh: command not found: docker`
Docker's CLI isn't on your PATH. See [Step 7.3](#step-73--make-sure-the-docker-command-works-in-your-terminal).
Remember the **newline gotcha**: make sure the `export PATH=...` line is on its own line
in `~/.zshrc`, then `source ~/.zshrc`.

### `503 UNAVAILABLE — The service is currently unavailable`
A **temporary** Google-side outage during an embedding/generation call. It is not your
fault and not a wrong model name (a wrong name returns `404`, not `503`). The app now
**auto-retries** these, so it usually self-heals. If it persists, wait a few minutes.

### `404 NOT_FOUND — This model ... is no longer available`
The model name is retired. Google deprecates models over time. Fix: set a current model
in `.env` (we moved `GEN_MODEL` from the retired `gemini-2.5-flash` to `gemini-3.5-flash`).
To see what your key can actually use, list the models:

```bash
.venv/bin/python -c "
import os; from dotenv import load_dotenv; from google import genai
load_dotenv(); c = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
for m in c.models.list():
    acts = getattr(m,'supported_actions',[]) or []
    if 'generateContent' in acts or 'embedContent' in acts:
        print(m.name.replace('models/',''), acts)
"
```

### `port is already allocated` when running `docker compose up`
Another container (your standalone Qdrant) already holds `6333`. Stop it:
`docker ps` then `docker stop <name>`. Don't stop unrelated client containers.

### The app can't reach Qdrant (connection refused) inside Docker
You're pointing at `localhost` from inside a container. The compose file already sets
`QDRANT_URL=http://qdrant:6333` for you — make sure you started via
`docker compose up`, not by running the container by hand without that variable.

### "Dimension mismatch" / vector size error
You changed `EMBED_DIM` after the collection was created. Press **🗑️ Reindex everything
(reset)** in the sidebar, or delete the Qdrant collection, to rebuild at the new size.

### Streamlit didn't pick up my code change
Without the `watchdog` package, Streamlit's auto-reload is unreliable. Either
`pip install watchdog`, or fully restart: `Ctrl+C` and `streamlit run app.py` again
(local), or `docker compose up -d --build` again (Docker).

---

## 15. Glossary

- **LLM** — Large Language Model; the AI that writes text (Gemini here).
- **Embedding / vector** — a list of numbers representing meaning; similar meanings →
  nearby vectors.
- **Multimodal** — works with more than one type of input (here: text *and* images).
- **Dimension (`EMBED_DIM`)** — how many numbers per vector (3072 for us).
- **Vector database** — stores vectors and finds the nearest ones fast (Qdrant).
- **Cosine similarity** — the "closeness" measure between two vectors.
- **Chunk** — a small piece of a document that gets its own vector (here: one PDF page).
- **RAG** — Retrieve relevant chunks, Augment the prompt, Generate a grounded answer.
- **top-k** — how many nearest chunks to retrieve per question.
- **Payload** — the metadata stored next to each vector (file, page, text, image path).
- **Upsert** — insert-or-update; writing to an ID that may already exist.
- **Grounding** — forcing the model to answer only from provided context, with citations.
- **Hallucination** — when an LLM confidently makes up a wrong answer; RAG reduces this.
- **Bind mount** — a host folder mapped into a container so data persists on your Mac.
- **Container / image** — a packaged, portable copy of the app and its dependencies.

---

## 16. Where to go next

Ideas once you're comfortable:

- **Add a `.gitignore`** excluding `.env`, `.venv/`, `storage/`, `qdrant_storage/`,
  `__pycache__/` before putting this in Git.
- **Auto-caption images** with `gemini-3.5-flash` at ingestion time to make pure-image
  files more searchable by text.
- **Show the score as a percentage** and filter out low-confidence sources.
- **Support more file types** (DOCX, HTML) by adding branches in `points_for_file()`.
- **Stream the answer** token-by-token for a snappier feel.
- **Deploy it** somewhere so others can use it (remember: your `GEMINI_API_KEY` must be
  set as a secret in that environment, never committed).

That's the whole system, top to bottom. Welcome to RAG. 🏍️
