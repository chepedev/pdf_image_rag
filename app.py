"""
KTM-RAG — Dashboard RAG multimodal con gemini-embedding-2 + Qdrant.
Suelta documentos en ./corpus, pulsa "Indexar", y chatea.
El chat SIEMPRE devuelve la fuente: archivo, página y la imagen/captura del documento.
"""
import os
import io
import json
import time
import uuid
import hashlib
import mimetypes
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "ktm_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "./corpus")).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage")).resolve()
TOP_K = int(os.getenv("TOP_K", "5"))
PAGE_DPI = int(os.getenv("PAGE_DPI", "150"))

PAGES_DIR = STORAGE_DIR / "pages"
MANIFEST = STORAGE_DIR / "manifest.json"
NS = uuid.UUID("12345678-1234-5678-1234-567812345678")
IMG_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}
TEXT_EXT = {".txt", ".md", ".markdown"}
INDEXABLE = set(IMG_MIME) | {".pdf"} | TEXT_EXT

for d in (CORPUS_DIR, PAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Clientes (cacheados)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_clients():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Falta GEMINI_API_KEY en el .env"); st.stop()
    gg = genai.Client(api_key=api_key)
    qc = QdrantClient(url=QDRANT_URL)
    names = [c.name for c in qc.get_collections().collections]
    if COLLECTION not in names:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    return gg, qc


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Embeddings (gemini-embedding-2 es multimodal: texto + imagen juntos)
# --------------------------------------------------------------------------- #
def _norm(v):
    s = sum(x * x for x in v) ** 0.5
    return [x / s for x in v] if s else v


def _with_retry(fn, tries=4, base=1.0):
    """Reintenta ante 503/500 (servidor) y 429 (rate limit) con backoff exponencial."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except errors.ServerError as e:                       # 5xx transitorios
            last = e
        except errors.ClientError as e:                       # solo reintentar 429
            if getattr(e, "code", None) != 429:
                raise
            last = e
        time.sleep(base * (2 ** i))
    raise last


def embed(parts, dim=EMBED_DIM):
    """parts: lista de str y/o types.Part. Devuelve 1 vector (embedding conjunto)."""
    gg, _ = get_clients()
    res = _with_retry(lambda: gg.models.embed_content(
        model=EMBED_MODEL,
        contents=parts,
        config={"output_dimensionality": dim},
    ))
    return _norm(list(res.embeddings[0].values))


def img_part(data: bytes, mime: str) -> types.Part:
    return types.Part.from_bytes(data=data, mime_type=mime)


# --------------------------------------------------------------------------- #
# Ingesta
# --------------------------------------------------------------------------- #
def file_sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def chunk_text(text: str, size=1200, overlap=150):
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out or [""]


def points_for_file(path: Path, sha: str):
    """Genera PointStruct(s) para un archivo. Devuelve lista de puntos."""
    ext = path.suffix.lower()
    pts = []

    if ext == ".pdf":
        doc = fitz.open(path)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=PAGE_DPI)
            png = pix.tobytes("png")
            img_path = PAGES_DIR / f"{sha}_p{i}.png"
            img_path.write_bytes(png)
            text = (page.get_text() or "").strip()
            vec = embed([f"task: search document | {text[:6000]}", img_part(png, "image/png")])
            pts.append(PointStruct(
                id=str(uuid.uuid5(NS, f"{sha}:{i}")),
                vector=vec,
                payload={"source": path.name, "type": "pdf", "page": i,
                         "text": text[:1500], "image_path": str(img_path)},
            ))
        doc.close()

    elif ext in IMG_MIME:
        data = path.read_bytes()
        vec = embed(["task: search document | imagen de documento / manual",
                     img_part(data, IMG_MIME[ext])])
        pts.append(PointStruct(
            id=str(uuid.uuid5(NS, f"{sha}:0")),
            vector=vec,
            payload={"source": path.name, "type": "image", "page": None,
                     "text": "", "image_path": str(path)},
        ))

    elif ext in TEXT_EXT:
        for j, ch in enumerate(chunk_text(path.read_text(errors="ignore"))):
            vec = embed([f"task: search document | {ch}"])
            pts.append(PointStruct(
                id=str(uuid.uuid5(NS, f"{sha}:{j}")),
                vector=vec,
                payload={"source": path.name, "type": "text", "page": None,
                         "text": ch, "image_path": None},
            ))
    return pts


def corpus_signature():
    """Firma barata del corpus (ruta:tamaño:mtime) para detectar cambios sin hashear."""
    sig = []
    for p in sorted(CORPUS_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in INDEXABLE:
            s = p.stat()
            sig.append(f"{p}:{s.st_size}:{int(s.st_mtime)}")
    return sig


def index_corpus(force=False):
    _, qc = get_clients()
    manifest = {} if force else load_manifest()
    files = [p for p in sorted(CORPUS_DIR.rglob("*"))
             if p.is_file() and p.suffix.lower() in INDEXABLE]
    new = [p for p in files if force or file_sha(p) not in manifest]
    if not new:
        return 0, len(files)

    bar = st.progress(0.0, text="Indexando…")
    total_pts = 0
    for k, path in enumerate(new, start=1):
        sha = file_sha(path)
        pts = points_for_file(path, sha)
        for i in range(0, len(pts), 64):
            qc.upsert(collection_name=COLLECTION, points=pts[i:i + 64])
        manifest[sha] = {"file": path.name, "chunks": len(pts)}
        save_manifest(manifest)          # incremental: sobrevive a interrupciones
        total_pts += len(pts)
        bar.progress(k / len(new), text=f"Indexado: {path.name}")
    bar.empty()
    return total_pts, len(files)


# --------------------------------------------------------------------------- #
# Recuperación + respuesta
# --------------------------------------------------------------------------- #
def retrieve(query: str, k=TOP_K):
    _, qc = get_clients()
    qvec = embed([f"task: question answering | query: {query}"])
    res = qc.query_points(collection_name=COLLECTION, query=qvec, limit=k, with_payload=True)
    return res.points


def answer(query: str, hits):
    gg, _ = get_clients()
    ctx_lines, img_parts = [], []
    for n, h in enumerate(hits, start=1):
        p = h.payload
        loc = f"pág. {p['page']}" if p.get("page") else "imagen"
        ctx_lines.append(f"[{n}] {p['source']} ({loc}): {p.get('text','')[:900]}")
        ip = p.get("image_path")
        if ip and Path(ip).exists() and n <= 4:
            mime = mimetypes.guess_type(ip)[0] or "image/png"
            img_parts.append(img_part(Path(ip).read_bytes(), mime))

    prompt = (
        "Eres un asistente que responde SÓLO con el CONTEXTO (texto e imágenes de "
        "documentos del usuario). Si no está en el contexto, dilo. Responde en español "
        "y cita las fuentes como [archivo, pág. N].\n\n"
        f"PREGUNTA: {query}\n\nCONTEXTO:\n" + "\n".join(ctx_lines)
    )
    resp = _with_retry(lambda: gg.models.generate_content(
        model=GEN_MODEL, contents=[prompt, *img_parts]))
    return resp.text


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="KTM-RAG", page_icon="🏍️", layout="wide")
st.title("🏍️ KTM-RAG — chatea con tus documentos")

@st.fragment(run_every="3s")
def corpus_watcher():
    """Vigila la carpeta e indexa automáticamente al soltar archivos (sin botón)."""
    sig = corpus_signature()
    st.metric("Archivos en corpus", len(sig))
    if st.session_state.get("_corpus_sig") != sig:
        with st.spinner("Cambios detectados, indexando…"):
            added, _ = index_corpus()
        st.session_state._corpus_sig = sig       # marca "visto" SÓLO si indexó sin error
        if added:
            st.toast(f"🆕 {added} fragmentos indexados", icon="✅")
    st.caption("👀 Vigilando la carpeta cada 3 s")


with st.sidebar:
    st.header("Corpus")
    st.caption(f"📂 Suelta archivos en:\n`{CORPUS_DIR}`")
    corpus_watcher()
    st.divider()
    if st.button("🗑️ Reindexar todo (reset)", use_container_width=True):
        _, qc = get_clients()
        qc.delete_collection(COLLECTION)
        if MANIFEST.exists():
            MANIFEST.unlink()
        st.session_state.pop("_corpus_sig", None)
        get_clients.clear()
        with st.spinner("Reindexando desde cero…"):
            index_corpus(force=True)
        st.success("Colección reconstruida."); st.rerun()
    top_k = st.slider("Fuentes por respuesta (top-k)", 1, 10, TOP_K)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    st.caption("📎 Fuentes")
    cols = st.columns(min(len(sources), 4))
    for i, s in enumerate(sources):
        with cols[i % len(cols)]:
            loc = f"pág. {s['page']}" if s.get("page") else "imagen"
            if s.get("image_path") and Path(s["image_path"]).exists():
                st.image(s["image_path"], caption=f"{s['source']} · {loc} · {s['score']:.2f}",
                         use_container_width=True)
            else:
                st.info(f"**{s['source']}** · {loc} · {s['score']:.2f}\n\n{s.get('text','')[:200]}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if q := st.chat_input("Pregunta sobre tus documentos…"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el corpus…"):
            hits = retrieve(q, k=top_k)
            text = answer(q, hits)
        st.markdown(text)
        sources = [{"source": h.payload["source"], "page": h.payload.get("page"),
                    "image_path": h.payload.get("image_path"),
                    "text": h.payload.get("text", ""), "score": h.score} for h in hits]
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
