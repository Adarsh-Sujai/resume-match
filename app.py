"""
ResumeMatch — a resume vs. job description analyzer.

Architecture:
  Browser → Flask server (this file) → Groq API
The API key lives ONLY on the server side, loaded from an environment
variable. The browser never sees it.
"""

import os
import re
import io
import secrets
import logging
from collections import Counter

from flask import Flask, render_template, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx
import requests

# Load .env file for local development. In production (Render), env vars
# are injected directly by the platform.
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload cap

# Render terminates TLS at its load balancer and forwards the request over a
# single proxy hop. Without this, request.remote_addr is the proxy's IP, so the
# rate limiter would key EVERY visitor to the same bucket. ProxyFix makes Flask
# trust one hop of X-Forwarded-For / -Proto so we see the real client IP.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Rate limit so a single visitor cannot drain the free Groq quota.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://",
)


@app.before_request
def make_csp_nonce():
    """Per-request nonce that whitelists our one inline <script> in the CSP."""
    g.csp_nonce = secrets.token_urlsafe(16)


@app.context_processor
def inject_csp_nonce():
    """Expose the nonce to templates as {{ csp_nonce }}."""
    return {"csp_nonce": getattr(g, "csp_nonce", "")}


def build_csp():
    # The frontend keeps all CSS and JS inline in index.html. The inline script
    # is allowed via a per-request nonce (real XSS protection). Inline styles
    # need 'unsafe-inline' because CSP nonces don't cover style="" attributes;
    # CSS injection is low risk. No external origins are used.
    nonce = getattr(g, "csp_nonce", "")
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    )


@app.after_request
def set_security_headers(response):
    """Apply hardening headers to every response."""
    response.headers["Content-Security-Policy"] = build_csp()
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=()"
    )
    # Only advertise HSTS over real HTTPS so local http dev isn't pinned.
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# A short, focused stopword list. Anything common enough to appear in most
# resumes/JDs gets stripped before keyword extraction.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "you", "your", "we", "our", "this",
    "they", "their", "them", "i", "me", "my", "but", "if", "not", "no",
    "do", "does", "did", "can", "could", "should", "would", "what",
    "which", "who", "when", "where", "how", "all", "any", "some",
    "such", "than", "then", "so", "about", "across", "after", "also",
    "been", "being", "both", "each", "into", "more", "most", "other",
    "over", "same", "up", "down", "out", "off", "above", "below",
    "role", "work", "team", "experience", "years", "year", "must",
    "able", "ability", "preferred", "required", "responsibilities",
    "qualifications", "skills", "ideal", "candidate", "looking", "join",
    "company", "us", "include", "including", "etc", "well", "good",
    "strong", "excellent", "great",
}


def extract_text_from_pdf(file_stream):
    """Extract text from an uploaded PDF stream."""
    try:
        reader = pypdf.PdfReader(file_stream)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        log.warning("PDF parse failed: %s", e)
        return ""


def extract_text_from_docx(file_stream):
    """Extract text from an uploaded DOCX stream."""
    try:
        d = docx.Document(file_stream)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        log.warning("DOCX parse failed: %s", e)
        return ""


def tokenize(text):
    """Lowercase, strip non-alphanumeric, drop stopwords and 1-2 char tokens."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#\.\-]{1,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def match_score(resume_text, jd_text):
    """TF-IDF cosine similarity between resume and JD, scaled to 0-100."""
    if not resume_text.strip() or not jd_text.strip():
        return 0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vec.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        # TF-IDF scores tend to look small; rescale so a reasonable match
        # reads as ~70-85% rather than 0.3.
        return round(min(sim * 180, 100))
    except Exception as e:
        log.warning("Similarity failed: %s", e)
        return 0


def missing_keywords(resume_text, jd_text, top_n=12):
    """Return tokens that appear frequently in JD but not in resume."""
    jd_tokens = tokenize(jd_text)
    resume_tokens = set(tokenize(resume_text))
    jd_counts = Counter(jd_tokens)
    missing = [
        (token, count) for token, count in jd_counts.most_common(80)
        if token not in resume_tokens and count >= 1
    ]
    return [token for token, _ in missing[:top_n]]


def ats_check(resume_text):
    """Run basic ATS-readability heuristics on the resume."""
    issues = []
    if len(resume_text.strip()) < 200:
        issues.append("Resume text is very short. Image-based or scanned PDFs cannot be parsed by ATS.")
    sections = ["experience", "education", "skill"]
    found = [s for s in sections if s in resume_text.lower()]
    missing_sections = [s for s in sections if s not in found]
    if missing_sections:
        issues.append(f"Missing common section headers: {', '.join(missing_sections)}.")
    if re.search(r"\{[^}]+\}", resume_text):
        issues.append("Placeholder text in curly braces detected (e.g. {Name}). Replace before sending.")
    if "  " in resume_text and resume_text.count("  ") > 20:
        issues.append("Many double spaces detected — could indicate broken text extraction.")
    return issues or ["No major ATS issues detected."]


def call_groq_rewrite(resume_snippet, jd_text):
    """Ask Groq's hosted Llama to rewrite a resume bullet to match the JD."""
    if not GROQ_API_KEY:
        return None, "AI rewrite is unavailable (server not configured)."

    prompt = (
        "You are helping a job applicant tailor their resume to a specific "
        "job description. Below is a short excerpt from their resume and "
        "the full job description.\n\n"
        "Rewrite ONE strong bullet point (1-2 lines, starts with an action "
        "verb, includes a measurable detail if possible) that better aligns "
        "this candidate with the job. Output only the rewritten bullet — no "
        "preamble, no quotes, no explanation.\n\n"
        f"RESUME EXCERPT:\n{resume_snippet[:1500]}\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:2500]}"
    )
    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 200,
            },
            timeout=20,
        )
        if r.status_code != 200:
            log.warning("Groq error %s: %s", r.status_code, r.text[:200])
            return None, "AI rewrite temporarily unavailable."
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text, None
    except requests.RequestException as e:
        log.warning("Groq request failed: %s", e)
        return None, "AI rewrite temporarily unavailable."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("10 per hour")
def analyze():
    """Main analysis endpoint. Accepts a resume file and JD text."""
    jd_text = (request.form.get("jd") or "").strip()
    if not jd_text:
        return jsonify({"error": "Please paste a job description."}), 400
    if len(jd_text) < 50:
        return jsonify({"error": "Job description is too short."}), 400

    if "resume" not in request.files:
        return jsonify({"error": "Please upload a resume file."}), 400

    f = request.files["resume"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = f.filename.lower()
    file_bytes = f.read()
    stream = io.BytesIO(file_bytes)

    if filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(stream)
    elif filename.endswith(".docx"):
        resume_text = extract_text_from_docx(stream)
    else:
        return jsonify({"error": "Only PDF and DOCX files are supported."}), 400

    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from resume. If it's a scanned PDF, save it as a text-based PDF first."}), 400

    score = match_score(resume_text, jd_text)
    missing = missing_keywords(resume_text, jd_text)
    ats = ats_check(resume_text)

    return jsonify({
        "score": score,
        "missing_keywords": missing,
        "ats_issues": ats,
        "resume_preview": resume_text[:600],
    })


@app.route("/api/rewrite", methods=["POST"])
@limiter.limit("8 per hour")
def rewrite():
    """LLM-powered bullet rewrite. Keyed by a snippet, not the whole resume."""
    data = request.get_json(silent=True) or {}
    snippet = (data.get("snippet") or "").strip()
    jd_text = (data.get("jd") or "").strip()
    if not snippet or not jd_text:
        return jsonify({"error": "Snippet and JD are required."}), 400

    result, err = call_groq_rewrite(snippet, jd_text)
    if err:
        return jsonify({"error": err}), 503
    return jsonify({"rewrite": result})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 2 MB."}), 413


@app.errorhandler(429)
def ratelimit(e):
    return jsonify({"error": "Rate limit hit. Try again in a bit."}), 429


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
