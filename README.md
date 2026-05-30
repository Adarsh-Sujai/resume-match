# ResumeMatch

A web app that scores how well a resume matches a job description, highlights missing keywords, runs ATS readability checks, and uses a hosted LLM to suggest tailored bullet rewrites.

Built by [Adarsh K Sujai](https://adarsh-sujai.github.io).

## Features

- **Match score** - TF-IDF cosine similarity between resume and JD, rescaled to a 0–100 score.
- **Missing keywords** - surfaces high-frequency JD terms absent from the resume.
- **ATS readability check** - flags scanned PDFs, missing section headers, placeholder text, broken text extraction.
- **AI rewrite (Groq + Llama 3.3)** - generates a tailored bullet point that aligns with the JD.
- **Rate-limited and secure** - API key stays server-side, per-IP rate limits prevent quota abuse.

## Tech stack

- Flask · scikit-learn · pypdf · python-docx
- Groq API (Llama 3.3 70B)
- Vanilla HTML/CSS/JS frontend (no build step)
- Deployed on Render

## Local setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Adarsh-Sujai/resumematch.git
   cd resumematch
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your Groq API key (get one free at https://console.groq.com/keys).

5. Run the app:
   ```bash
   python app.py
   ```
   Open http://localhost:5000.

## How the API key stays secure

- The key is loaded server-side from an environment variable via `python-dotenv`.
- The browser only ever talks to your Flask server (`/api/analyze`, `/api/rewrite`). Server calls Groq. The key never appears in any HTML, JS, or response payload.
- `.env` is git-ignored. On Render, the key is stored as an encrypted environment variable.
- `Flask-Limiter` enforces per-IP rate limits to prevent abuse of the free Groq quota.

## Project structure

```
resumematch/
├── app.py                  # Flask backend with all routes
├── templates/
│   └── index.html          # Single-page frontend
├── requirements.txt
├── render.yaml             # Render deployment config
├── .env.example            # Template for environment variables
├── .gitignore
└── README.md
```

## Roadmap

- [ ] User accounts to save analyses
- [ ] Bulk JD analysis (paste 5 JDs, get aggregate skill gaps)
- [ ] Better ATS check (font detection, image count)
- [ ] Cover letter generator using same JD + resume context

## License

MIT
