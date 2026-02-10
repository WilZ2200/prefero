# Roadmap (Internal — Developers Only)

## 1. Cloud Computing (GPU Offload)

**Status:** Planning

Offload heavy estimation jobs to cloud GPU instances instead of running on the HF Space CPU.

- **Provider candidates:** AWS (p3/g4 instances), GCP (T4/A100), Lambda Labs, Modal, RunPod
- **Architecture:** Job queue model — user submits spec, backend dispatches to a GPU worker, polls for result
- **Key decisions:**
  - Serverless (Modal/RunPod) vs persistent instance (EC2/GCE)?
  - Pricing model: free tier with limits vs pay-per-run?
  - How to handle long-running jobs (Mixed Logit with 5000 draws, Bootstrap 500 reps)?
- **Implementation sketch:**
  - `src/dce_analyzer/cloud.py` — job submission, status polling, result retrieval
  - `app/pages/2_Model.py` — toggle: "Run locally" vs "Run on cloud GPU"
  - Backend worker: containerized `estimate_from_spec()` with GPU PyTorch
  - Auth: tie cloud jobs to user accounts (rate limiting per user)
- **Milestones:**
  - [ ] Prototype with Modal (serverless GPU, simplest to set up)
  - [ ] Job status UI (progress bar, estimated time, cancel button)
  - [ ] Result caching (don't re-run identical specs)
  - [ ] Cost tracking and usage limits

## 2. Email Verification (OTP Auth)

**Status:** Planning

Replace or supplement the current MUCHE member + invitation code auth with proper email-based OTP authentication.

- **Current auth:** Name fuzzy-match + password (MUCHE members) or invitation code
- **Target:** Email input -> send OTP code -> verify -> create/login account
- **SMTP setup:** Already have `SMTP_USER` / `SMTP_PASS` env vars wired up (from earlier OTP code that was replaced)
- **Key decisions:**
  - Keep MUCHE member whitelist as a separate path, or unify under email?
  - OTP expiry time (5 min?)
  - Rate limiting on OTP sends (prevent spam)
  - Email domain restrictions (e.g. only `.edu` or `.edu.au`)?
- **Implementation sketch:**
  - `app/auth.py` — add "Email OTP" tab alongside existing tabs
  - OTP generation: 6-digit code, stored in session state with expiry timestamp
  - Email template: simple HTML with code and branding
  - After verification: create user account with verified email, or login if account exists
- **Milestones:**
  - [ ] SMTP integration (send test email)
  - [ ] OTP generation + verification flow
  - [ ] Rate limiting (max 3 OTP sends per email per hour)
  - [ ] Account linking (email -> existing username)

## 3. AI Analysis Report

**Status:** Planning

Auto-generate a plain-language interpretation of estimation results using an LLM.

- **Trigger:** After estimation completes, user clicks "Generate AI Report"
- **Input to LLM:** Model spec, parameter estimates, fit metrics, WTP, significance levels, data summary
- **Output:** Structured report with sections:
  - Executive summary (1-2 sentences: what did we learn?)
  - Key findings (which attributes matter most, direction of effects)
  - WTP interpretation (in dollar terms, relative importance)
  - Model diagnostics (convergence, fit quality, red flags)
  - Recommendations (next steps, model improvements to try)
- **Key decisions:**
  - LLM provider: Claude API (Anthropic) vs OpenAI vs local model?
  - API key management: per-user keys or shared app key with rate limits?
  - Streaming vs batch response?
  - Caching: cache report per (model_spec_hash, estimates_hash)?
- **Implementation sketch:**
  - `src/dce_analyzer/ai_report.py` — prompt construction, API call, response parsing
  - `app/pages/3_Results.py` — "Generate AI Report" button, streaming display
  - Prompt template: structured with all estimation data, ask for specific sections
  - Fallback: if API fails, show raw summary dict instead
- **Milestones:**
  - [ ] Prompt engineering (test with Claude API on real estimation outputs)
  - [ ] API integration with key management
  - [ ] Streaming UI in Results page
  - [ ] PDF export of AI report
  - [ ] Multi-model comparison report (compare 2+ models in one report)
