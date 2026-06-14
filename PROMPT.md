# PostForge — Instagram Analytics & Optimization Tool
## Prompt for Claude Code (uv-powered Streamlit App)

You are an expert full-stack Python developer, data analyst, and social media growth strategist specializing in modern tooling. Create a complete, polished, production-ready web application called **PostForge** — an advanced Instagram analytics, insights, and optimization tool.

**Critical Requirement:** The entire project **must use `uv`** (Astral's ultra-fast Python package and project manager) for all dependency management, environment handling, and running. Do **not** use pip/venv directly in instructions. Use `uv init`, `uv add`, `uv sync`, and `uv run` throughout.

The tool must be built as a modern `uv` project that feels premium, modern, and intuitive. Use a clean professional design with Instagram-inspired accents (deep purple, pink, and orange gradients) and excellent UX.

### Project Setup with `uv` (Mandatory)
- Initialize the project using `uv init` (or `uv init --app`).
- Manage **all** dependencies exclusively with `uv add <package>`.
- The project must include a proper `pyproject.toml` (this is the single source of truth for dependencies).
- Provide a `.python-version` file (managed by `uv`).
- Include clear instructions in the README for:
  - `uv sync`
  - `uv run streamlit run app.py`
  - How to add new dependencies (`uv add ...`)
  - How to export a `requirements.txt` for platforms that don't support `uv` natively (e.g., some deployment targets).
- For Streamlit Community Cloud deployment, note that users can either use the `uv` workflow or fall back to a generated `requirements.txt`.

### Core Requirements

**1. Tech Stack & Dependencies (managed via `uv`)**
Primary dependencies to add via `uv`:
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `requests`
- `vaderSentiment`
- `scikit-learn`
- `Pillow`
- `reportlab`
- `python-dotenv`

Optional/recommended:
- `wordcloud`
- `matplotlib` (for wordcloud backend)

Claude should generate a clean `pyproject.toml` with these dependencies under `[project.dependencies]`.

**2. Authentication & Data Connection**
- The tool connects exclusively via the **official Instagram Graph API** (Meta Graph API, latest stable version).
- User inputs a **Long-Lived User Access Token** (60 days) and their **Instagram User ID** (numeric).
- Include a prominent, well-written **"How to get your Access Token & IG User ID (2026)"** expander or dedicated help section with accurate, up-to-date steps:
  - Switch account to Professional (Creator or Business) if needed.
  - Create Meta for Developers app.
  - Required permissions (`instagram_basic`, `instagram_manage_insights`, `pages_read_engagement`, etc.).
  - How to generate and extend the token to long-lived.
  - How to retrieve the IG User ID.
- On connect: Validate token + fetch and beautifully display profile info (username, profile picture via URL, biography, followers_count, media_count, etc.).
- Excellent error handling for invalid tokens, rate limits, non-professional accounts, or missing insights.

**3. Data Fetching (respect rate limits)**
- Fetch media using cursor-based pagination (`after` parameter).
- Request rich fields: `id, caption, media_type, media_url, thumbnail_url, timestamp, like_count, comments_count, permalink, shares_count, saved_count, children{id,media_type,media_url}`.
- For insights: Call `/{media-id}/insights` with currently supported metrics (`reach`, `engagement`, `saves`, `shares`, `views`). Handle deprecations gracefully (try/except + fallbacks to engagement rate calculated from `like_count` + `comments_count`).
- **Smart & respectful fetching**:
  - Default: Analyze the **last 150 posts** (user-adjustable 50–300 range).
  - Clear warnings about rate limits.
  - Progress bar + live status ("Fetching page 2/8 of media...", "Fetching insights for post 47/150...").
  - `time.sleep()` between calls where appropriate.
- Cache results aggressively in `st.session_state` + optional local SQLite (`sqlite3`).
- "Refresh Data" and "Clear Cache" controls.

**4. Data Processing & Feature Engineering**
Build a rich pandas DataFrame with meaningful engineered features:
- `engagement_rate` (primary KPI): `(likes + comments + shares + saves) / max(reach or followers, 1) * 100`
- Time features: `posted_at`, `hour`, `weekday_name`, `is_weekend`, `month`, `year`
- Caption features: `caption_length`, `word_count`, `has_question`, `emoji_count`, `cta_score` (smart detection of calls-to-action)
- Hashtag features: `hashtag_count`, list of hashtags
- Media features: `media_type`, `is_carousel`, `num_slides` (when available)
- NLP: VADER sentiment score + label
- Any other high-signal derived columns

**5. Dashboard Structure (Premium UX)**
Use a clean sidebar + top-level tabs or multi-page feel:

**A. Overview Dashboard**
- Large, beautiful KPI cards.
- Engagement trend line chart (rolling average).
- Media type performance comparison (interactive bar/pie charts).
- Top 5 performing posts displayed as attractive cards with `st.image` thumbnails, metrics, caption preview, and "Open on Instagram" buttons.

**B. Post Explorer**
- Powerful interactive table with search, filters, and sorting.
- Quick preview modal or expander for any post (thumbnail + full caption + all metrics + direct link).

**C. Content Analysis**
- Posting time heatmap (weekday × hour) — color by average engagement rate.
- Hashtag performance analysis (usage vs. average engagement when used).
- Caption & sentiment deep-dive (length distributions, sentiment pie, high vs low performer comparisons).
- Word cloud or top terms from top-performing content.

**D. What Makes a Post Good? (The Star Feature)**
This section must feel intelligent and data-driven:
- Automatically identify "Good posts" = top 25% by `engagement_rate`.
- Statistical comparisons between top vs bottom performers across all features.
- Clear, quantified insights (e.g., "Carousel posts achieve 2.4× higher average engagement rate than single images in your account.").
- Correlation heatmap and/or `scikit-learn` feature importance analysis (RandomForest or permutation importance) showing which factors most strongly predict success in *this specific account*.
- Visual examples of winning patterns with real posts from the user's data.

**E. Personalized Recommendations & Growth Plan**
- Sophisticated rule-based + statistical recommendation engine that reads the actual analysis results.
- Prioritized, actionable advice tailored to the user's data (posting schedule, content mix, caption strategy, hashtag usage, CTA tactics, etc.).
- "Smart Growth Plan" summary with clear next steps.
- Optional: Field for user to paste an Anthropic / OpenAI / Grok API key for even richer LLM-generated recommendations (the tool should construct a high-quality context prompt from the analysis).

**F. Audience Insights**
- Display account-level insights (demographics, reach trends, etc.) when the Graph API provides them.

**G. Export & Reporting**
- One-click beautiful multi-page **PDF Report** (using `reportlab`) containing KPIs, key charts, top/bottom posts, "What Makes a Post Good" findings, and the full Growth Plan.
- Export analyzed data as CSV.

**6. UX & Polish**
- Modern, clean, Instagram-feeling interface.
- Excellent loading states, progress indicators, and helpful error messages.
- Responsive layout and interactive Plotly charts.
- Helpful tooltips and explanations.
- First-time user guidance and clear onboarding.

**7. Robustness**
- Comprehensive error handling and user-friendly messaging (rate limits, missing insights on older posts, permission issues, etc.).
- Respect Meta rate limits and document them transparently.
- Secure token handling (session-only by default).
- Well-structured, modular, commented code.

**8. Deliverables (organized as a proper `uv` project)**
Generate the following files clearly separated:

1. `pyproject.toml` (complete, with all dependencies under `[project.dependencies]`)
2. `app.py` (the full Streamlit application — well-organized with functions for fetching, analysis, UI rendering, etc.)
3. `README.md` (excellent quality) containing:
   - Project description
   - `uv` setup & run instructions (`uv sync`, `uv run streamlit run app.py`, adding packages, etc.)
   - Complete 2026-accurate guide: "How to obtain your Instagram Graph API Access Token & IG User ID"
   - Deployment notes (Streamlit Cloud + `uv` workflow)
   - Limitations and future enhancement ideas
4. `.python-version` (suggested)
5. (Optional but nice) A minimal `.gitignore`

Make the code clean, maintainable, and genuinely useful. The "What Makes a Post Good?" and Recommendations sections should feel like they were built by a smart growth strategist who analyzed the user's real data — not generic advice.

Begin building the project now. Start by creating the `uv` project structure and core connection/fetching logic, then layer on the rich analysis and beautiful UI components.

---

**End of Prompt**

**Instructions for user (do not include in the file sent to Claude):**
- Place this `PROMPT.md` in your Claude Code workspace.
- In Claude Code, you can reference it or paste its content when starting a new project.
- After Claude generates the files, you can immediately run:
  ```bash
  uv sync
  uv run streamlit run app.py
  ```
