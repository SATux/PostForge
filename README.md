# PostForge 🔥

**PostForge** is an Instagram analytics and optimization tool built with Streamlit. It connects to the official Meta Graph API to analyze your posts, surface what content performs best, and generate a personalized growth plan.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv run streamlit run app.py
```

That's it — uv handles the virtual environment and all dependencies automatically.

### Adding dependencies

```bash
uv add <package-name>
```

### Export requirements.txt (for platforms without uv support)

```bash
uv export --format requirements-txt > requirements.txt
```

---

## How to Get Your Instagram Graph API Access Token & User ID (2026)

### 1. Switch to a Professional Account

In the Instagram app: **Settings → Account → Switch to Professional Account**. Choose Creator or Business.

### 2. Create a Meta for Developers App

1. Go to [developers.facebook.com](https://developers.facebook.com) and log in with the Facebook account linked to your Instagram.
2. Click **My Apps → Create App**.
3. Select **Business** as the app type.
4. Add the **Instagram Graph API** product to your app.

### 3. Request Required Permissions

In your app's dashboard under **App Review / Permissions**, add:
- `instagram_basic`
- `instagram_manage_insights`
- `pages_read_engagement`

For testing, you can use these as a developer without full App Review.

### 4. Generate a Long-Lived Access Token

**Step A — Get a short-lived token:**
1. Open the [Graph API Explorer](https://developers.facebook.com/tools/explorer/).
2. Select your app, click **Generate Access Token**, and grant the permissions above.

**Step B — Extend to a long-lived token (60 days):**
```
GET https://graph.facebook.com/oauth/access_token
    ?grant_type=fb_exchange_token
    &client_id=YOUR_APP_ID
    &client_secret=YOUR_APP_SECRET
    &fb_exchange_token=SHORT_LIVED_TOKEN
```

Use the `access_token` from the response in PostForge.

### 5. Get Your Instagram User ID

```
GET https://graph.facebook.com/v21.0/me/accounts?access_token=YOUR_TOKEN
```

Find the `instagram_business_account` object inside your Page — its `id` is your numeric IG User ID.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the main file to `app.py`.
4. Either commit a `requirements.txt` (generated above) or configure Streamlit Cloud to use uv via `packages.txt` + a startup script.

---

## Limitations

- **Rate limits:** The Meta Graph API enforces per-app and per-user rate limits. PostForge adds deliberate delays between insight calls. Analyzing 300 posts will take several minutes.
- **Insights availability:** Post-level insights (reach, saves) may be unavailable for posts older than ~2 years or for accounts that recently switched to Professional. PostForge falls back to calculating engagement rate from likes + comments when insights are missing.
- **Token expiry:** Long-lived tokens last 60 days. Refresh by repeating step 4B above.
- **Reels/Stories:** Story insights are not supported by this version. Reels are included as VIDEO media type.

## Future Ideas

- Automated token refresh via OAuth flow
- Competitor benchmarking (public accounts)
- Scheduled digest emails
- Content calendar with optimal-time suggestions
- Hashtag research against public engagement data
