# PostForge 🔥

**PostForge** is an Instagram analytics and optimization tool built with Streamlit and the official **Meta Instagram Graph API**. It analyzes your posts, surfaces what content performs best, and generates a personalized growth plan.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv run streamlit run app.py
```

---

## How to Connect PostForge to Instagram

The Meta Instagram Graph API requires a **Business or Creator** Instagram account linked to a **Facebook Page**, plus a Meta Developer app. Follow these steps exactly.

---

### Step 1 — Switch to a Professional Instagram Account

In the **Instagram mobile app**:

> Settings → Account → Switch to Professional Account

Choose **Creator** or **Business**. This is required — personal accounts cannot access the Graph API.

---

### Step 2 — Link Your Instagram Account to a Facebook Page

In the **Instagram mobile app**:

> Settings → Account → Sharing to other apps → Facebook

Connect it to a Facebook Page. If you don't have one, create a placeholder Page — it just needs to exist. This link is what lets the Graph API identify your Instagram account.

---

### Step 3 — Create a Meta Developer App

1. Go to **[developers.facebook.com](https://developers.facebook.com)** and log in with the **Facebook account that owns your Page**.
2. Click **My Apps → Create App**.
3. When asked for a use case, select **Other**, then choose **Business** as the app type.
4. Give the app a name (e.g. "PostForge") and click **Create App**.
5. On the app dashboard, scroll to **Add a Product** and click **Set Up** next to **Instagram Graph API**.

---

### Step 4 — Generate a Short-Lived Access Token

1. Open the **[Graph API Explorer](https://developers.facebook.com/tools/explorer/)**.
2. At the top, set **Meta App** to the app you just created.
3. Click the **Generate Access Token** button.
4. A permissions dialog will appear — check these three:
   - `instagram_basic`
   - `instagram_manage_insights`
   - `pages_read_engagement`
5. Click **Generate Access Token** and complete the Facebook login prompt.
6. Copy the token string — this is your **short-lived token** (valid ~1 hour).

---

### Step 5 — Extend to a Long-Lived Token (60 days)

Short-lived tokens expire quickly. Extend yours with this API call.

You need your **App ID** and **App Secret**, found at:
> App Dashboard → Settings → Basic

Make this GET request in your browser or with curl (replace the placeholders):

```
https://graph.facebook.com/v25.0/oauth/access_token
  ?grant_type=fb_exchange_token
  &client_id=YOUR_APP_ID
  &client_secret=YOUR_APP_SECRET
  &fb_exchange_token=YOUR_SHORT_LIVED_TOKEN
```

**Example with curl:**
```bash
curl "https://graph.facebook.com/v25.0/oauth/access_token?grant_type=fb_exchange_token&client_id=APP_ID&client_secret=APP_SECRET&fb_exchange_token=SHORT_TOKEN"
```

The response looks like:
```json
{
  "access_token": "EAABs...(long string)...",
  "token_type": "bearer",
  "expires_in": 5183944
}
```

Copy the `access_token` value — this is your **Long-Lived Token** (valid 60 days). Paste it into the PostForge sidebar.

---

### Step 6 — Get Your Instagram User ID

Your Instagram User ID is a long numeric string (e.g. `17841400000000001`). It is **not** your username and cannot be found in the app UI.

**Option A — Auto-detect (recommended):**  
In PostForge, paste your token and click **Auto-detect User ID**. PostForge will find the ID automatically via your Facebook Page connection.

**Option B — Manual lookup:**  
Make this API call with your long-lived token:

```
https://graph.facebook.com/v25.0/me/accounts
  ?fields=name,instagram_business_account
  &access_token=YOUR_LONG_LIVED_TOKEN
```

Look for the `instagram_business_account` object inside your Page entry:

```json
{
  "data": [{
    "name": "My Page",
    "instagram_business_account": {
      "id": "17841400000000001"    ← this is your Instagram User ID
    }
  }]
}
```

---

### Step 7 — Connect in PostForge

1. Run the app: `uv run streamlit run app.py`
2. Paste your **Long-Lived Access Token** into the sidebar
3. Enter your **Instagram User ID** (or click Auto-detect)
4. Click **Connect & Fetch Data**

---

## Token Refresh

Long-Lived tokens expire after **60 days**. When yours expires, repeat Step 5 — you don't need to redo the other steps. You can also refresh a non-expired token by calling the same exchange endpoint again with the existing long-lived token as `fb_exchange_token`.

---

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `[190] Invalid OAuth access token` | Token expired or malformed | Repeat Step 5 |
| `[100] Invalid parameter` or wrong User ID | User ID doesn't match token | Use Auto-detect |
| `[10] Permission denied` | Missing permission on token | Re-generate token with all three permissions |
| Auto-detect returns nothing | IG account not linked to a Facebook Page | Complete Step 2 |
| Insights show all zeros | Personal account, or posts older than 2 years | Switch to Business/Creator |

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (never commit your token).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the main file to `app.py`.
4. Run `uv export --format requirements-txt > requirements.txt` and commit it, or configure the platform to use uv.

Store your token in **Streamlit Secrets** (`.streamlit/secrets.toml`) rather than entering it in the UI each time — but never commit that file.

---

## Limitations

- **Rate limits:** The Graph API enforces per-app and per-user rate limits. PostForge adds deliberate delays between insight calls. Analyzing 300 posts takes several minutes.
- **Insights availability:** Post-level insights (reach, saves) may be unavailable for posts older than ~2 years. PostForge falls back to `(likes + comments) / followers` when insights are missing.
- **Reels/Stories:** Story insights are not included. Reels appear as VIDEO type.
- **Token expiry:** Long-lived tokens last 60 days. Refresh by repeating Step 5.
