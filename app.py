"""
PostForge — Instagram Analytics & Optimization Tool
Uses the official Meta Instagram Graph API (v22.0).
"""

import datetime
import os
import re
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
GRAPH_API_VERSION = "v22.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
INSIGHT_SLEEP = 0.5   # seconds between per-post insight calls

EMOJI_RE = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
CTA_KEYWORDS = [
    "click", "link", "bio", "shop", "buy", "order", "follow", "subscribe",
    "comment", "share", "save", "tag", "dm", "swipe", "check", "visit",
    "grab", "get", "discover", "learn", "watch", "join", "sign up",
]

_vader = SentimentIntensityAnalyzer()


# ── Graph API Layer ────────────────────────────────────────────────────────────
def _graph_get(path: str, token: str, **params) -> dict:
    """Make a Graph API GET request. Raises RuntimeError on API error."""
    r = requests.get(
        f"{GRAPH_BASE}/{path.lstrip('/')}",
        params={"access_token": token, **params},
        timeout=30,
    )
    data = r.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"[{err.get('code', '?')}] {err.get('message', 'Unknown API error')}")
    return data


def _graph_get_fields(path: str, token: str, fields: list[str]) -> dict:
    """Request a set of fields, automatically dropping any the API rejects."""
    remaining = list(fields)
    while remaining:
        try:
            return _graph_get(path, token, fields=",".join(remaining))
        except RuntimeError as e:
            m = re.search(r"nonexisting field \((\w+)\)", str(e))
            if m:
                remaining = [f for f in remaining if f != m.group(1)]
            else:
                raise
    return {}


def find_ig_accounts(token: str) -> tuple[list[dict], str, list[dict]]:
    """
    Find all Instagram Business/Creator accounts reachable from the token.
    Returns (accounts, diagnostic, raw_debug) where:
      - accounts: list of dicts with ig_user_id, ig_username, page_name
      - diagnostic: human-readable failure reason (empty on success)
      - raw_debug: raw per-page API responses for debugging
    """
    accounts: list[dict] = []
    raw_debug: list[dict] = []

    try:
        data = _graph_get("me/accounts", token, fields="id,name,access_token")
        pages = data.get("data", [])

        if not pages:
            return [], (
                "Your token is valid but **no Facebook Pages** were returned. "
                "The Facebook account you used to generate this token must manage "
                "at least one Page.\n\n"
                "Check **facebook.com/pages**, then regenerate the token from that "
                "same Facebook account."
            ), []

        for page in pages:
            page_id = page["id"]
            page_name = page.get("name", page_id)
            page_token = page.get("access_token") or token

            debug_entry: dict = {"page_id": page_id, "page_name": page_name, "results": {}}
            ig_id = None

            for field in ("instagram_business_account", "connected_instagram_account"):
                try:
                    resp = _graph_get(page_id, page_token, fields=field)
                    raw = resp.get(field)
                    debug_entry["results"][field] = raw
                    if isinstance(raw, dict) and raw.get("id") and not ig_id:
                        ig_id = raw["id"]
                except RuntimeError as e:
                    debug_entry["results"][field] = {"error": str(e)}

            # Third attempt: /page-id/instagram_accounts edge
            if not ig_id:
                try:
                    resp = _graph_get(f"{page_id}/instagram_accounts", page_token, fields="id,username")
                    items = resp.get("data", [])
                    debug_entry["results"]["instagram_accounts_edge"] = items
                    if items:
                        ig_id = items[0].get("id")
                except RuntimeError as e:
                    debug_entry["results"]["instagram_accounts_edge"] = {"error": str(e)}

            raw_debug.append(debug_entry)

            if not ig_id:
                continue

            try:
                ig_info = _graph_get_fields(ig_id, token, fields=["username", "name"])
            except Exception:
                ig_info = {}

            accounts.append({
                "ig_user_id": ig_id,
                "ig_username": ig_info.get("username", ""),
                "ig_name": ig_info.get("name", ""),
                "page_name": page_name,
            })

        if not accounts:
            page_names = ", ".join(f'**{p["page_name"]}**' for p in raw_debug)
            return [], (
                f"Found {len(pages)} Facebook Page(s) ({page_names}) but all three "
                "Instagram lookup methods returned empty. See the **Raw API debug** "
                "expander below for the exact API responses.\n\n"
                "**Most common cause:** the Instagram account is not connected at the "
                "Meta Business Suite level (even if it shows as linked in the Instagram "
                "app). Go to **business.facebook.com → Settings → Instagram Accounts** "
                "and add your account there."
            ), raw_debug

        return accounts, "", raw_debug

    except RuntimeError as e:
        return [], f"API error: {e}", raw_debug


def fetch_profile(token: str, user_id: str) -> dict:
    data = _graph_get_fields(
        user_id, token,
        fields=[
            "id", "username", "name", "biography",
            "followers_count", "follows_count", "media_count",
            "profile_picture_url", "website",
        ],
    )
    # `username` only exists on Instagram nodes. If it's missing the ID is a
    # Facebook User/Page ID, not an Instagram Business/Creator account ID.
    if not data.get("username"):
        raise RuntimeError(
            "The User ID you entered does not belong to an Instagram Business or Creator account. "
            "Make sure you are using the **Instagram User ID** (not a Facebook User or Page ID). "
            "Click **Auto-detect User ID** in the sidebar to find the correct value."
        )
    return {
        "user_id": data.get("id", user_id),
        "username": data.get("username", ""),
        "full_name": data.get("name", ""),
        "biography": data.get("biography", ""),
        "followers_count": data.get("followers_count", 0),
        "following_count": data.get("follows_count", 0),
        "media_count": data.get("media_count", 0),
        "profile_picture_url": data.get("profile_picture_url"),
        "website": data.get("website"),
    }


def _fetch_post_insights(media_id: str, media_type: str, token: str) -> dict:
    """Fetch per-post insights. Silently returns {} on any error."""
    metrics = "reach,saved,shares"
    if media_type == "VIDEO":
        metrics += ",video_views"
    try:
        ins = _graph_get(f"{media_id}/insights", token, metric=metrics)
        return {
            item["name"]: item["values"][0]["value"]
            for item in ins.get("data", [])
            if item.get("values")
        }
    except Exception:
        return {}


def fetch_all_media(
    token: str, user_id: str, limit: int, progress_bar, status_text
) -> tuple[list[dict], dict]:
    """
    Cursor-paginate through media and fetch per-post insights.
    Returns (medias, insights_map) where medias is a list of Graph API dicts.
    """
    fields = (
        "id,caption,media_type,media_url,thumbnail_url,timestamp,"
        "like_count,comments_count,permalink,"
        "children{id,media_type,media_url}"
    )

    status_text.text(f"Fetching your last {limit} posts…")
    progress_bar.progress(0.05)

    medias: list[dict] = []
    next_url: str | None = f"{GRAPH_BASE}/{user_id}/media"
    # First request passes params; subsequent requests use the full `next` URL
    req_params: dict | None = {"fields": fields, "limit": min(limit, 50), "access_token": token}

    while next_url and len(medias) < limit:
        if req_params:
            r = requests.get(next_url, params=req_params, timeout=30).json()
            req_params = None
        else:
            r = requests.get(next_url, timeout=30).json()

        if "error" in r:
            err = r["error"]
            msg = err.get("message", "")
            if "nonexisting field" in msg and "media" in msg:
                raise RuntimeError(
                    "Cannot access posts for this account. "
                    "This usually means the User ID is a Facebook Page ID rather than an "
                    "Instagram Business/Creator User ID — use the **Auto-detect User ID** "
                    "button to find the correct value. Also confirm your token includes the "
                    "`instagram_basic` permission."
                )
            raise RuntimeError(f"[{err.get('code')}] {msg}")


        batch = r.get("data", [])
        if not batch:
            break

        medias.extend(batch)
        progress_bar.progress(min(0.38, 0.05 + 0.33 * len(medias) / max(limit, 1)))
        next_url = r.get("paging", {}).get("next")

    medias = medias[:limit]
    progress_bar.progress(0.4)

    insights_map: dict[str, dict] = {}
    total = len(medias)
    for i, m in enumerate(medias):
        progress_bar.progress(0.4 + 0.6 * (i / max(total, 1)))
        status_text.text(f"Fetching insights for post {i + 1}/{total}…")
        insights_map[m["id"]] = _fetch_post_insights(m["id"], m.get("media_type", "IMAGE"), token)
        time.sleep(INSIGHT_SLEEP)

    return medias, insights_map


# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(medias: list[dict], insights_map: dict, followers: int) -> pd.DataFrame:
    rows = []
    for m in medias:
        caption = m.get("caption", "") or ""
        ts = pd.to_datetime(m["timestamp"])
        ins = insights_map.get(m["id"], {})
        media_type = m.get("media_type", "IMAGE")  # "IMAGE", "VIDEO", "CAROUSEL_ALBUM"

        likes = m.get("like_count", 0) or 0
        comments = m.get("comments_count", 0) or 0
        shares = ins.get("shares", 0) or 0
        saves = ins.get("saved", 0) or 0
        reach = ins.get("reach", 0) or 0
        views = ins.get("video_views", 0) or 0

        raw = likes + comments + shares + saves
        denom = max(reach if reach > 0 else followers, 1)

        hashtags = re.findall(r"#\w+", caption.lower())
        vs = _vader.polarity_scores(caption)
        children = m.get("children", {}).get("data", [])

        rows.append(
            {
                "id": m["id"],
                "permalink": m.get("permalink", ""),
                "timestamp": ts,
                "hour": ts.hour,
                "weekday": ts.day_of_week,
                "weekday_name": ts.day_name(),
                "is_weekend": ts.day_of_week >= 5,
                "month": ts.month,
                "year": ts.year,
                "media_type": media_type,
                "is_carousel": media_type == "CAROUSEL_ALBUM",
                "num_slides": len(children) if children else 1,
                "thumbnail_url": m.get("thumbnail_url") or m.get("media_url") or "",
                "caption": caption,
                "caption_length": len(caption),
                "word_count": len(caption.split()) if caption else 0,
                "has_question": "?" in caption,
                "emoji_count": len(EMOJI_RE.findall(caption)),
                "cta_score": sum(1 for kw in CTA_KEYWORDS if kw in caption.lower()),
                "hashtag_count": len(hashtags),
                "hashtags": hashtags,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
                "reach": reach,
                "views": views,
                "engagement_rate": round(raw / denom * 100, 4),
                "sentiment_compound": vs["compound"],
                "sentiment_label": (
                    "Positive" if vs["compound"] >= 0.05
                    else "Negative" if vs["compound"] <= -0.05
                    else "Neutral"
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


# ── UI Helpers ─────────────────────────────────────────────────────────────────
def _fmt(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def apply_theme() -> None:
    st.markdown(
        """
<style>
.stApp { background: linear-gradient(135deg,#0f0f1a 0%,#1a0a2e 100%); }

.kpi-card {
    background: linear-gradient(135deg,#1e1e3a,#2a1a4e);
    border: 1px solid #833AB4;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg,#833AB4,#FD1D1D,#FCAF45);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label { font-size: 0.82rem; color: #a0a0c0; margin-top: 4px; }

.post-card {
    background: linear-gradient(135deg,#1e1e3a,#2a1a4e);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}
.itag {
    display: inline-block;
    background: #833AB420;
    border: 1px solid #833AB4;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #c080ff;
    margin: 2px;
}
.sec-hdr {
    background: linear-gradient(90deg,#833AB4,#FD1D1D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 16px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _post_card_html(row: pd.Series, rank: int | None = None) -> str:
    preview = (row.caption[:120] + "…") if len(row.caption) > 120 else row.caption
    rank_html = f'<span style="color:#FCAF45;font-weight:700;">#{rank} </span>' if rank else ""
    no_cap = '<em style="color:#606080">No caption</em>'
    body = preview if preview else no_cap
    return (
        f'<div class="post-card">'
        f'<div style="font-size:.85rem;color:#a0a0c0;margin-bottom:6px;">'
        f'{rank_html}{row.timestamp.strftime("%b %d, %Y")} · {row.media_type}</div>'
        f'<div style="margin-bottom:8px;color:#e0e0f0;">'
        f"{body}</div>"
        f'<span class="itag">❤️ {_fmt(row.likes)}</span>'
        f'<span class="itag">💬 {_fmt(row.comments)}</span>'
        f'<span class="itag">📤 {_fmt(row.shares)}</span>'
        f'<span class="itag">🔖 {_fmt(row.saves)}</span>'
        f'<span class="itag">📈 {row.engagement_rate:.2f}%</span>'
        f"</div>"
    )


def _dark_chart(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c0c0e0",
    )
    st.plotly_chart(fig, use_container_width=True)


def _try_image(url: str | None) -> None:
    if url:
        try:
            st.image(url, use_container_width=True)
        except Exception:
            st.markdown("🖼️")


# ── Tab: Overview ──────────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame, profile: dict) -> None:
    st.markdown('<div class="sec-hdr">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("Total Posts", _fmt(len(df))),
        ("Avg Engagement Rate", f"{df.engagement_rate.mean():.2f}%"),
        ("Total Likes", _fmt(df.likes.sum())),
        ("Total Comments", _fmt(df.comments.sum())),
        ("Total Saves", _fmt(df.saves.sum())),
    ]
    for col, (label, val) in zip(cols, metrics):
        with col:
            kpi_card(label, val)

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        trend = df.sort_values("timestamp").copy()
        trend["7-post avg"] = trend["engagement_rate"].rolling(7, min_periods=1).mean()
        fig = px.line(
            trend, x="timestamp", y=["engagement_rate", "7-post avg"],
            title="Engagement Rate Over Time",
            labels={"value": "ER %", "timestamp": "", "variable": ""},
            color_discrete_map={"engagement_rate": "#833AB450", "7-post avg": "#FCAF45"},
            template="plotly_dark",
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        _dark_chart(fig)

    with c2:
        mt = df.groupby("media_type")["engagement_rate"].mean().reset_index()
        fig2 = px.bar(
            mt, x="media_type", y="engagement_rate",
            color="media_type",
            color_discrete_sequence=["#833AB4", "#FD1D1D", "#FCAF45"],
            title="Avg ER by Media Type",
            labels={"engagement_rate": "Avg ER %", "media_type": ""},
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False)
        _dark_chart(fig2)

    st.markdown("### 🏆 Top 5 Posts")
    top5 = df.nlargest(5, "engagement_rate").reset_index(drop=True)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        c1, c2 = st.columns([1, 4])
        with c1:
            _try_image(row.get("thumbnail_url") or row.get("media_url"))
        with c2:
            st.markdown(_post_card_html(row, rank=rank), unsafe_allow_html=True)
            st.markdown(
                f'<a href="{row.permalink}" target="_blank" '
                f'style="color:#833AB4;font-size:.8rem;">Open on Instagram ↗</a>',
                unsafe_allow_html=True,
            )
        st.markdown("")


# ── Tab: Post Explorer ─────────────────────────────────────────────────────────
def render_post_explorer(df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-hdr">🔍 Post Explorer</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        search = st.text_input("Search captions", placeholder="keyword…")
    with c2:
        types = ["All"] + sorted(df.media_type.unique().tolist())
        media_filter = st.selectbox("Media type", types)
    with c3:
        sort_by = st.selectbox(
            "Sort by", ["engagement_rate", "likes", "comments", "saves", "timestamp"]
        )

    filtered = df.copy()
    if search:
        filtered = filtered[filtered.caption.str.contains(search, case=False, na=False)]
    if media_filter != "All":
        filtered = filtered[filtered.media_type == media_filter]
    filtered = filtered.sort_values(sort_by, ascending=(sort_by == "timestamp")).reset_index(drop=True)

    st.caption(f"Showing {len(filtered)} of {len(df)} posts")
    st.dataframe(
        filtered[
            ["timestamp", "media_type", "engagement_rate", "likes", "comments", "shares", "saves", "reach"]
        ].rename(
            columns={
                "timestamp": "Posted", "media_type": "Type",
                "engagement_rate": "ER %", "likes": "Likes",
                "comments": "Comments", "shares": "Shares",
                "saves": "Saves", "reach": "Reach",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not filtered.empty:
        st.markdown("### Post Preview")
        idx = st.slider("Select post", 0, max(len(filtered) - 1, 0), 0)
        row = filtered.iloc[idx]
        c1, c2 = st.columns([1, 3])
        with c1:
            _try_image(row.get("thumbnail_url") or row.get("media_url"))
        with c2:
            st.markdown(f"**Posted:** {row.timestamp.strftime('%A, %B %d %Y at %H:%M')}")
            st.markdown(f"**Type:** {row.media_type}  |  **ER:** {row.engagement_rate:.3f}%")
            st.markdown(
                f"❤️ {row.likes} · 💬 {row.comments} · 📤 {row.shares} · 🔖 {row.saves}"
            )
            if row.caption:
                with st.expander("Full caption"):
                    st.write(row.caption)
            st.markdown(f"[Open on Instagram ↗]({row.permalink})")


# ── Tab: Content Analysis ──────────────────────────────────────────────────────
def render_content_analysis(df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-hdr">📈 Content Analysis</div>', unsafe_allow_html=True)

    st.markdown("#### ⏰ Best Posting Times")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hm = (
        df.groupby(["weekday_name", "hour"])["engagement_rate"]
        .mean()
        .reset_index()
        .pipe(lambda d: d.assign(weekday_name=pd.Categorical(d.weekday_name, day_order, ordered=True)))
        .pivot(index="weekday_name", columns="hour", values="engagement_rate")
    )
    fig = px.imshow(
        hm,
        labels={"x": "Hour", "y": "Day", "color": "Avg ER %"},
        color_continuous_scale=[[0, "#1a0a2e"], [0.5, "#833AB4"], [1, "#FCAF45"]],
        title="Engagement Rate Heatmap (Weekday × Hour)",
        aspect="auto",
        template="plotly_dark",
    )
    _dark_chart(fig)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### # Hashtag Performance")
        ht_rows = [
            {"hashtag": tag, "engagement_rate": row.engagement_rate}
            for _, row in df.iterrows()
            for tag in row.hashtags
        ]
        if ht_rows:
            ht = (
                pd.DataFrame(ht_rows)
                .groupby("hashtag")
                .agg(avg_er=("engagement_rate", "mean"), count=("engagement_rate", "count"))
                .reset_index()
                .query("count >= 2")
                .nlargest(20, "avg_er")
            )
            if not ht.empty:
                fig2 = px.bar(
                    ht, x="avg_er", y="hashtag", orientation="h",
                    color="avg_er",
                    color_continuous_scale=["#833AB4", "#FD1D1D", "#FCAF45"],
                    title="Top Hashtags by Avg ER",
                    labels={"avg_er": "Avg ER %", "hashtag": ""},
                    template="plotly_dark",
                )
                fig2.update_layout(
                    coloraxis_showscale=False,
                    showlegend=False,
                    yaxis={"categoryorder": "total ascending"},
                )
                _dark_chart(fig2)
            else:
                st.info("Need ≥2 posts per hashtag for analysis.")
        else:
            st.info("No hashtag data found.")

    with c2:
        st.markdown("#### 💬 Caption Sentiment")
        sent = df.sentiment_label.value_counts().reset_index()
        sent.columns = ["Sentiment", "Count"]
        cmap = {"Positive": "#4caf50", "Neutral": "#833AB4", "Negative": "#FD1D1D"}
        fig3 = px.pie(
            sent, names="Sentiment", values="Count",
            color="Sentiment", color_discrete_map=cmap,
            title="Sentiment Distribution", hole=0.5,
            template="plotly_dark",
        )
        _dark_chart(fig3)

        sent_er = df.groupby("sentiment_label")["engagement_rate"].mean().reset_index()
        fig4 = px.bar(
            sent_er, x="sentiment_label", y="engagement_rate",
            color="sentiment_label", color_discrete_map=cmap,
            title="Avg ER by Sentiment",
            labels={"engagement_rate": "Avg ER %", "sentiment_label": ""},
            template="plotly_dark",
        )
        fig4.update_layout(showlegend=False)
        _dark_chart(fig4)

    st.markdown("#### 📝 Caption Length vs Engagement")
    fig5 = px.scatter(
        df, x="caption_length", y="engagement_rate",
        color="media_type",
        trendline="lowess",
        labels={"caption_length": "Caption Length (chars)", "engagement_rate": "ER %"},
        title="Caption Length vs Engagement Rate",
        color_discrete_sequence=["#833AB4", "#FD1D1D", "#FCAF45"],
        template="plotly_dark",
    )
    _dark_chart(fig5)

    st.markdown("#### ☁️ Word Cloud — Top Performing Posts")
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud

        top_text = " ".join(df.nlargest(max(10, len(df) // 4), "engagement_rate").caption.dropna())
        top_text = re.sub(r"#\w+|http\S+", "", top_text)
        if top_text.strip():
            wc = WordCloud(
                width=800, height=280, background_color=None,
                mode="RGBA", colormap="cool", max_words=80,
            ).generate(top_text)
            fig_wc, ax = plt.subplots(figsize=(10, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_alpha(0)
            st.pyplot(fig_wc)
        else:
            st.info("Not enough caption text for word cloud.")
    except ImportError:
        st.info("Run `uv add wordcloud` to enable word cloud.")


# ── Tab: What Makes a Post Good? ───────────────────────────────────────────────
def render_what_makes_good(df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-hdr">⭐ What Makes a Post Good?</div>', unsafe_allow_html=True)

    threshold = df.engagement_rate.quantile(0.75)
    df = df.copy()
    df["is_top"] = df.engagement_rate >= threshold
    top = df[df.is_top]
    rest = df[~df.is_top]

    st.markdown(
        f"**Top performers:** ≥ **{threshold:.2f}%** engagement rate (top 25% of your posts).  "
        f"Comparing **{len(top)} top** vs **{len(rest)} other** posts."
    )
    st.markdown("---")

    st.markdown("#### 📊 Head-to-Head: Top vs Rest")
    features = [
        "likes", "comments", "shares", "saves",
        "caption_length", "word_count", "emoji_count",
        "cta_score", "hashtag_count", "sentiment_compound",
    ]
    rows_cmp = []
    for feat in [f for f in features if f in df.columns]:
        tm = top[feat].mean()
        rm = rest[feat].mean()
        ratio = tm / rm if rm > 0 else float("inf")
        rows_cmp.append(
            {
                "Feature": feat.replace("_", " ").title(),
                "Top 25%": f"{tm:.2f}",
                "Bottom 75%": f"{rm:.2f}",
                "Ratio": f"{ratio:.2f}×",
            }
        )
    st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

    if df.is_carousel.any():
        c_er = df[df.is_carousel].engagement_rate.mean()
        nc_er = df[~df.is_carousel].engagement_rate.mean()
        if nc_er > 0:
            mult = c_er / nc_er
            direction = "higher" if mult > 1 else "lower"
            st.info(
                f"🎠 **Carousel posts** achieve **{mult:.1f}× {direction}** average engagement "
                f"than single images/videos on your account."
            )

    st.markdown("---")
    st.markdown("#### 🎨 Engagement by Media Type")
    mt = df.groupby("media_type")["engagement_rate"].agg(["mean", "count"]).reset_index()
    mt.columns = ["Media Type", "Avg ER %", "Posts"]
    fig = px.bar(
        mt, x="Media Type", y="Avg ER %", color="Media Type", text="Posts",
        color_discrete_sequence=["#833AB4", "#FD1D1D", "#FCAF45", "#4caf50"],
        title="Average Engagement Rate by Media Type",
        template="plotly_dark",
    )
    fig.update_layout(showlegend=False)
    _dark_chart(fig)

    st.markdown("#### 🤖 ML Feature Importance")
    feat_cols = [
        "hour", "weekday", "is_weekend", "is_carousel",
        "caption_length", "word_count", "has_question",
        "emoji_count", "cta_score", "hashtag_count", "sentiment_compound",
    ]
    avail = [c for c in feat_cols if c in df.columns]
    if len(df) >= 20 and avail:
        try:
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            rf.fit(df[avail].fillna(0).astype(float), df["engagement_rate"])
            imp = pd.DataFrame(
                {
                    "Feature": [f.replace("_", " ").title() for f in avail],
                    "Importance": rf.feature_importances_,
                }
            ).sort_values("Importance")
            fig2 = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                title="Which factors predict your post's success?",
                color="Importance",
                color_continuous_scale=["#1a0a2e", "#833AB4", "#FCAF45"],
                template="plotly_dark",
            )
            fig2.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )
            _dark_chart(fig2)
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")
    else:
        st.info("Need ≥20 posts to run feature importance analysis.")

    st.markdown("#### 🏆 Examples: Your Winning Posts")
    for _, row in top.nlargest(5, "engagement_rate").iterrows():
        with st.expander(
            f"{row.timestamp.strftime('%b %d, %Y')} — ER: {row.engagement_rate:.2f}% — {row.media_type}"
        ):
            c1, c2 = st.columns([1, 3])
            with c1:
                _try_image(row.get("thumbnail_url") or row.get("media_url"))
            with c2:
                st.write(row.caption or "_No caption_")
                st.markdown(
                    f"❤️ {row.likes} · 💬 {row.comments} · 📤 {row.shares} · 🔖 {row.saves}"
                )
                st.markdown(f"[Open on Instagram ↗]({row.permalink})")


# ── Tab: Recommendations ───────────────────────────────────────────────────────
def render_recommendations(df: pd.DataFrame, profile: dict) -> None:
    st.markdown(
        '<div class="sec-hdr">🚀 Personalized Recommendations & Growth Plan</div>',
        unsafe_allow_html=True,
    )

    top = df.nlargest(max(len(df) // 4, 1), "engagement_rate")
    best_hour = df.groupby("hour")["engagement_rate"].mean().idxmax()
    best_day = df.groupby("weekday_name")["engagement_rate"].mean().idxmax()
    best_mt = df.groupby("media_type")["engagement_rate"].mean().idxmax()
    avg_cap_top = top.caption_length.mean()
    avg_ht_top = top.hashtag_count.mean()
    avg_ht_all = df.hashtag_count.mean()
    sent_er = df.groupby("sentiment_label")["engagement_rate"].mean()

    insights: list[str] = [
        f"⏰ **Optimal posting time:** {best_day}s around **{best_hour}:00–{best_hour + 1}:00**",
        f"🎨 **Best content type:** **{best_mt}** posts outperform all others on your account",
        f"📝 **Caption length sweet spot:** Your top posts average **{avg_cap_top:.0f} characters**",
        f"# **Hashtag count:** Top posts use ~**{avg_ht_top:.0f} hashtags** "
        f"(your overall avg is {avg_ht_all:.0f})",
    ]

    if df.is_carousel.any():
        c_er = df[df.is_carousel].engagement_rate.mean()
        nc_er = df[~df.is_carousel].engagement_rate.mean()
        if c_er > nc_er and nc_er > 0:
            insights.append(
                f"🎠 **Post more carousels:** They drive **{c_er / nc_er:.1f}× higher** ER for you"
            )

    cta_er = df[df.cta_score > 0].engagement_rate.mean()
    no_cta_er = df[df.cta_score == 0].engagement_rate.mean()
    if not np.isnan(cta_er) and cta_er > no_cta_er and no_cta_er > 0:
        insights.append(
            f"📣 **Use calls-to-action:** Posts with CTAs earn **{cta_er / no_cta_er:.1f}× more** engagement"
        )

    if not sent_er.empty:
        insights.append(
            f"💬 **Tone:** **{sent_er.idxmax()}** captions perform best on your account"
        )

    q_er = df[df.has_question].engagement_rate.mean()
    nq_er = df[~df.has_question].engagement_rate.mean()
    if not np.isnan(q_er) and q_er > nq_er and nq_er > 0:
        insights.append(
            f"❓ **Ask questions:** Captions with questions get **{q_er / nq_er:.1f}× more** engagement"
        )

    wk_er = df[df.is_weekend].engagement_rate.mean()
    wd_er = df[~df.is_weekend].engagement_rate.mean()
    if not (np.isnan(wk_er) or np.isnan(wd_er)) and min(wk_er, wd_er) > 0:
        better = "weekends" if wk_er > wd_er else "weekdays"
        mult = max(wk_er, wd_er) / min(wk_er, wd_er)
        insights.append(
            f"📅 **Day strategy:** Posting on **{better}** performs **{mult:.1f}× better** for you"
        )

    st.markdown("### 🎯 Smart Growth Plan")
    for i, line in enumerate(insights, 1):
        st.markdown(f"{i}. {line}")

    st.markdown("---")
    st.markdown("### 🤖 AI-Enhanced Recommendations _(Optional)_")
    st.caption(
        "Paste an Anthropic or OpenAI API key for richer, LLM-generated advice tailored to your data."
    )

    api_key = st.text_input("API Key", type="password", key="llm_key")
    provider = st.radio("Provider", ["Anthropic (Claude)", "OpenAI (GPT)"], horizontal=True)

    if api_key and st.button("✨ Generate AI Growth Plan"):
        context = (
            f"Instagram account analysis summary:\n"
            f"- Posts analyzed: {len(df)}\n"
            f"- Avg engagement rate: {df.engagement_rate.mean():.2f}%\n"
            f"- Followers: {profile.get('followers_count', 'N/A')}\n"
            f"- Best posting time: {best_day}s at {best_hour}:00\n"
            f"- Best content type: {best_mt}\n"
            f"- Top hashtag count: {avg_ht_top:.0f}\n"
            f"- Top caption length: {avg_cap_top:.0f} chars\n\n"
            f"Key findings: {'; '.join(insights[:5])}\n\n"
            f"Give 5 specific, prioritized, data-driven growth recommendations for this account."
        )
        with st.spinner("Generating…"):
            try:
                if "Anthropic" in provider:
                    import anthropic

                    client = anthropic.Anthropic(api_key=api_key)
                    msg = client.messages.create(
                        model="claude-opus-4-8",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": context}],
                    )
                    st.markdown(msg.content[0].text)
                else:
                    from openai import OpenAI

                    client = OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": context}],
                        max_tokens=1024,
                    )
                    st.markdown(resp.choices[0].message.content)
            except ImportError as e:
                st.error(f"Package missing: {e}. Run `uv add anthropic` or `uv add openai`.")
            except Exception as e:
                st.error(f"API error: {e}")


# ── Tab: Audience Insights ─────────────────────────────────────────────────────
def render_audience_insights(profile: dict, df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-hdr">👥 Audience Insights</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Followers", _fmt(profile.get("followers_count", 0)))
    with c2:
        kpi_card("Following", _fmt(profile.get("following_count", 0)))
    with c3:
        kpi_card("Total Posts", _fmt(profile.get("media_count", 0)))

    if profile.get("biography"):
        st.markdown(f"**Bio:** {profile['biography']}")
    if profile.get("website"):
        st.markdown(f"**Website:** {profile['website']}")

    st.markdown("---")
    st.markdown("#### 📅 Posting Activity — Last 90 Days")
    recent = df[df.timestamp >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)]
    if not recent.empty:
        activity = recent.groupby(recent.timestamp.dt.date).size().reset_index()
        activity.columns = ["Date", "Posts"]
        fig = px.bar(
            activity, x="Date", y="Posts",
            title="Posts per Day",
            color_discrete_sequence=["#833AB4"],
            template="plotly_dark",
        )
        _dark_chart(fig)
    else:
        st.info("No posts in the last 90 days.")

    st.markdown("#### 📆 Posting Frequency by Month")
    monthly = df.groupby(["year", "month"]).size().reset_index(name="posts")
    monthly["period"] = monthly.apply(
        lambda r: f"{int(r.year)}-{int(r.month):02d}", axis=1
    )
    fig2 = px.bar(
        monthly, x="period", y="posts",
        title="Posts per Month",
        color_discrete_sequence=["#FD1D1D"],
        template="plotly_dark",
        labels={"period": "Month", "posts": "Posts"},
    )
    _dark_chart(fig2)

    st.info(
        "Deep audience demographics (age, gender, location) require advanced Graph API "
        "permissions (`instagram_manage_insights`) and a verified Business account. "
        "The data above is derived from your own post history."
    )


# ── Tab: Export ────────────────────────────────────────────────────────────────
def render_export(df: pd.DataFrame, profile: dict) -> None:
    st.markdown('<div class="sec-hdr">📄 Export & Reporting</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### CSV Export")
        csv = df.drop(columns=["hashtags"], errors="ignore").to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Data (CSV)",
            data=csv,
            file_name=f"postforge_{profile.get('username', 'data')}_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.markdown("#### PDF Report")
        if st.button("📑 Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF…"):
                try:
                    pdf = _generate_pdf(df, profile)
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf,
                        file_name=f"postforge_report_{profile.get('username', 'report')}_{datetime.date.today()}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF error: {e}")


def _generate_pdf(df: pd.DataFrame, profile: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    purple = colors.HexColor("#833AB4")

    title_s = ParagraphStyle("T", parent=styles["Title"], textColor=purple, fontSize=22)
    h2_s = ParagraphStyle("H2", parent=styles["Heading2"], textColor=purple)
    body_s = styles["BodyText"]

    def _table(data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), purple),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#f4f4ff"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return t

    story = [
        Paragraph("PostForge Analytics Report", title_s),
        Paragraph(
            f"Account: @{profile.get('username', 'N/A')} · Generated {datetime.date.today()}",
            body_s,
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("Key Performance Indicators", h2_s),
        _table(
            [
                ["Metric", "Value"],
                ["Posts Analyzed", str(len(df))],
                ["Avg Engagement Rate", f"{df.engagement_rate.mean():.2f}%"],
                ["Total Likes", _fmt(df.likes.sum())],
                ["Total Comments", _fmt(df.comments.sum())],
                ["Total Saves", _fmt(df.saves.sum())],
                ["Followers", _fmt(profile.get("followers_count", 0))],
            ],
            col_widths=[10 * cm, 6 * cm],
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("Top 5 Posts by Engagement Rate", h2_s),
        _table(
            [["Date", "Type", "ER %", "Likes", "Comments"]]
            + [
                [
                    r.timestamp.strftime("%Y-%m-%d"),
                    r.media_type,
                    f"{r.engagement_rate:.2f}%",
                    str(r.likes),
                    str(r.comments),
                ]
                for _, r in df.nlargest(5, "engagement_rate").iterrows()
            ],
            col_widths=[4 * cm, 4 * cm, 3 * cm, 3 * cm, 3 * cm],
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("Key Findings", h2_s),
    ]

    best_hour = df.groupby("hour")["engagement_rate"].mean().idxmax()
    best_day = df.groupby("weekday_name")["engagement_rate"].mean().idxmax()
    best_mt = df.groupby("media_type")["engagement_rate"].mean().idxmax()
    top_cap = df.nlargest(max(len(df) // 4, 1), "engagement_rate").caption_length.mean()

    for line in [
        f"• Best posting time: {best_day}s at {best_hour}:00",
        f"• Best content type: {best_mt}",
        f"• Avg caption length in top posts: {top_cap:.0f} chars",
        f"• Top engagement rate: {df.engagement_rate.max():.2f}%",
    ]:
        story.append(Paragraph(line, body_s))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, str, int, bool, bool]:
    """Returns (token, user_id, limit, connect_clicked, find_id_clicked)."""
    st.sidebar.markdown(
        """
<div style="text-align:center;padding:16px 0;">
    <div style="font-size:2.2rem;">🔥</div>
    <div style="font-size:1.4rem;font-weight:700;
        background:linear-gradient(90deg,#833AB4,#FD1D1D,#FCAF45);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        PostForge
    </div>
    <div style="font-size:.75rem;color:#a0a0c0;">Instagram Analytics</div>
</div>
""",
        unsafe_allow_html=True,
    )

    profile = st.session_state.get("profile")

    if profile:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**@{profile.get('username', '')}**")
        st.sidebar.caption(
            f"👥 {_fmt(profile.get('followers_count', 0))} followers · "
            f"📸 {profile.get('media_count', 0)} posts"
        )
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
                st.session_state.pop("df", None)
                st.rerun()
        with c2:
            if st.sidebar.button("🚪 Log Out", use_container_width=True):
                for k in ("df", "profile", "token", "user_id"):
                    st.session_state.pop(k, None)
                st.rerun()
        st.sidebar.markdown("---")

    st.sidebar.markdown("### 🔐 Connect via Meta API")
    token = st.sidebar.text_input(
        "Long-Lived Access Token",
        type="password",
        key="sb_token",
        help="60-day token from the Meta Graph API Explorer. See README for setup steps.",
    )
    user_id = st.sidebar.text_input(
        "Instagram User ID",
        key="sb_user_id",
        help="Your numeric IG Business/Creator account ID. Click 'Auto-detect' to find it.",
        placeholder="e.g. 17841400000000001",
    )
    find_id = st.sidebar.button(
        "🔍 Auto-detect User ID",
        use_container_width=True,
        help="Looks up your IG User ID from the token via your connected Facebook Page.",
    )
    limit = st.sidebar.slider("Posts to analyze", 50, 300, 150, 25)
    connect = st.sidebar.button(
        "🔗 Connect & Fetch Data",
        use_container_width=True,
        type="primary",
        disabled=not (token and user_id),
    )

    return token, user_id, limit, connect, find_id


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="PostForge — Instagram Analytics",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    token, user_id, limit, connect, find_id = render_sidebar()

    # Auto-detect User ID from token
    if find_id and token:
        with st.spinner("Looking up connected Instagram accounts…"):
            accounts, diagnostic, raw_debug = find_ig_accounts(token)

        if len(accounts) == 1:
            acc = accounts[0]
            st.session_state["sb_user_id"] = acc["ig_user_id"]
            label = f"@{acc['ig_username']}" if acc["ig_username"] else acc["ig_user_id"]
            st.sidebar.success(f"Found **{label}** (ID: `{acc['ig_user_id']}`) — now click Connect.")
            st.rerun()
        elif len(accounts) > 1:
            st.sidebar.info(f"Found {len(accounts)} linked Instagram accounts. Click one to use it:")
            for acc in accounts:
                label = f"@{acc['ig_username']}" if acc["ig_username"] else acc["ig_user_id"]
                if acc["page_name"]:
                    label += f" (via Page: {acc['page_name']})"
                if st.sidebar.button(label, key=f"pick_{acc['ig_user_id']}"):
                    st.session_state["sb_user_id"] = acc["ig_user_id"]
                    st.rerun()
        else:
            st.sidebar.error(diagnostic or "No Instagram accounts found.")
            if raw_debug:
                with st.sidebar.expander("🔍 Raw API debug output"):
                    st.json(raw_debug)

    # Connect + fetch data
    if connect and token and user_id:
        user_id = user_id.strip()
        with st.spinner("Validating credentials…"):
            try:
                profile = fetch_profile(token, user_id)
            except RuntimeError as e:
                st.error(f"Connection failed: {e}")
                st.info(
                    "Common causes:\n"
                    "- Token has expired (60-day limit) — generate a new one\n"
                    "- Wrong User ID — use the Auto-detect button\n"
                    "- Missing `instagram_basic` permission on your token"
                )
                profile = None

        if profile:
            st.session_state["token"] = token
            st.session_state["user_id"] = user_id
            st.session_state["profile"] = profile

            progress = st.progress(0.0)
            status = st.empty()
            try:
                medias, insights_map = fetch_all_media(token, user_id, limit, progress, status)
            except RuntimeError as e:
                progress.empty()
                status.empty()
                st.session_state.pop("profile", None)
                st.error(str(e))
                medias = []

            if medias:
                progress.progress(1.0)
                status.empty()
                df = engineer_features(medias, insights_map, profile.get("followers_count", 1))
                st.session_state["df"] = df
                st.success(f"✅ Loaded {len(df)} posts for @{profile.get('username')}")

    df: pd.DataFrame | None = st.session_state.get("df")
    profile: dict = st.session_state.get("profile", {})

    if df is None or df.empty:
        st.markdown(
            """
<div style="text-align:center;padding:60px 20px 20px;">
    <div style="font-size:4rem;">🔥</div>
    <h1 style="background:linear-gradient(90deg,#833AB4,#FD1D1D,#FCAF45);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Welcome to PostForge
    </h1>
    <p style="color:#a0a0c0;font-size:1.1rem;max-width:540px;margin:0 auto 32px;">
        Connect your Instagram account to unlock deep analytics, discover what content
        performs best, and get a personalized, data-driven growth plan.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Quick Setup Guide")
        with st.expander("Step 1 — Switch to a Professional Account", expanded=False):
            st.markdown(
                "In the Instagram app: **Settings → Account → Switch to Professional Account**.  \n"
                "Choose **Creator** or **Business**."
            )
        with st.expander("Step 2 — Link Instagram to a Facebook Page", expanded=False):
            st.markdown(
                "In the Instagram app: **Settings → Account → Sharing to other apps → Facebook**.  \n"
                "Connect to an existing Facebook Page (or create one — it can be a placeholder)."
            )
        with st.expander("Step 3 — Create a Meta Developer App", expanded=False):
            st.markdown(
                "1. Go to **developers.facebook.com** → **My Apps** → **Create App**\n"
                "2. Choose **Business** as the app type\n"
                "3. Under **Products**, add **Instagram Graph API**"
            )
        with st.expander("Step 4 — Get a Long-Lived Access Token", expanded=False):
            st.markdown(
                "**4a. Short-lived token** — open the [Graph API Explorer]"
                "(https://developers.facebook.com/tools/explorer/), select your app, "
                "add permissions `instagram_basic`, `instagram_manage_insights`, "
                "`pages_read_engagement`, then click **Generate Access Token**.\n\n"
                "**4b. Extend to 60 days** — make this API call (replace placeholders):\n"
                "```\n"
                f"GET https://graph.facebook.com/{GRAPH_API_VERSION}/oauth/access_token\n"
                "  ?grant_type=fb_exchange_token\n"
                "  &client_id=YOUR_APP_ID\n"
                "  &client_secret=YOUR_APP_SECRET\n"
                "  &fb_exchange_token=SHORT_LIVED_TOKEN\n"
                "```\n"
                "Copy the `access_token` from the JSON response into the sidebar."
            )
        with st.expander("Step 5 — Get Your Instagram User ID", expanded=False):
            st.markdown(
                "Paste your token in the sidebar and click **Auto-detect User ID** — PostForge "
                "will find it automatically via your connected Facebook Page.\n\n"
                "Or find it manually:\n"
                "```\n"
                f"GET https://graph.facebook.com/{GRAPH_API_VERSION}/me/accounts\n"
                "  ?fields=instagram_business_account\n"
                "  &access_token=YOUR_TOKEN\n"
                "```\n"
                "Look for `instagram_business_account.id` in the response."
            )
        return

    tabs = st.tabs(
        [
            "📊 Overview",
            "🔍 Post Explorer",
            "📈 Content Analysis",
            "⭐ What Makes a Post Good?",
            "🚀 Recommendations",
            "👥 Audience Insights",
            "📄 Export",
        ]
    )

    with tabs[0]:
        render_overview(df, profile)
    with tabs[1]:
        render_post_explorer(df)
    with tabs[2]:
        render_content_analysis(df)
    with tabs[3]:
        render_what_makes_good(df)
    with tabs[4]:
        render_recommendations(df, profile)
    with tabs[5]:
        render_audience_insights(profile, df)
    with tabs[6]:
        render_export(df, profile)


if __name__ == "__main__":
    main()
