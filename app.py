import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import textstat

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

STATS = {
    "Facebook": {
        "mean": np.array([6.88875, -15.38, 69.43, 5.259, 1.378, 14.26,
                          0.4510, 2.168, 0.597, 24.27]),
        "std":  np.array([8.3354, 92.22, 72.74, 10.404, 1.617, 5.514,
                          0.4947, 0.8824, 0.491, 18.97]),
        "beta": np.array([0.1409, 0.0897, 0.2486, -0.1203, -0.2412,
                          -0.0176, -0.0130, 0.0629, 0.0842, -0.0242]),
        "intercept": -1.4364,
        "auc": 0.5548,
    },
    "Instagram": {
        "mean": np.array([6.3605, 36.06, 36.31, 10.348, 2.799, 13.400,
                          0.3731, 2.562, 0.559, 29.38]),
        "std":  np.array([10.58, 46.19, 37.66, 10.23, 1.894, 7.431,
                          0.4554, 0.7033, 0.4968, 23.95]),
        "beta": np.array([0.1473, 0.6257, 0.5144, -0.3663, 0.2376,
                          -0.0774, -0.1812, 0.1405, 0.0228, -0.2432]),
        "intercept": -1.4296,
        "auc": 0.5112,
    },
    "TikTok": {
        "mean": np.array([2.4625, 52.59, 12.52, 7.797, 3.556, 15.378,
                          0.2054, 1.015, 0.249, 24.11]),
        "std":  np.array([7.152, 37.41, 15.74, 6.642, 1.306, 5.243,
                          0.4094, 0.259, 0.433, 28.77]),
        "beta": np.array([0.2282, 0.0520, 0.2382, 0.0673, 0.1831,
                           0.1867, -0.0012, -0.2097, -0.0868, -0.0439]),
        "intercept": -1.4383,
        "auc": 0.6475,
    },
}

ORDER = [
    "Keyword Density (%)", "Readability Score", "Caption Length",
    "Hashtag Count", "Hashtag Type", "Post Timing", "Sentiment Score",
    "Caption Format", "CTA Presence", "Emphasis Style (%)"
]

ADVICE = {
    "Keyword Density (%)": {
        "pos": "Higher keyword density tends to support visibility. Try weaving in your main phrase once more — aim for at least 2% without sounding repetitive.",
        "neg": "Keyword usage may be a bit heavy here. Consider removing one instance or surrounding it with more context for clarity and flow."
    },
    "Readability Score": {
        "pos": "Clear, readable captions see slightly better outcomes. Shorten long sentences, space out emojis, and keep structure gentle on the eyes.",
        "neg": "This caption is already very readable. Adding complexity likely won’t help — better to focus on clarity of message."
    },
    "Caption Length": {
        "pos": "A bit of added detail could lift visibility. Consider a short second line, a benefit phrase, or a structured CTA if it adds value.",
        "neg": "Concise is key here. See if a line or two can be trimmed — or shift details to the image or carousel instead."
    },
    "Hashtag Count": {
        "pos": "Hashtags seem to support visibility on this platform. You might add 1–2 well-matched tags (balancing niche and generic).",
        "neg": "This platform tends to penalize over-tagging. Stay under 8 total tags, and avoid copy-paste bundles."
    },
    "Hashtag Type": {
        "pos": "A balanced mix of generic and niche tags performs well. One branded or niche tag paired with a trending generic one can help discoverability.",
        "neg": "The current mix may be too tilted toward one side. Try sticking purely to generic tags, or keep a clean niche focus depending on your goal."
    },
    "Post Timing": {
        "pos": "Later post times are acceptable here. Focus more on what you say than the hour you post — content leads the way.",
        "neg": "Timing matters less on this platform. If you’ve crafted a strong caption, don’t stress too much about the exact hour."
    },
    "Sentiment Score": {
        "pos": "Emotion adds lift. A subtle emoji, a hopeful phrase, or a light-hearted tone can round out the caption nicely.",
        "neg": "Neutral isn’t bad here. If your tone feels clear and appropriate, there’s no need to force emotional cues."
    },
    "Caption Format": {
        "pos": "Structure supports scanning. Bullet points, line breaks, or mini-sections can help your message land faster.",
        "neg": "Dense blocks may overwhelm here. Try condensing into a punchy one-liner or adding breaks for clarity."
    },
    "CTA Presence": {
        "pos": "One clear call-to-action works well. Ending with 'Shop now', 'Learn more', or a soft prompt can give direction without pressure.",
        "neg": "Hard CTAs might not perform as well. Consider ending with a question, thought, or gentle nudge instead."
    },
    "Emphasis Style (%)": {
        "pos": "A little emphasis — like one emoji or a standout word — draws attention. Think of it as seasoning, not the whole dish.",
        "neg": "Heavy styling may lower visibility. Consider softening ALL CAPS, emoji repeats, or decorative symbols for a cleaner feel."
    }
}

import unicodedata, re, itertools, nltk
from nltk.corpus import words as nltk_words, wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def safe_read_csv(path: Path, **kwargs):
    if not path.exists():
        st.warning(f"Missing file: `{path.name}` (expected in `data/`). Feature depending on it is skipped.")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.error(f"Error reading {path.name}: {e}")
        return None

CAPTION_DATA_FILES = {
    "Facebook":  "Caption Science - Facebook F.csv",
    "Instagram": "Caption Science - Instagram F.csv",
    "TikTok":    "Caption Science - TikTok F.csv",
}

SCORE_FILES = {
    "Facebook":  "Caption Science - Facebook Caption Score.csv",
    "Instagram": "Caption Science - Instagram Caption Score.csv",
    "TikTok":    "Caption Science - TikTok Caption Score.csv",
}

Z_FILES = {
    "Facebook":  "facebook_z.csv",
    "Instagram": "instagram_z.csv",
    "TikTok":    "tiktok_z.csv",
}

VAL_PROB_FILES = {  
    "Facebook":  "fb_val_probs.csv",
    "Instagram": "ig_val_probs.csv",
    "TikTok":    "tk_val_probs.csv",
}

RF_PROB_FILES = {
    "Facebook":  "facebook_rf_val_probs.csv",
    "Instagram": "instagram_rf_val_probs.csv",
    "TikTok":    "tiktok_rf_val_probs.csv",
}

DT_PROB_FILES = {
    "Facebook":  "facebook_dt_val_probs.csv",
    "Instagram": "instagram_dt_val_probs.csv",
    "TikTok":    "tiktok_dt_val_probs.csv",
}

@st.cache_data(show_spinner=False)
def load_caption_raw(platform: str) -> pd.DataFrame | None:
    path = DATA_DIR / CAPTION_DATA_FILES[platform]
    df = safe_read_csv(path)
    if df is None:
        return None
    if "Emphasis Style (%)" in df.columns:
        df["Emphasis Style (%)"] = df["Emphasis Style (%)"].astype(str).str.replace("%", "", regex=False)

    for col in ORDER:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def platform_matrix(plat: str):
    path = DATA_DIR / Z_FILES[plat]
    df = safe_read_csv(path)
    if df is None or "Search Rank" not in df.columns:
        return None, None
    ranks = df["Search Rank"].to_numpy()
    Z = df.drop(columns=["Search Rank"]).to_numpy()
    return Z, ranks

@st.cache_data(show_spinner=False)
def load_score_file(plat: str):
    return safe_read_csv(DATA_DIR / SCORE_FILES[plat])

@st.cache_data(show_spinner=False)
def load_prob_file(mapping: dict, plat: str):
    return safe_read_csv(DATA_DIR / mapping[plat])

@st.cache_resource(show_spinner=False)
def _nltk_ready():
    nltk.download("words", quiet=True)
    nltk.download("wordnet", quiet=True)
_nltk_ready()

english = set(nltk_words.words())
english.update(wn.words())

_base = [
    "Local streetwear brands PH", "Popular clothing brands Philippines",
    "Fashion trends PH", "Trendy outfits Philippines",
    "Shop online clothing PH", "Street fashion Manila",
    "Where to buy trendy clothes PH", "Online clothing stores PH",
    "Filipino fashion brands", "PH clothing brands online",
]
_qw = {w.lower() for p in _base for w in p.split()}
GENERIC = {f"#{w}" for w in _qw}
for a, b in itertools.combinations(_qw, 2):
    GENERIC.update({f"#{a}{b}", f"#{b}{a}"})
GENERIC.update({"#fyp", "#foryou", "#foryoupage"})

def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def clean_and_tokenize(caption: str):
    out = []
    for raw in caption.split():
        raw = re.sub(r"^[^\w@#]+", "", raw)
        if raw.startswith(('#', '@')):
            continue
        word = re.sub(r"[^\w\s]", "", ascii_fold(raw))
        if word:
            out.append(word.lower())
    return out

def keyword_density_and_len(cap: str, query: str):
    toks = clean_and_tokenize(cap)
    if not toks:
        return 0.0, 0
    kws = [ascii_fold(w).lower() for w in query.split()]
    hits = sum(t in kws for t in toks)
    return round(hits / len(toks) * 100, 2), len(toks)

def readability_score(caption: str):
    text = " ".join(clean_and_tokenize(caption))
    return round(textstat.flesch_reading_ease(text), 2) if text else 0

def hashtag_count_and_type(txt: str):
    tags = re.findall(r"#\w+", (txt or "").lower())
    if not tags:
        return 0, None
    g = sum(1 for t in tags if t in GENERIC or re.fullmatch(r"#fypp+", t) or t[1:] in english)
    n = len(tags) - g
    tag_type = 1 if g == len(tags) else 2 if n == len(tags) else 3 if g == n else 4 if n > g else 5
    return len(tags), tag_type

analyser = SentimentIntensityAnalyzer()

def sentiment(txt: str):
    txt = unicodedata.normalize("NFKD", txt or "")
    txt = re.sub(r"(@\w+|#\w+|http\S+)", "", txt)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"“”‘’…-–—()[]{}")
    txt = "".join(c for c in txt if c in allowed)
    return round(analyser.polarity_scores(txt)["compound"], 3) if txt else 0.0

def emphasis_pct(cap: str):
    if not cap:
        return 0.0
    txt = re.sub(r"(@\w+|#\w+|\s)", "", cap)
    emph = 0
    for ch in txt:
        is_stylised = (ord(ch) > 127 and ch.isalpha())
        is_symb_emo = ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,"
        is_caps = ch.isupper() and ch.lower() != ch
        if is_stylised or is_symb_emo or is_caps:
            emph += 1
    return round(emph / len(txt) * 100, 2)

def sigmoid(x): return 1 / (1 + math.exp(-x))

def caption_score(vals: dict, plat: str):
    stx = STATS[plat]
    raw = np.array([vals[k] for k in ORDER], float)
    z = (raw - stx["mean"]) / stx["std"]
    logit = stx["intercept"] + np.dot(stx["beta"], z)
    return logit, sigmoid(logit) * 100

st.set_page_config("Caption Science", "🧑🏻‍💻", layout="centered")
st.title("[🧪 Caption Science – instant caption quantifier](https://docs.google.com/document/d/1Yh2aES8NQLnA74r7HV3-FGM767_dpF8qo6tjUK7AYRQ/edit?usp=sharing)")

with st.sidebar:
    st.markdown("### Data folders")
    st.write(f"`data/` exists: {DATA_DIR.exists()}")
    st.write(f"`assets/` exists: {ASSETS_DIR.exists()}")
    st.caption("If a file is missing you'll see a warning in the main area.")

st.header("1. Analyze a caption")

colL, colR = st.columns([3, 2])
with colL:
    platform = st.selectbox("Platform", ["Instagram", "Facebook", "TikTok"])
    caption = st.text_area("Caption", height=160, placeholder="Paste or type your caption here...")
    hashtags = st.text_input("Hashtags (optional)")
    keywords = st.text_input("Target Keywords / Query")
    fmt = st.radio("Format", ["One-liner", "Paragraph", "Bullet Points"], horizontal=True)
    cta = st.radio("CTA present?", ["Yes", "No"], horizontal=True)
    post_t = st.time_input("Planned post time (local)")
with colR:
    st.markdown("**What this does**")
    st.write(
        "- Extracts 10 attributes (density, readability, length, hashtags, etc.)\n"
        "- Standardises them vs platform averages\n"
        "- Applies a simple logistic model → *Caption Score* & probability\n"
        "- Lets you inspect how you differ from typical top results"
    )
    st.info("Prediction power is limited (AUC near 0.5–0.65). Treat outputs as exploratory, not definitive.")

if st.button("Analyze"):
    dens, len_ = keyword_density_and_len(caption, keywords) if keywords else (0.0, len(clean_and_tokenize(caption)))
    rd = readability_score(caption)
    h_n, h_t = hashtag_count_and_type(f"{caption} {hashtags}")
    sent = sentiment(caption)
    fmt_n = {"One-liner": 1, "Paragraph": 2, "Bullet Points": 3}[fmt]
    cta_n = 1 if cta == "Yes" else 0
    hr = post_t.hour + post_t.minute / 60 if post_t else STATS[platform]["mean"][5]
    emph = emphasis_pct(caption)

    vals = {"Keyword Density (%)": dens, "Readability Score": rd, "Caption Length": len_,
            "Hashtag Count": h_n, "Hashtag Type": h_t or 0, "Post Timing": hr,
            "Sentiment Score": sent, "Caption Format": fmt_n,
            "CTA Presence": cta_n, "Emphasis Style (%)": emph}
    st.markdown("### Extracted attributes")
    st.dataframe({k: [v] for k, v in vals.items()}, hide_index=True)

    logit, prob = caption_score(vals, platform)
    st.markdown(f"### 📊 Caption Score (log-odds): **{logit:.2f}**")
    st.markdown(f"### 🎯 Estimated Top-20 probability: **{prob:.1f}%**")
    st.caption(
        "⚠️ Disclaimer: This is an estimate based on limited predictive power. For a full breakdown of how we tried to predict visibility from captions — including the models, results, and why these values should be interpreted with caution — see 🎞️ Story-telling: The Full Journey of Caption Science below."
    )

    auc = STATS[platform]["auc"]

    with st.expander("Model reliability (AUC): For Transparency"):
        st.caption(f"AUC: {auc:.3f}  (1 = perfect, 0.5 = chance)")

        roc_img = f"assets/roc_{platform.lower()}.png"
        if os.path.exists(roc_img):
            st.image(roc_img, caption=f"{platform} ROC • AUC {auc:.3f}")
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot([0, 1], [0, 1], "--", lw=1, label="Chance")
            ax.scatter(1 - auc, auc, color="black")
            ax.set_xlabel("False-positive rate (FPR)")
            ax.set_ylabel("True-positive rate (TPR)")
            ax.set_title(f"{platform} ROC (AUC {auc:.3f})", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=.3, lw=.5)
            st.pyplot(fig)

        st.markdown(
            f"""
            **How to read this:** the ROC curve shows the trade-off between
            catching true *Top-20* captions (TPR) and accidentally flagging
            lower-rank captions as *Top-20* (FPR).  
            An AUC of **{auc:.3f}** means the model orders captions correctly
            about **{auc * 100:.0f}%** of the time — better than random (0.50), but
            not a crystal ball. That’s why most probabilities you see stay in the
            mid-range rather than shooting straight to 90 %.
            """
        )
    from pathlib import Path  

    with st.expander("🔍 Where captions like yours actually appear"):
        df_path = {
            "Instagram": DATA_DIR / "Caption Science - Instagram Caption Score.csv",
            "Facebook": DATA_DIR / "Caption Science - Facebook Caption Score.csv",
            "TikTok": DATA_DIR / "Caption Science - TikTok Caption Score.csv",
        }[platform]

        if df_path.exists():
            df = pd.read_csv(df_path)
            fig, ax = plt.subplots(figsize=(4, 2.8))

            sns.scatterplot(
                data=df, x="Search Rank", y="Caption Score", ax=ax,
                s=15, color="gray", alpha=0.5
            )

            ax.axhline(logit, color="red", lw=2)

            ax.set_xlabel("Search Rank")
            ax.set_ylabel("Caption Score")
            ax.set_title(f"{platform}: Score vs Actual Rank")
            ax.invert_xaxis()  

            st.pyplot(fig)
            st.caption(
                "This shows where captions with similar scores to yours tend to appear in actual search ranks. "
                "The red line is your caption score. While high scores lean toward higher ranks, high scores also show up on lower ranks."
            )
        else:
            st.warning(f"Missing score file: {df_path.name}")

    tweaks = []
    if keywords and dens < 2 and STATS[platform]["beta"][0] > 0:
        tweaks.append("Add your main keyword once – density is very low.")
    if platform in ("Facebook", "Instagram") and h_n > 8:
        tweaks.append("Trim hashtags; > 8 shows slight downward trend.")
    if rd and rd < 30:
        tweaks.append("Rewrite a bit for clarity – readability < 30 is hard to scan.")
    if tweaks:
        st.markdown("#### ✏️ Quick tweaks")
        for tip in tweaks[:3]:
            st.write(f"- {tip}")
    else:
        st.info("No obvious tweaks – values within typical ranges.")

    st.session_state["last_run"] = {
        "platform": platform,
        "vals": vals,
        "logit": logit,
        "prob": prob,
    }

if st.button("🔍 Explain my caption"):
    if "last_run" not in st.session_state:
        st.error("Run **Analyze** first 🙂")
    else:
        platform = st.session_state["last_run"]["platform"]
        vals = st.session_state["last_run"]["vals"]
        beta, mu, sd = STATS[platform]["beta"], STATS[platform]["mean"], STATS[platform]["std"]
        raw = np.array([vals[k] for k in ORDER])
        z = (raw - mu) / sd
        imp = beta * z

        df = pd.DataFrame({"Attribute": ORDER,
                           "Your value": raw.round(2),
                           "z-score": z.round(2),
                           "β": beta.round(3),
                           "β×z": imp.round(3)})
        st.markdown("### 🕵️‍♂️ Score breakdown")
        st.dataframe(df, hide_index=True, use_container_width=True)

        pos_i, neg_i = int(np.argmax(imp)), int(np.argmin(imp))
        st.caption(
            f"🔍 **Biggest boost:** {ORDER[pos_i]} (+{imp[pos_i]:.2f})  "
            f"🪨 **Biggest drag:** {ORDER[neg_i]} ({imp[neg_i]:.2f})"
        )

        drag_attr = ORDER[neg_i]
        tip_key = "pos" if beta[neg_i] > 0 else "neg"
        advice = ADVICE.get(drag_attr, {}).get(tip_key)

        if advice:
            st.markdown("#### 📈 Boost your caption’s impact",
                        help="This advice targets the biggest drag pulling your score down.")

            st.info(
                f"**{drag_attr}**\n\n{advice}",
                icon="💡"
            )

        st.markdown(
            "**β** = model weight  \n"
            "*z*-score = distance from platform mean  \n"
            "**β×z** = contribution to Caption Score.")

if st.checkbox("📊 Where my caption stands?"):

    if "last_run" not in st.session_state:
        st.error("Run **Analyze** first so we have numbers to plot 🙂")
        st.stop()

    import matplotlib.pyplot as plt, seaborn as sns

    plat  = st.session_state["last_run"]["platform"]
    vals  = st.session_state["last_run"]["vals"]
    raw   = np.array([vals[k] for k in ORDER])
    mu,sd = STATS[plat]["mean"], STATS[plat]["std"]
    user_z= (raw-mu)/sd
    Z,ranks = platform_matrix(plat)
    if Z is None or ranks is None:
        st.warning("Missing z-score file for this platform; dashboard disabled.")
        st.stop()

    st.markdown("### 📈 Optimize for the Rank You Want")
    top_x = st.number_input(
        "🎯 Which top ranks are you targeting?",
        min_value=1, max_value=100, value=20, step=1,
        help="This defines your goal: Posts ranked 1 to this number form the benchmark group you're optimizing for."
    )
    show_rest = st.checkbox("Overlay the other ranks", value=False)

    top_mask   = ranks<=top_x
    bench_Z    = Z[top_mask]
    others_Z   = Z[~top_mask]

    st.markdown("### 🔭 Distribution check (z-scores)")

    PAIR_ROWS = [
        ("Keyword Density (%)", "Readability Score"),
        ("Caption Length", "Hashtag Count"),
        ("Hashtag Type", "Post Timing"),
        ("Sentiment Score", "Caption Format"),
        ("CTA Presence", "Emphasis Style (%)"),
    ]

    import seaborn as sns

    for row in PAIR_ROWS:
        cols = st.columns(2)
        for i, attr in enumerate(row):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(3.2, 2.6))

                sns.kdeplot(
                    bench_Z[:, ORDER.index(attr)],
                    ax=ax, lw=2, color="#4c78a8", label=f"Top {top_x}"
                )

                if show_rest and others_Z.size:
                    sns.kdeplot(
                        others_Z[:, ORDER.index(attr)],
                        ax=ax, lw=2, color="#f58518", linestyle="--",
                        label=f"{top_x + 1}–100"
                    )

                ax.axvline(user_z[ORDER.index(attr)], color="red", lw=2)

                ax.set_title(attr, fontsize=9)
                ax.set_xlim(-3.5, 3.5)
                ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
                ax.set_xlabel("z-score")
                ax.set_yticks([])

                if (row is PAIR_ROWS[0]) and (i == 0) and show_rest:
                    ax.legend(fontsize=6, frameon=False)

                st.pyplot(fig, clear_figure=True)

    st.caption(
        f"**How to read:** each curve shows where most captions land on this attribute. "
        f"Blue = Top {top_x}, orange = the rest (if shown). Your caption is the red line. "
        f"If your score sits in the thickest part of the **blue curve**, you're similar to the Top {top_x}. "
        f"But if it also overlaps with the **orange**, you're not uniquely high-performing. "
        f"**So explore both sides before optimizing!**"
    )

    score_df = load_score_file(plat)
    if score_df is not None and "Caption Score" in score_df.columns:
        fig_s, ax_s = plt.subplots(figsize=(3.6, 2.4))
        sc_top = score_df.loc[top_mask, "Caption Score"].dropna()
        sc_rest = score_df.loc[~top_mask, "Caption Score"].dropna()
        ax_s.hist(sc_top, bins=25, alpha=.6, color="#4c78a8", label=f"Top {top_x}")
        if show_rest and sc_rest.size:
            ax_s.hist(sc_rest, bins=25, alpha=.4, color="#f58518", label=f"{top_x + 1}–100")
        ax_s.axvline(st.session_state["last_run"]["logit"], color="red", lw=2)
        ax_s.set_xlabel("Caption Score (log-odds)")
        ax_s.set_ylabel("Frequency")
        ax_s.set_title(f"{plat}: Caption-Score distribution")
        if show_rest:
            ax_s.legend(fontsize=7, frameon=False)
        st.pyplot(fig_s, clear_figure=True)
        st.caption(
            f"**How to read:** This shows how your caption's score (red line) compares to real posts. "
            f"Higher scores (right side) mean the model sees your caption as stronger. "
            f"But a strong score doesn’t always mean similarity to Top-{top_x} captions. "
            f"Use the overlay to check if optimizing for Top-{top_x} might also overlap with "
            f"captions from ranks {top_x + 1}–100. Aim wisely!"
        )

    st.markdown("### 🌐 Overall shape")
    theta = np.linspace(0, 2 * np.pi, len(ORDER), endpoint=False)
    fig_r, ax_r = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4, 4))
    bench_m = bench_Z.mean(axis=0)
    ax_r.plot(np.append(theta, theta[0]), np.append(bench_m, bench_m[0]),
              lw=1.5, color="#4c78a8", label=f"Top {top_x}")
    if show_rest and others_Z.size:
        rest_m = others_Z.mean(axis=0)
        ax_r.plot(np.append(theta, theta[0]), np.append(rest_m, rest_m[0]),
                  lw=1.5, color="#f58518", label=f"{top_x + 1}-100")
    ax_r.plot(np.append(theta, theta[0]), np.append(user_z, user_z[0]),
              color="red", lw=2.2, marker="o", label="you")
    ax_r.fill_between(np.append(theta, theta[0]), np.append(user_z, user_z[0]),
                      color="red", alpha=.15)
    ax_r.set_xticks(theta);
    ax_r.set_xticklabels(ORDER, fontsize=7)
    ax_r.set_yticks([-2, 0, 2]);
    ax_r.set_yticklabels(["−2σ", "0", "+2σ"])
    ax_r.legend(fontsize=7, frameon=False, loc="upper right")
    st.pyplot(fig_r, clear_figure=True)
    st.caption(
        "**How to read:** Each point shows how your caption compares to the average. "
        "Closer to the center = more typical. Closer to the edge = more extreme. "
        f"The blue shape shows the average for Top-{top_x} captions, **use it as a guide**. "
        "If your point is farther *out* than the blue, try softening that trait. "
        "If it’s *inside* the blue, consider boosting it a little. "
        f"But always check the overlay, maybe the Top {top_x + 1}–100 look similar, too!"
    )

    st.markdown("### 🎯 What if I tweak just one attribute?")
    slice_attr = st.selectbox("Select attribute to isolate", ORDER,
                              index=ORDER.index("Keyword Density (%)"))
    idx = ORDER.index(slice_attr)
    beta = STATS[plat]["beta"][idx];
    intrc = STATS[plat]["intercept"]
    z_grid = np.linspace(-3, 3, 200)
    prob_g = 1 / (1 + np.exp(-(intrc + beta * z_grid)))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(z_grid, prob_g * 100, label="model sigmoid")
    user_p = 1 / (1 + np.exp(-(intrc + beta * user_z[idx]))) * 100
    ax.scatter(user_z[idx], user_p, color="red", zorder=5, label="you")
    ax.set_xlabel(f"{slice_attr} (z)");
    ax.set_ylabel("Predicted Top-20 %")
    ax.set_title(f"{plat}: influence of {slice_attr}", fontsize=9)
    ax.legend(fontsize=8);
    ax.grid(alpha=.3)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "**How to read:** This shows what happens if you adjust **only one attribute**, while keeping the rest unchanged. "
        "The blue curve is the model’s prediction: it rises as this attribute increases. "
        f"The red dot is *you*, showing your current z-score and predicted chance of reaching Top-{top_x}. "
        "Try imagining: what if I raised this a bit? Or lowered it? Would I get closer to a higher-performing zone? "
        "Remember: this plot is a ‘what-if.' Not a promise, but a *directional nudge*."
    )

    st.markdown("### 🎻 Compare attribute by rank slice")

    attr_v = st.selectbox(
        "Attribute to inspect", ORDER,
        index=ORDER.index("Keyword Density (%)")
    )
    v_idx = ORDER.index(attr_v)
    X = st.slider("Choose ‘Top X’", 10, 100, 20, step=5)

    fig_v, ax_v = plt.subplots(figsize=(3.6, 3))

    top_slice = Z[ranks <= X, v_idx]
    bottom_slice = Z[ranks > X, v_idx]

    ax_v.violinplot(
        [top_slice, bottom_slice],
        showmeans=True, showextrema=False
    )

    ax_v.set_xticks([1, 2])
    ax_v.set_xticklabels([f"Top {X}", f"{X + 1}-100"])

    ax_v.set_ylabel(f"{attr_v}  z-score")

    your_z = user_z[v_idx]  

    ax_v.scatter(1.5, your_z,
                 color="red", s=60, zorder=5,
                 marker="o", label="You")

    ax_v.hlines(your_z, xmin=0.8, xmax=2.2,
                colors="red", linestyles=":", lw=1)

    ax_v.legend(frameon=False, fontsize=7, loc="upper right")

    st.pyplot(fig_v, clear_figure=True)

    st.caption(
        "**How to read:** each blue shape shows how z-scores are spread for this attribute — "
        f"left = Top-{X} captions, right = {X + 1}-100. The **red dot** is where *your* caption stands. "
        "The **blue bar** inside each shape marks the average (always 0 because of z-scoring). "
        "If your dot sits near the thickest part of a violin, you're similar to that group. "
        "If it's far from both, you're doing something unique on this trait!"
    )

if st.checkbox("📚 Insight dashboard", help=
    "This section explores how caption attributes behave *across all posts*. "
    "It's *independent* from your input. **Think of it as an explorer's map of caption trends**."
):
    plat_ins = st.radio("Dataset:", ["Facebook", "Instagram", "TikTok"], horizontal=True)
    dfraw = load_caption_raw(plat_ins)
    if dfraw is None:
        st.stop()

    st.subheader(f"Attribute Trends Across Rank Clusters – {plat_ins}")
    attr1 = st.selectbox("Pick attribute", ORDER, key="bucket_attr")
    width = st.number_input(
        "Cluster width", 1, 50, 10, step=1,
        help="This sets how many ranks go into one cluster. For example, a width of 13 means ranks 1–13, 14–26, etc."
    )
    dfraw["bucket"] = ((dfraw["Search Rank"] - 1) // width) + 1
    trend = dfraw.groupby("bucket")[attr1].mean()
    fig1, ax1 = plt.subplots(figsize=(4.8, 3))
    ax1.plot(trend.index * width - width / 2 + .5, trend.values, marker="o")
    ax1.set_xlabel("Search rank (bucket mid-point)")
    ax1.set_ylabel(f"Mean {attr1} by cluster")
    ax1.set_title(f"{attr1} vs rank (bucket={width})")
    ax1.grid(alpha=.3)
    st.pyplot(fig1, clear_figure=True)
    st.caption(
        f"**How to read:** we group captions into rank clusters of {width} (e.g., Top 1–{width}, "
        f"{width + 1}–{width * 2}, and so on), then take the average *{attr1}* for each cluster. "
        f"Each dot shows the middle rank of that group. \n\n"
        "- If the line is mostly flat, it means this attribute doesn’t change much across ranks. \n"
        "- If it slopes upward, it means higher values tend to appear in lower-ranked posts. \n"
        f"- If it zigzags, then there's no clear trend for rank clusters of {width}."
    )

    score_df = load_score_file(plat_ins)
    if score_df is not None and "Caption Score" in score_df.columns:
        merged = dfraw.merge(score_df[["Search Rank", "Caption Score"]], on="Search Rank", how="left")
        merged["bucket"] = ((merged["Search Rank"] - 1) // width) + 1
        trend_sc = merged.groupby("bucket")["Caption Score"].mean()
        fig_sc, ax_sc = plt.subplots(figsize=(4.8, 3))
        ax_sc.plot(trend_sc.index * width - width / 2 + .5, trend_sc.values,
                   marker="s", color="#6a51a3")
        ax_sc.set_xlabel("Search rank (cluster mid-point)")
        ax_sc.set_ylabel("Mean Caption Score")
        ax_sc.set_title(f"Caption Score vs rank (cluster={width})")
        ax_sc.grid(alpha=.3)
        st.pyplot(fig_sc, clear_figure=True)
        st.caption(
            f"**How to read:** we group captions into rank clusters of {width} (e.g., Top 1–{width}, "
            f"{width + 1}–{width * 2}, and so on), then take the average *Caption Score* for each cluster. "
            "Each dot shows the middle rank of that group. \n\n"
            "- If the purple line stays flat, it means similar scores are found across all ranks. \n"
            "- If it slopes **downward**, stronger scores are clustering in **better** ranks (e.g., Top 20). \n"
            "- If it jumps up and down, the pattern is inconsistent."
        )

    st.subheader("🔍 Relationship explorer")
    xattr = st.selectbox("Horizontal axis", ORDER, 0, key="sc_x")
    yattr = st.selectbox("Vertical axis", ORDER, 1, key="sc_y")
    topX = st.slider("Define *Top X*", 5, 99, 20, step=1)
    maskT = dfraw["Search Rank"] <= topX
    maskR = ~maskT
    import scipy.stats as ss

    def rho_label(x, y):
        rho, p = ss.spearmanr(x, y, nan_policy="omit")
        strength = "None" if abs(rho) < .1 else "Weak" if abs(rho) < .3 else "Moderate" if abs(rho) < .6 else "Strong"
        sign = "Positive" if rho > 0 else "Negative" if rho < 0 else ""
        sig = "sig." if p < .05 else "n.s."
        return f"ρ {rho:+.2f} ({strength} {sign}, {sig})"

    c1, c2 = st.columns(2)
    with c1:
        figT, axT = plt.subplots(figsize=(3.5, 3))
        axT.scatter(dfraw.loc[maskT, xattr], dfraw.loc[maskT, yattr], alpha=.8, edgecolor="none", color="#4c78a8")
        axT.set_xlabel(xattr); axT.set_ylabel(yattr)
        axT.set_title(rho_label(dfraw.loc[maskT, xattr], dfraw.loc[maskT, yattr]), fontsize=8)
        st.pyplot(figT, clear_figure=True)
    with c2:
        figR, axR = plt.subplots(figsize=(3.5, 3))
        axR.scatter(dfraw.loc[maskR, xattr], dfraw.loc[maskR, yattr], alpha=.25, edgecolor="none", color="#f58518")
        axR.set_xlabel(xattr); axR.set_ylabel(yattr)
        axR.set_title(rho_label(dfraw.loc[maskR, xattr], dfraw.loc[maskR, yattr]), fontsize=8)
        st.pyplot(figR, clear_figure=True)

    st.caption(
        "**How to read:** each dot is a real post. These two charts compare how two attributes "
        "relate in high-ranked vs. lower-ranked captions.\n\n"
        "- The chart titles show Spearman’s ρ (a correlation score) and how strong that pattern is.\n"
        "- If the pattern exists *only* in the Top-X panel, that relationship might be unique to top-performing captions.\n"
        "- If both sides look similar, that relationship exists across all captions, so it may not help you move up.\n\n"
        "*Tip*: Try switching attributes or tweaking Top-X to spot exclusive patterns that only happen in the top ranks."
    )

    st.subheader("Rank-wise mean heat-map (z-scores)")
    z_cols = (dfraw[ORDER] - dfraw[ORDER].mean()) / dfraw[ORDER].std(ddof=0)
    df_z = pd.concat([dfraw["Search Rank"].astype(int), z_cols], axis=1)
    hm_z = df_z.groupby("Search Rank", as_index=True)[ORDER].mean().sort_index()
    figH, axH = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        hm_z.T,                      
        cmap="coolwarm",
        center=0,
        vmin=-2.5, vmax=2.5,
        cbar_kws=dict(label="mean z-score"),
        ax=axH
    )
    ranks = hm_z.index.to_numpy()   # these are the columns after transpose
    tick_locs = [i for i, r in enumerate(ranks) if r % 10 == 0]
    tick_labels = [str(ranks[i]) for i in tick_locs]
    axH.set_xticks(tick_locs)
    axH.set_xticklabels(tick_labels, rotation=0)

    axH.set_xlabel("Rank (1 = best)")
    axH.set_ylabel("Attribute")

    st.pyplot(figH, clear_figure=True)

    st.caption(
        "**How to read:** each row is an exact search rank (Rank 1 = best), "
        "and each column is a caption attribute. The color shows the *average* "
        "z-score for that attribute at each rank:\n\n"
        "- 🔵 Blue = lower than average\n"
        "- ⚪ White = average\n"
        "- 🔴 Red = higher than average\n\n"
        "*Key takeaway:* if you expected certain attributes (like high keyword density or positivity) "
        "to dominate the top ranks, this heatmap lets you test that. If red patches appear across many ranks, not just the top, "
        "it means high values **don’t always** lead to better performance.\n\n"
        "This also reveals how **attribute combinations** behave. If a caption with high keyword density and low readability appears "
        "both in Rank 5 and Rank 95, it’s likely *not* a winning formula on its own. Patterns here help you see if success has a clear shape,"
        "or if it’s more scattered than expected."
    )

if st.checkbox("🎞️ Story-telling: The Full Journey of Caption Science",
               help="An honest overview of every method we tried to predict search visibility with captions. What worked, what didn’t, and what we learned."):
    st.markdown("## 🧩 We Tried Everything")
    st.write("**Goal:** to see if caption attributes can predict search rank. So we began...")

    st.markdown("### 🔗 Spearman correlation")
    st.caption(
        "We began simple: does any *single* caption attribute correlate with Search Rank? Most don’t. "
        "Spearman’s ρ shows weak or no relationship, captions alone aren’t enough to explain rank."
    )

    def _rho_label(rho: float, p: float) -> str:
        abs_r = abs(rho)
        if abs_r == 1: strength = "Perfect"
        elif abs_r > .8: strength = "Strong"
        elif abs_r > .6: strength = "Moderate"
        elif abs_r > .3: strength = "Weak"
        else: strength = "None"
        direction = "pos" if rho > 0 else "neg" if rho < 0 else ""
        sig_tag = "sig" if p < .05 else "n.s."
        return f"ρ = {rho:+.2f} ({strength}-{direction}) · p = {p:.3f} {sig_tag}"

    options = ["Facebook", "Instagram", "TikTok"]
    default_plat = st.session_state.get("last_run", {}).get("platform", "Facebook")

    plat_for_scatter = st.selectbox(
        "Platform to explore:", options, index=options.index(default_plat)
    )

    df_sc = load_caption_raw(plat_for_scatter)  

    st.markdown(f"#### {plat_for_scatter}: each attribute vs. rank")
    scatter_cols = st.columns(5)  

    import scipy.stats as ss

    for i, attr in enumerate(ORDER):
        with scatter_cols[i % 5]:
            fig, ax = plt.subplots(figsize=(2.6, 2.2))

            ax.scatter(df_sc["Search Rank"],
                       df_sc[attr],
                       s=10, alpha=.55, edgecolors="none")

            ax.invert_xaxis()  
            ax.set_title(attr, fontsize=7)
            ax.set_xticks([]);
            ax.set_yticks([])  

            rho, p = ss.spearmanr(df_sc["Search Rank"],
                                  df_sc[attr],
                                  nan_policy="omit")
            ax.text(0.02, 0.94, _rho_label(rho, p),
                    transform=ax.transAxes,
                    fontsize=6, va="top")

            st.pyplot(fig, clear_figure=True)

    st.caption(
        "- **X-axis = Search Rank** (1 = best).  \n"
        "- **Y-axis = attribute value** (e.g., Keyword Density %).  \n"
        "- Each dot = 1 post. Spearman’s ρ shows strength & direction; p-value tells us if it’s statistically significant.\n"
        "- Dots may overlap if many captions share the same rank."
    )

    for p in ["Facebook", "Instagram", "TikTok"]:
        df = load_caption_raw(p)
        corr = df[ORDER + ["Search Rank"]].corr("spearman")["Search Rank"].drop("Search Rank")
        fig, ax = plt.subplots(figsize=(5, 2.2))
        sns.barplot(x=corr.values, y=corr.index, ax=ax,
                    palette="coolwarm", orient="h")
        ax.set_title(f"{p} – correlation to rank (ρ)")
        st.pyplot(fig, clear_figure=True)

    st.markdown("#### How to read the ρ label")
    st.markdown("""
                        | Strength label | ρ range |
                        |----------------|---------|
                        | Perfect ±1.00  | `±1.00` |
                        | Strong         | `0.80 < |ρ| < 1.00` |
                        | Moderate       | `0.30 < |ρ| ≤ 0.60` |
                        | Weak           | `0.00 < |ρ| ≤ 0.30` |
                        | None           | `≈ 0` |

                        *Significant* = p < 0.05 · *Not sig.* = p ≥ 0.05
                        """)

    st.markdown("### 🔁 Logistic regression")
    st.caption(
        "Maybe a combo of attributes could explain rank? We tried logistic regression, a linear formula combining all attributes. "
        "But the AUC stayed near 0.5, no better than flipping a coin."
    )

    lr_auc = pd.DataFrame({p: [STATS[p]["auc"]] for p in STATS.keys()},
                          index=["LogReg"])

    fig_lr, ax_lr = plt.subplots(figsize=(4, 1.8))
    sns.heatmap(lr_auc, annot=True, fmt=".3f", cmap="Greens",
                vmin=.5, vmax=.75, cbar=False, ax=ax_lr)
    ax_lr.set_title("LogReg — test-set AUC")
    st.pyplot(fig_lr, clear_figure=True)

    st.markdown(
        """
        <small><b>AUC legend:</b>&nbsp;
        <span style='background:#e8f5e9;padding:2px 4px'>0.50–0.55</span> = coin-toss &nbsp;·&nbsp;
        <span style='background:#c8e6c9;padding:2px 4px'>0.55–0.65</span> = weak &nbsp;·&nbsp;
        <span style='background:#a5d6a7;padding:2px 4px'>0.65–0.75</span> = fair
        </small>
        """, unsafe_allow_html=True)

    st.markdown("##### ROC curve – how well does the model separate the classes?")
    prob_demo = {
        "Facebook": DATA_DIR / "fb_val_probs.csv",
        "Instagram": DATA_DIR / "ig_val_probs.csv",
        "TikTok": DATA_DIR / "tk_val_probs.csv",
    }

    default_plat = st.session_state.get("last_run", {}).get("platform", "Facebook")
    plat_choice = st.selectbox("Platform to inspect (ROC & overlap)",
                               ["Facebook", "Instagram", "TikTok"],
                               index=["Facebook", "Instagram", "TikTok"].index(default_plat))

    csv_path = prob_demo.get(plat_choice, "")
    if os.path.exists(csv_path):
        demo_df = pd.read_csv(csv_path)  

        from sklearn.metrics import roc_curve, auc  

        fpr, tpr, _ = roc_curve(demo_df["y"], demo_df["p"])
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(3.8, 3.8))
        ax_roc.plot(fpr, tpr, lw=2, label=f"ROC (AUC {roc_auc:.3f})")

        dots = np.linspace(0, len(fpr) - 1, 6, dtype=int)
        ax_roc.scatter(fpr[dots], tpr[dots], color="#4c78a8", s=22, zorder=5)
        ax_roc.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax_roc.set_xlabel("False-positive rate");
        ax_roc.set_ylabel("True-positive rate")
        ax_roc.set_title(f"{plat_choice}: ROC curve", fontsize=9)
        ax_roc.set_aspect("equal", adjustable="box")
        ax_roc.grid(alpha=.3)
        ax_roc.legend(fontsize=8)
        st.pyplot(fig_roc, clear_figure=True)

        st.caption(
            "Each dot shows the model's performance at different thresholds. "
            "If the curve hugs the top-left, the model can tell Top-20 from the rest. Ours doesn't, it's close to the diagonal (random)."
        )
    else:
        st.info(f"Add a CSV with validation probabilities for {plat_choice} to show its ROC curve.")

    if os.path.exists(csv_path):
        demo_df = pd.read_csv(csv_path)  
        fig_sep, ax_sep = plt.subplots(figsize=(4, 2.4))
        sns.violinplot(data=demo_df, x="y", y="p", inner="quartile",
                       palette=["#f58518", "#4c78a8"], ax=ax_sep)
        ax_sep.set_xticklabels(["Not Top-20", "Top-20"])
        ax_sep.set_ylabel("Predicted probability")
        ax_sep.set_xlabel("")
        ax_sep.set_title(f"{plat_choice}: probability overlap", fontsize=9)
        st.pyplot(fig_sep, clear_figure=True)
        st.caption(
            "These violins show predicted probabilities by class. Heavy overlap? It means the model often gives similar scores to both Top-20 and not-Top-20 captions (not useful)."
        )
    else:
        st.info("No validation-probability CSV found – skipping the overlap plot.")

    st.markdown("### 🌳 Random forest")
    st.caption(
        "We thought: maybe it’s not linear. Maybe some caption attributes only matter when combined with others. "
        "So we tried Random Forest, a flexible method for discovering complex interactions."
    )

    rf_auc = pd.DataFrame({"Facebook": [0.583],
                           "Instagram": [0.652],
                           "TikTok": [0.629]},
                          index=["Random Forest"])

    fig_rf, ax_rf = plt.subplots(figsize=(4, 1.8))
    sns.heatmap(rf_auc, annot=True, fmt=".3f", cmap="Blues",
                vmin=.5, vmax=.75, cbar=False, ax=ax_rf)
    ax_rf.set_title("Random Forest — test-set AUC")
    st.pyplot(fig_rf, clear_figure=True)

    st.caption(
        "Slight gains: TikTok hit AUC 0.63, Instagram 0.65. Better, but still unreliable. Captions alone can’t carry the prediction."
    )

    st.markdown("##### How the Random-Forest probabilities overlap")

    rf_prob_demo = {
        "Facebook": DATA_DIR / "facebook_rf_val_probs.csv",
        "Instagram": DATA_DIR / "instagram_rf_val_probs.csv",
        "TikTok": DATA_DIR / "tiktok_rf_val_probs.csv",
    }
    plat_rf = st.selectbox("Platform (RF probs)", ["Facebook", "Instagram", "TikTok"],
                           key="rf_prob_plat")
    rf_path = rf_prob_demo[plat_rf]
    if os.path.exists(rf_path):
        df_rf = pd.read_csv(rf_path)
        fig_rf_v, ax_rf_v = plt.subplots(figsize=(4, 2.4))
        sns.violinplot(data=df_rf, x="y", y="p", split=True, inner="quartile",
                       palette=["#f58518", "#4c78a8"], ax=ax_rf_v)
        ax_rf_v.set_xticklabels(["Not Top-20", "Top-20"])
        ax_rf_v.set_ylabel("Predicted probability")
        ax_rf_v.set_xlabel("")
        ax_rf_v.set_title(f"{plat_rf}: RF probability overlap", fontsize=9)
        st.pyplot(fig_rf_v, clear_figure=True)
        st.caption(
            "Even with more power, the model still assigns similar probabilities to both groups, Top-20 and not. Slightly better separation, but still blurry."
        )
    else:
        st.info(f"File {os.path.basename(rf_path)} missing – generate it first.")

    st.markdown("### 🌲 Decision tree")
    st.caption(
        "What if there’s a clean decision path? A hierarchy like: *If Keyword Density > X, then check CTA, then check Sentiment...*? We tried Decision Trees."
    )

    dt_auc = pd.DataFrame({"Facebook": [0.560],
                           "Instagram": [0.534],
                           "TikTok": [0.592]},
                          index=["Decision Tree"])
    fig, ax = plt.subplots()
    sns.heatmap(dt_auc, annot=True, fmt=".3f", cmap="Purples", cbar=False, ax=ax)
    ax.set_title("Decision Tree — test-set AUC")
    st.pyplot(fig, clear_figure=True)

    st.caption("No luck. AUCs stayed low, the tree couldn’t find strong rules from captions alone.")

    st.markdown("##### How the Decision-Tree probabilities overlap")

    dt_prob_demo = {
        "Facebook": DATA_DIR / "facebook_dt_val_probs.csv",
        "Instagram": DATA_DIR / "instagram_dt_val_probs.csv",
        "TikTok": DATA_DIR / "tiktok_dt_val_probs.csv",
    }
    plat_dt = st.selectbox("Platform (DT probs)", ["Facebook", "Instagram", "TikTok"],
                           key="dt_prob_plat")
    dt_path = dt_prob_demo[plat_dt]
    if os.path.exists(dt_path):
        df_dt = pd.read_csv(dt_path)
        fig_dt_v, ax_dt_v = plt.subplots(figsize=(4, 2.4))
        sns.violinplot(data=df_dt, x="y", y="p", split=True, inner="quartile",
                       palette=["#f58518", "#4c78a8"], ax=ax_dt_v)
        ax_dt_v.set_xticklabels(["Not Top-20", "Top-20"])
        ax_dt_v.set_ylabel("Predicted probability")
        ax_dt_v.set_xlabel("")
        ax_dt_v.set_title(f"{plat_dt}: DT probability overlap", fontsize=9)
        st.pyplot(fig_dt_v, clear_figure=True)
        st.caption(
            "Small shifts in probability, but the groups still blend together. Decision Trees gave clearer logic, but not clearer separation.")
    else:
        st.info(f"File {os.path.basename(dt_path)} missing – generate it first.")

    st.markdown("### 🎬 Final takeaway")
    st.info(
        "We tried it all. Simple stats, linear models, random forests, decision trees. Nothing could reliably predict Top-20 ranks just from captions. "
        "There is no clear separation from how Top-20 captions are compared to the rest. AUCs hovered between 0.5 and 0.65, not enough for strategic decisions."
    )
    st.caption(
        "Captions matter, but they’re just one piece. Visuals, engagement, timing, and algorithm design likely matter more, and most are outside our control."
    )

    st.markdown("## ✏️ Author’s Note: *The absence of radical difference is itself a meaningful result.*")

    st.markdown(
        "<small>Contrary to popular belief, caption attributes, no matter how well crafted, don’t strongly predict search visibility.\n\n"
        "This might go against what many of us believe. We began this study because *we* believed captions could be optimized.\n\n"
        "That there was a science to discover. A breakthrough waiting to be found. \n\n"
        "*But the data tells a different story.* \n\n"
        "We aren’t here to discourage creativity, strategy, or care in caption writing. Quite the **opposite**: we wanted to crack the formula, to offer something practical, actionable, and real. We dreamed of that. We dreamt of a perfect formula.\n\n"
        "_How I wish I could tell you there was._ How I wish I could tell you we found the magic ratio, the perfect combination of readability, keyword density, timing, and sentiment that guaranteed a spot at the top search results.\n\n"
        "But in a time where virality equates to success, it turns out being seen isn't that simple, that even the *'best'* captions may not guarantee reach. And, to me, that may not be the magic formula we wished for, but \n\n"
        "even that itself *is* a result.\n\n"
        "But after months of data collection, modeling, and testing, the signal was weak. The vision of optimizing captions for search visibility wasn't anywhere to be seen. Still, we chose not to hide the result. \n\n"
        "*We chose to tell the truth.* \n\n"
        "Because maybe, just maybe \n\n"
        "*The absence of radical difference is, in itself, a meaningful result.*\n\n"
        "*and we’re okay with that absence*, because that's why we're researchers. Because we're here to show the **truth**.\n\n"
        "— *Ta-asan, Vincent*</small>",
        unsafe_allow_html=True
    )

    st.caption(
        "for those interested in reading vin's full acknowledgements and deeper (sappy 😂) sentiments, "
        "[click here → 'an open letter to caption science and everything in between'](https://docs.google.com/document/d/1rdU5nuCuchRO3Y0r7Wc6oeJvtJ2DIcaeC5ZmJIij40k/edit?usp=sharing)"
    )

if st.checkbox(
            "📂 Raw data & model coefficients",
            value=False,
            help="Opens a panel showing every CSV used plus the β/μ/σ tables.",
    ):

        plat_sel = st.selectbox(
            "Choose platform", ["Facebook", "Instagram", "TikTok"], key="dj_plat"
        )

        prefix = {"Facebook": "fb", "Instagram": "ig", "TikTok": "tk"}[plat_sel]

        file_opts = {
            "Platform raw": DATA_DIR / f"Caption Science - {plat_sel} Caption Score.csv",
            "Z-scores": DATA_DIR / f"{plat_sel.lower()}_z.csv",
            "Logistic regression probs": DATA_DIR / f"{prefix}_val_probs.csv",
            "Random Forest probs": DATA_DIR / f"{plat_sel.lower()}_rf_val_probs.csv",
            "Decision Trees probs": DATA_DIR / f"{plat_sel.lower()}_dt_val_probs.csv",
        }

        ds_name = st.selectbox(
            "Dataset",
            list(file_opts.keys()),
            key="dj_file",
            help="Note: Validation probabilities only include the 300 posts from the 30% hold-out set, not the full dataset."
                 "*The other 700 posts were used as training data*"
        )

        path = file_opts[ds_name]
        if path.exists():
            st.caption(f"*Showing the 1000 rows of **{path.name}***")
            st.dataframe(pd.read_csv(path).head(1000))
        else:
            st.error("File not found – double-check the path.")

        with st.expander(f"ℹ️  Model coefficients & {plat_sel} means"):
            s = STATS[plat_sel]
            coeff_df = pd.DataFrame({
                "Attribute": ORDER,
                "β": s["beta"].round(4),
                "μ (mean)": s["mean"].round(3),
                "σ (std)": s["std"].round(3),
            })
            st.dataframe(coeff_df, hide_index=True)
            st.markdown(
                f"- **Intercept:** `{s['intercept']:+.4f}`  \n"
                f"- **AUC (test):** `{s['auc']:.3f}`"
            )
