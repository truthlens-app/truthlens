
import streamlit as st
import plotly.graph_objects as go
import pickle, torch, re, os, sys
import numpy as np
import requests
import json
import hashlib
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


import os
from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO = "YOUR_HF_USERNAME/truthlens-models"

# Download models if not already present
if not os.path.exists('/tmp/tfidf_model.pkl'):
    hf_hub_download(
        repo_id=HF_REPO,
        filename="tfidf_model.pkl",
        local_dir="/tmp"
    )

if not os.path.exists('/tmp/roberta_final'):
    snapshot_download(
        repo_id=HF_REPO,
        local_dir="/tmp/roberta_final",
        ignore_patterns=["*.md"]
    )


import os
from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO = "Abhichakra/truthlens-models"

# Download models if not already present
if not os.path.exists('/tmp/tfidf_model.pkl'):
    hf_hub_download(
        repo_id=HF_REPO,
        filename="tfidf_model.pkl",
        local_dir="/tmp"
    )

if not os.path.exists('/tmp/roberta_final'):
    snapshot_download(
        repo_id=HF_REPO,
        local_dir="/tmp/roberta_final",
        ignore_patterns=["*.md"]
    )


import os
from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO = "Abhichakra/truthlens-models"

# Download models if not already present
if not os.path.exists('/tmp/tfidf_model.pkl'):
    hf_hub_download(
        repo_id=HF_REPO,
        filename="tfidf_model.pkl",
        local_dir="/tmp"
    )

if not os.path.exists('/tmp/roberta_final'):
    snapshot_download(
        repo_id=HF_REPO,
        local_dir="/tmp/roberta_final",
        ignore_patterns=["*.md"]
    )

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session State Init ───────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'guest_attempts' not in st.session_state:
    st.session_state.guest_attempts = 0
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = 'text'
if 'users_db' not in st.session_state:
    st.session_state.users_db = {}

GUEST_LIMIT = 5

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

.stApp {
    background: linear-gradient(135deg, #050510 0%, #0a0a1a 40%, #0d0818 100%) !important;
    min-height: 100vh;
}

.tl-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    border-radius: 0 0 16px 16px;
    margin-bottom: 1.5rem;
}
.tl-logo {
    font-size: 1.5rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF, #FF4B6E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav-badge {
    background: rgba(123,47,255,0.2);
    border: 1px solid rgba(123,47,255,0.4);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    color: #a78bfa;
    font-size: 0.82rem;
    font-weight: 600;
}
.nav-badge-guest {
    background: rgba(255,149,0,0.15);
    border: 1px solid rgba(255,149,0,0.4);
    color: #fb923c;
}

.hero-section {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.hero-title {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 900;
    background: linear-gradient(90deg, #00D4FF 0%, #7B2FFF 50%, #FF4B6E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 0.5rem;
}
.hero-tagline {
    display: inline-block;
    background: linear-gradient(90deg, rgba(0,212,255,0.15), rgba(123,47,255,0.15));
    border: 1px solid rgba(123,47,255,0.3);
    border-radius: 20px;
    padding: 0.35rem 1.2rem;
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.stats-grid {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0 1.5rem;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 0.8rem 1.4rem;
    text-align: center;
    min-width: 130px;
}
.stat-pill .num {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-pill .label { color: #64748b; font-size: 0.78rem; margin-top: 0.1rem; }

.input-label {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 14px !important;
    font-size: 0.96rem !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(123,47,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(123,47,255,0.1) !important;
}
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 14px !important;
    font-size: 0.96rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
}

div.stButton > button {
    background: linear-gradient(90deg, #00D4FF, #7B2FFF, #FF4B6E) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.85rem 2.5rem !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.3px;
    transition: all 0.3s !important;
    box-shadow: 0 4px 24px rgba(123,47,255,0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(123,47,255,0.5) !important;
}

.verdict-fake {
    background: linear-gradient(135deg, rgba(255,75,110,0.18), rgba(255,75,110,0.05));
    border: 2px solid rgba(255,75,110,0.55);
    border-radius: 24px; padding: 2.2rem; text-align: center; margin: 1rem 0;
    box-shadow: 0 8px 40px rgba(255,75,110,0.15);
}
.verdict-real {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,212,255,0.05));
    border: 2px solid rgba(0,212,255,0.55);
    border-radius: 24px; padding: 2.2rem; text-align: center; margin: 1rem 0;
    box-shadow: 0 8px 40px rgba(0,212,255,0.15);
}
.verdict-medium {
    background: linear-gradient(135deg, rgba(251,146,60,0.18), rgba(251,146,60,0.05));
    border: 2px solid rgba(251,146,60,0.55);
    border-radius: 24px; padding: 2.2rem; text-align: center; margin: 1rem 0;
    box-shadow: 0 8px 40px rgba(251,146,60,0.15);
}
.verdict-unverified {
    background: linear-gradient(135deg, rgba(255,215,0,0.18), rgba(255,215,0,0.05));
    border: 2px solid rgba(255,215,0,0.55);
    border-radius: 24px; padding: 2.2rem; text-align: center; margin: 1rem 0;
    box-shadow: 0 8px 40px rgba(255,215,0,0.15);
}
.verdict-text { font-size: 2rem; font-weight: 800; color: white; margin-bottom: 0.5rem; }
.prob-text { font-size: 4rem; font-weight: 900; margin: 0.3rem 0; }
.prob-sub { color: #64748b; font-size: 0.92rem; margin-top: 0.4rem; }

.reason-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.65rem 1.1rem;
    margin: 0.35rem 0;
    font-size: 0.92rem;
    color: #cbd5e1;
    display: block;
}

.section-title {
    color: white;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 1.8rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.info-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    margin: 1rem 0;
}
.info-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
}

.guest-banner {
    background: linear-gradient(90deg, rgba(255,149,0,0.12), rgba(255,75,110,0.12));
    border: 1px solid rgba(255,149,0,0.35);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.guest-banner-text { color: #fcd34d; font-size: 0.92rem; font-weight: 500; }

.auth-card {
    max-width: 440px;
    margin: 3rem auto;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 2.5rem 2rem;
}
.auth-title {
    font-size: 2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.auth-sub {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
.auth-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.2rem 0;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-name { color: white; font-weight: 600; font-size: 0.95rem; }
.feature-desc { color: #64748b; font-size: 0.8rem; margin-top: 0.2rem; }

.stCheckbox label { color: #94a3b8 !important; font-size: 0.9rem !important; }

.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(123,47,255,0.4), transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════════════════════════
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password, email):
    db = st.session_state.users_db
    if username in db:
        return False, "Username already exists."
    db[username] = {"password": hash_pw(password), "email": email, "analyses": 0}
    return True, "Account created!"

def login_user(username, password):
    db = st.session_state.users_db
    if username not in db:
        return False, "User not found."
    if db[username]["password"] != hash_pw(password):
        return False, "Incorrect password."
    return True, "Welcome back!"


# ══════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    with open('/tmp/tfidf_model.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = RobertaTokenizer.from_pretrained('/tmp/roberta_final')
    rob = RobertaForSequenceClassification.from_pretrained('/tmp/roberta_final')
    rob.eval()
    rob.to(device)
    return tfidf, tok, rob, device

tfidf_model, rob_tok, rob_model, device = load_all_models()
vader = SentimentIntensityAnalyzer()
lemma = WordNetLemmatizer()
sw    = set(stopwords.words('english'))

FAKE_WORDS = [
    'shocking','exposed','secret','you won\'t believe','miracle',
    'banned','censored','deep state','hoax','conspiracy',
    'crisis actor','false flag','wake up','plandemic',
    'share before deleted','bombshell','whistleblower','cover-up',
    'they don\'t want','government hiding'
]


# ══════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([lemma.lemmatize(w) for w in text.split() if w not in sw])

def predict_roberta(text):
    inputs = rob_tok(text[:512], return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = rob_model(**inputs)
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    return float(probs[1])

def predict_tfidf(text):
    cleaned = clean_text(text)
    prob = tfidf_model.predict_proba([cleaned])[0]
    return float(prob[1])

def analyze_features(text):
    vs    = vader.polarity_scores(text)
    tb    = TextBlob(text)
    tl    = text.lower()
    words = text.split()
    return {
        'vader_compound'  : vs['compound'],
        'subjectivity'    : tb.sentiment.subjectivity,
        'polarity'        : tb.sentiment.polarity,
        'fake_word_count' : sum(1 for w in FAKE_WORDS if w in tl),
        'caps_ratio'      : sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1) * 100,
        'exclamations'    : text.count('!'),
        'word_count'      : len(words)
    }

def scrape_url(url):
    try:
        art = Article(url)
        art.download(); art.parse()
        if len(art.text) > 100:
            return art.title, art.text, art.authors, str(art.publish_date), True
        raise Exception("Too short")
    except:
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            h1 = soup.find('h1')
            ps = soup.find_all('p')
            body = ' '.join(p.text for p in ps[:40])
            if len(body) < 100:
                return '', '', [], '', False
            return (h1.text.strip() if h1 else ''), body, [], '', True
        except:
            return '', '', [], '', False

def get_verdict(prob):
    if prob >= 80:   return "🔴 HIGH RISK — Likely Fake",    "fake",       "#FF4B6E"
    elif prob >= 60: return "🟠 SUSPICIOUS — Possibly Fake", "medium",     "#fb923c"
    elif prob >= 40: return "🟡 UNVERIFIED — Use Caution",   "unverified", "#FFD700"
    else:            return "🟢 CREDIBLE — Likely Real",      "real",       "#00D4FF"

def build_reasons(features, fake_prob):
    r = []
    if features['fake_word_count'] > 0:
        r.append(f"⚠️ Contains {features['fake_word_count']} sensationalist / clickbait keyword(s)")
    if features['caps_ratio'] > 5:
        r.append(f"⚠️ {features['caps_ratio']:.1f}% words in ALL CAPS — panic-baiting signal")
    if features['vader_compound'] < -0.5:
        r.append("⚠️ Strongly negative / alarmist emotional tone detected")
    if features['subjectivity'] > 0.7:
        r.append(f"⚠️ High subjectivity score ({features['subjectivity']:.2f}/1.0) — opinion-heavy")
    if features['exclamations'] > 3:
        r.append(f"⚠️ {features['exclamations']} exclamation marks — exaggerated urgency")
    if not r:
        if fake_prob >= 60:
            r.append("🤖 AI detected writing patterns consistent with misinformation")
        elif fake_prob >= 40:
            r.append("🔍 Could not fully verify — content may lack credible sourcing")
        else:
            r.append("✅ Language appears neutral, factual, and credible")
    return r

def run_analysis(text, url_mode=False):
    title, authors, pub_date = '', [], ''
    if url_mode:
        title, body, authors, pub_date, ok = scrape_url(text)
        if not ok or len(body) < 80:
            return None, "Could not extract enough content from this URL. Try pasting the article text directly."
        text = title + ' ' + body

    if len(text.strip()) < 50:
        return None, "Text is too short. Please provide at least 50 characters."

    rob_fp   = predict_roberta(text[:1024])
    tfidf_fp = predict_tfidf(text)

    boost = 0.0
    if rob_fp > 0.90 and tfidf_fp > 0.70: boost =  0.08
    if rob_fp < 0.20 and tfidf_fp < 0.30: boost = -0.05
    fake_prob = min(max(tfidf_fp + boost, 0.0), 1.0)

    features = analyze_features(text)
    if features['fake_word_count'] >= 3 and fake_prob > 0.5:
        fake_prob = min(fake_prob + 0.05, 1.0)

    verdict, v_class, v_color = get_verdict(fake_prob * 100)
    reasons = build_reasons(features, fake_prob * 100)

    return {
        'verdict'  : verdict,
        'v_class'  : v_class,
        'v_color'  : v_color,
        'fake_pct' : round(fake_prob * 100, 1),
        'real_pct' : round((1 - fake_prob) * 100, 1),
        'rob_pct'  : round(rob_fp * 100, 1),
        'tfidf_pct': round(tfidf_fp * 100, 1),
        'features' : features,
        'reasons'  : reasons,
        'title'    : title,
        'authors'  : authors,
        'pub_date' : pub_date,
    }, None


# ══════════════════════════════════════════════════════
#  NAV BAR
# ══════════════════════════════════════════════════════
def render_nav():
    col_logo, col_right = st.columns([3, 2])
    with col_logo:
        st.markdown("<div class='tl-logo'>🔍 TruthLens</div>", unsafe_allow_html=True)
    with col_right:
        if st.session_state.logged_in:
            nc1, nc2 = st.columns([3, 1])
            with nc1:
                st.markdown(
                    f"<div class='nav-badge'>👤 {st.session_state.username}</div>",
                    unsafe_allow_html=True
                )
            with nc2:
                if st.button("Log out", key="nav_logout"):
                    st.session_state.logged_in = False
                    st.session_state.username  = ''
                    st.session_state.page      = 'main'
                    st.rerun()
        else:
            attempts_left = GUEST_LIMIT - st.session_state.guest_attempts
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                st.markdown(
                    f"<div class='nav-badge nav-badge-guest'>🎯 {attempts_left} free left</div>",
                    unsafe_allow_html=True
                )
            with nc2:
                if st.button("Log In", key="nav_login"):
                    st.session_state.page = 'login'
                    st.rerun()
            with nc3:
                if st.button("Sign Up", key="nav_signup"):
                    st.session_state.page = 'signup'
                    st.rerun()


# ══════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════
def render_login():
    render_nav()
    st.markdown("""
    <div class='auth-card'>
        <div class='auth-title'>Welcome Back 👋</div>
        <div class='auth-sub'>Sign in to get unlimited analyses</div>
    </div>
    """, unsafe_allow_html=True)

    col_c, col_form, col_c2 = st.columns([1, 2, 1])
    with col_form:
        username = st.text_input("Username", key="li_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="li_pass", placeholder="Enter your password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔐 Sign In", key="do_login"):
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                ok, msg = login_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.session_state.page      = 'main'
                    st.success(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown("<hr class='auth-divider'>", unsafe_allow_html=True)
        if st.button("Don't have an account? Sign Up →", key="go_signup"):
            st.session_state.page = 'signup'
            st.rerun()
        if st.button("← Back to TruthLens", key="back_from_login"):
            st.session_state.page = 'main'
            st.rerun()


# ══════════════════════════════════════════════════════
#  SIGNUP PAGE
# ══════════════════════════════════════════════════════
def render_signup():
    render_nav()
    st.markdown("""
    <div class='auth-card'>
        <div class='auth-title'>Join TruthLens 🚀</div>
        <div class='auth-sub'>Free account — unlimited analyses forever</div>
    </div>
    """, unsafe_allow_html=True)

    col_c, col_form, col_c2 = st.columns([1, 2, 1])
    with col_form:
        username = st.text_input("Choose a username", key="su_user", placeholder="e.g. newshunter99")
        email    = st.text_input("Email address",     key="su_email", placeholder="you@example.com")
        password = st.text_input("Create password",   type="password", key="su_pass", placeholder="At least 6 characters")
        confirm  = st.text_input("Confirm password",  type="password", key="su_conf", placeholder="Repeat password")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Create Account", key="do_signup"):
            if not all([username, email, password, confirm]):
                st.error("Please fill in all fields.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(username, password, email)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.session_state.page      = 'main'
                    st.success(f"✅ {msg} You're in!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown("<hr class='auth-divider'>", unsafe_allow_html=True)
        if st.button("Already have an account? Log In →", key="go_login"):
            st.session_state.page = 'login'
            st.rerun()
        if st.button("← Back to TruthLens", key="back_from_signup"):
            st.session_state.page = 'main'
            st.rerun()


# ══════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════
def render_results(result):
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Analysis Results</div>", unsafe_allow_html=True)

    css = f"verdict-{result['v_class']}"
    st.markdown(f"""
    <div class='{css}'>
        <div class='verdict-text'>{result['verdict']}</div>
        <div class='prob-text' style='color:{result["v_color"]};'>
            {result['fake_pct']}%
        </div>
        <div class='prob-sub'>Misinformation Probability &nbsp;|&nbsp; Real: {result['real_pct']}%</div>
    </div>
    """, unsafe_allow_html=True)

    if result['title']:
        authors_str = ', '.join(result['authors']) if result['authors'] else 'Unknown author'
        date_str = result['pub_date'][:10] if result['pub_date'] and result['pub_date'] != 'None' else ''
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:14px;padding:1.1rem 1.3rem;margin:0.8rem 0;'>
            <span style='color:#64748b;font-size:0.8rem;'>📰 ARTICLE DETECTED</span><br>
            <span style='color:white;font-weight:600;font-size:1rem;'>{result['title'][:140]}</span><br>
            <div style='margin-top:0.4rem;'>
                <span style='color:#94a3b8;font-size:0.82rem;'>👤 {authors_str}</span>
                {'&nbsp;&nbsp;|&nbsp;&nbsp;<span style="color:#94a3b8;font-size:0.82rem;">📅 ' + date_str + '</span>' if date_str else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        fig_gauge = go.Figure(go.Indicator(
            mode   = "gauge+number",
            value  = result['fake_pct'],
            number = {"suffix": "%", "font": {"color": "white", "size": 36}},
            title  = {"text": "Misinformation Risk Score", "font": {"color": "#94a3b8", "size": 13}},
            gauge  = {
                "axis"     : {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
                "bar"      : {"color": result['v_color'], "thickness": 0.28},
                "steps"    : [
                    {"range": [0,  40], "color": "rgba(0,212,255,0.10)"},
                    {"range": [40, 60], "color": "rgba(255,215,0,0.10)"},
                    {"range": [60, 80], "color": "rgba(251,146,60,0.10)"},
                    {"range": [80,100], "color": "rgba(255,75,110,0.10)"},
                ],
                "threshold": {"line": {"color": "rgba(255,255,255,0.3)", "width": 2}, "value": 60},
                "bgcolor"  : "rgba(0,0,0,0)"
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color   ="white",
            height=280, margin=dict(t=70, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        fig_bar = go.Figure(go.Bar(
            x            = ["RoBERTa", "TF-IDF + LR", "Final Score"],
            y            = [result['rob_pct'], result['tfidf_pct'], result['fake_pct']],
            marker_color = ["#7B2FFF", "#00D4FF", result['v_color']],
            marker_line_width=0,
            text         = [f"{v:.1f}%" for v in [result['rob_pct'], result['tfidf_pct'], result['fake_pct']]],
            textposition = "outside",
            textfont     = {"color": "white", "size": 13}
        ))
        fig_bar.update_layout(
            title      = {"text": "🧠 Model Breakdown", "font": {"color": "white", "size": 14}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            yaxis      = {"range": [0, 120], "tickfont": {"color": "#64748b"},
                          "gridcolor": "rgba(255,255,255,0.04)"},
            xaxis      = {"tickfont": {"color": "#94a3b8"}},
            font_color ="white",
            height=280, margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    f    = result['features']
    cats = ["Negativity", "Subjectivity", "Sensationalism", "CAPS Use", "Exclamations"]
    vals = [
        min(abs(f['vader_compound']) * 100, 100),
        f['subjectivity'] * 100,
        min(f['fake_word_count'] * 20, 100),
        min(f['caps_ratio'] * 5, 100),
        min(f['exclamations'] * 12, 100)
    ]
    fill_color = "rgba(255,75,110,0.2)" if result['fake_pct'] > 60 else "rgba(0,212,255,0.2)"
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor=fill_color,
        line_color=result['v_color'], name="Risk Profile"
    ))
    fig_radar.update_layout(
        polar={
            "bgcolor"    : "rgba(0,0,0,0)",
            "radialaxis" : {"visible": True, "range": [0, 100],
                            "tickfont": {"color": "#64748b"},
                            "gridcolor": "rgba(255,255,255,0.08)"},
            "angularaxis": {"tickfont": {"color": "#94a3b8", "size": 12},
                            "gridcolor": "rgba(255,255,255,0.08)"}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title={"text": "🎯 Linguistic Risk Profile", "font": {"color": "white", "size": 15}},
        height=380, margin=dict(t=60, b=20, l=60, r=60)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<div class='section-title'>💡 Why This Verdict?</div>", unsafe_allow_html=True)
    for reason in result['reasons']:
        st.markdown(f"<div class='reason-pill'>{reason}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Linguistic Statistics</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("😤 Sentiment",    f"{f['vader_compound']:.2f}", "-1=negative, +1=positive")
    s2.metric("🎭 Subjectivity", f"{f['subjectivity']:.2f}",  "0=fact, 1=opinion")
    s3.metric("📢 CAPS Ratio",   f"{f['caps_ratio']:.1f}%",   "% words in ALL CAPS")
    s4.metric("📝 Word Count",   f"{f['word_count']}",         "words analyzed")

    show_tech = st.checkbox("🔬 Show technical details", value=False)
    if show_tech:
        with st.expander("Raw model scores"):
            st.json({
                "roberta_fake_prob" : f"{result['rob_pct']}%",
                "tfidf_fake_prob"   : f"{result['tfidf_pct']}%",
                "final_fake_prob"   : f"{result['fake_pct']}%",
                "ensemble_method"   : "TF-IDF primary + RoBERTa boost",
                "linguistic_features": f
            })

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⚠️ TruthLens assists detection — always verify with multiple trusted sources like Reuters, AP News, or Snopes.")


# ══════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════
def render_main():
    render_nav()

    st.markdown("""
    <div class='hero-section'>
        <div class='hero-tagline'>🤖 Powered by RoBERTa + TF-IDF Ensemble · Trained on 44,000+ Articles</div>
        <div class='hero-title'>🔍 TruthLens</div>
        <div class='hero-sub'>AI-Powered Fake News & Misinformation Detector</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='stats-grid'>
        <div class='stat-pill'><div class='num'>99.5%</div><div class='label'>🎯 Accuracy</div></div>
        <div class='stat-pill'><div class='num'>&lt;2s</div><div class='label'>⚡ Per Article</div></div>
        <div class='stat-pill'><div class='num'>44K+</div><div class='label'>📰 Articles Trained</div></div>
        <div class='stat-pill'><div class='num'>2 AI</div><div class='label'>🧠 Models Ensemble</div></div>
        <div class='stat-pill'><div class='num'>Free</div><div class='label'>🎁 5 Guest Tries</div></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        attempts_left = GUEST_LIMIT - st.session_state.guest_attempts
        if attempts_left <= 2 and attempts_left > 0:
            col_b, col_b2 = st.columns([4, 1])
            with col_b:
                st.markdown(
                    f"<div class='guest-banner'>"
                    f"<span class='guest-banner-text'>⚡ You have <b>{attempts_left} free analysis attempt(s)</b> left. Sign up for unlimited access!</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_b2:
                if st.button("Sign Up Free →", key="banner_signup"):
                    st.session_state.page = 'signup'
                    st.rerun()
        elif attempts_left <= 0:
            st.markdown("""
            <div class='guest-banner' style='background:linear-gradient(90deg,rgba(255,75,110,0.15),rgba(123,47,255,0.15));
                 border-color:rgba(255,75,110,0.4);'>
                <span class='guest-banner-text' style='color:#f87171;'>
                    🔒 You have used all 5 free attempts. Create a free account for unlimited access!
                </span>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Create Free Account", key="limit_signup"):
                    st.session_state.page = 'signup'
                    st.rerun()
            with col2:
                if st.button("🔐 Already Have Account? Log In", key="limit_login"):
                    st.session_state.page = 'login'
                    st.rerun()
            st.stop()

    col_tc, col_toggle, col_tc2 = st.columns([1, 2, 1])
    with col_toggle:
        mode_choice = st.radio(
            "Choose input type:",
            ["📝 Paste Article Text", "🔗 Enter URL"],
            horizontal=True,
            key="input_mode_radio",
            label_visibility="collapsed"
        )

    is_url = "URL" in mode_choice

    st.markdown("<div style='max-width:820px;margin:0 auto;'>", unsafe_allow_html=True)

    if is_url:
        st.markdown("<div class='input-label'>🔗 Paste a news article URL</div>", unsafe_allow_html=True)
        user_input = st.text_input(
            "url_input",
            placeholder="https://news-website.com/article-title-here",
            key="url_field",
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class='info-strip'>
            <div class='info-chip'>✅ Supports most news sites</div>
            <div class='info-chip'>⚡ Auto-extracts article text</div>
            <div class='info-chip'>📰 Detects title & author</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='input-label'>📝 Paste article text or headline</div>", unsafe_allow_html=True)
        user_input = st.text_area(
            "text_input",
            height=190,
            placeholder="Paste the full news article, headline, or any text you want to fact-check here...",
            key="text_field",
            label_visibility="collapsed"
        )
        if user_input:
            wc = len(user_input.split())
            st.markdown(f"<div style='color:#64748b;font-size:0.8rem;text-align:right;'>{wc} words</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        analyze_clicked = st.button("🔍 Analyze for Misinformation", use_container_width=True)

    if analyze_clicked:
        if not user_input or len(user_input.strip()) < 10:
            st.warning("⚠️ Please enter some text or a valid URL to analyze.")
            st.stop()

        with st.spinner("🤖 Running AI models... this takes ~2 seconds"):
            result, err = run_analysis(user_input, url_mode=is_url)

        if err:
            st.error(f"❌ {err}")
            st.stop()

        if not st.session_state.logged_in:
            st.session_state.guest_attempts += 1
        else:
            db = st.session_state.users_db
            if st.session_state.username in db:
                db[st.session_state.username]['analyses'] += 1

        render_results(result)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:1rem;'>
        <span style='color:white;font-size:1.2rem;font-weight:700;'>Why TruthLens?</span><br>
        <span style='color:#64748b;font-size:0.88rem;'>The most powerful open fake-news detector, powered by two AI models</span>
    </div>
    <div class='features-grid'>
        <div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <div class='feature-name'>RoBERTa AI</div>
            <div class='feature-desc'>Fine-tuned transformer model — state of the art NLP</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>📊</div>
            <div class='feature-name'>TF-IDF Ensemble</div>
            <div class='feature-desc'>Logistic regression on 44K article patterns</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>🎯</div>
            <div class='feature-name'>99.5% Accuracy</div>
            <div class='feature-desc'>Verified on held-out test set of 8,800+ articles</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>🔗</div>
            <div class='feature-name'>URL Scraping</div>
            <div class='feature-desc'>Paste a link — we extract and analyze automatically</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>⚡</div>
            <div class='feature-name'>Under 2 Seconds</div>
            <div class='feature-desc'>Real-time results with GPU acceleration</div>
        </div>
        <div class='feature-card'>
            <div class='feature-icon'>📈</div>
            <div class='feature-name'>Linguistic Analysis</div>
            <div class='feature-desc'>Sentiment, subjectivity, sensationalism detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;color:#334155;font-size:0.78rem;margin-top:3rem;
                padding:1.5rem;border-top:1px solid rgba(255,255,255,0.05);'>
        🔍 <b style='color:#475569;'>TruthLens</b> — AI Fake News Detector &nbsp;|&nbsp;
        Built with RoBERTa + TF-IDF + Streamlit &nbsp;|&nbsp;
        Trained on 44,000+ Articles &nbsp;|&nbsp;
        <span style='color:#7B2FFF;'>Always verify with trusted sources</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════
if st.session_state.page == 'login':
    render_login()
elif st.session_state.page == 'signup':
    render_signup()
else:
    render_main()
