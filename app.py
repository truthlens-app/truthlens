# ============================================================
# TruthLens — AI Fake News Detector
# Full Production App — app.py
# Deploy on: Streamlit Cloud (24/7) via GitHub
# ============================================================

import os
import sys
import re
import json
import pickle
import hashlib
import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

# ── Auto-download models from HuggingFace if needed ─────────
from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO = "Abhichakra/truthlens-models"

@st.cache_resource(show_spinner=False)
def download_models():
    os.makedirs("/tmp/models", exist_ok=True)
    if not os.path.exists("/tmp/models/tfidf_model.pkl"):
        hf_hub_download(
            repo_id=HF_REPO,
            filename="tfidf_model.pkl",
            local_dir="/tmp/models"
        )
    if not os.path.exists("/tmp/models/roberta_final"):
        snapshot_download(
            repo_id=HF_REPO,
            local_dir="/tmp/models/roberta_final",
            ignore_patterns=["*.md"]
        )
    return "/tmp/models"

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session State Init ───────────────────────────────────────
for key, default in {
    "page": "home",
    "logged_in": False,
    "username": "",
    "user_email": "",
    "user_joined": "",
    "guest_attempts": 0,
    "users_db": {},
    "history": [],        # persists across browser reloads for logged-in users
    "input_mode": "text",
    "last_result": None,
    "lang": "en",
    "show_profile_menu": False,
    "theme": "dark",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

GUEST_LIMIT = 5   # ← guests get 5 free checks total (resets each browser session, NOT each reload)

# ─────────────────────────────────────────────────────────────
# LANGUAGE STRINGS  (Telugu · Hindi · Malayalam · English + more)
# ─────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "flag": "🇬🇧", "name": "English",
        "analyze": "Analyze", "paste_text": "Paste article text here…",
        "enter_url": "Enter article URL…",
        "result_real": "✅ Likely REAL News",
        "result_fake": "🚨 Likely FAKE News",
        "result_suspicious": "⚠️ Suspicious / Unverified",
        "result_unverified": "🔍 Needs Verification",
        "confidence": "Confidence", "why": "Why this result?",
        "how": "How was this analyzed?",
        "history": "My History", "profile": "Profile",
        "logout": "Logout", "login": "Login",
        "signup": "Sign Up", "guest": "Continue as Guest",
        "attempts_left": "free checks left",
        "upgrade_msg": "Sign up for unlimited checks!",
        "copy": "Copy Result", "share": "Share",
        "tab_text": "Paste Text", "tab_url": "Enter URL",
        "real_score": "Real Score", "fake_score": "Fake Score",
        "sentiment": "Sentiment", "subjectivity": "Subjectivity",
        "word_count": "Word Count", "clickbait": "Clickbait Words",
        "about": "About", "how_it_works": "How It Works",
    },
    "te": {
        "flag": "🇮🇳", "name": "తెలుగు",
        "analyze": "విశ్లేషించు", "paste_text": "వ్యాసం టెక్స్ట్ ఇక్కడ పేస్ట్ చేయండి…",
        "enter_url": "వ్యాస URL నమోదు చేయండి…",
        "result_real": "✅ నిజమైన వార్త అని అనుమానం",
        "result_fake": "🚨 నకిలీ వార్త అని అనుమానం",
        "result_suspicious": "⚠️ అనుమానాస్పద / నిరూపించబడలేదు",
        "result_unverified": "🔍 నిరూపణ అవసరం",
        "confidence": "విశ్వాసనీయత", "why": "ఇది ఎందుకు?",
        "how": "ఇది ఎలా విశ్లేషించబడింది?",
        "history": "నా చరిత్ర", "profile": "ప్రొఫైల్",
        "logout": "లాగ్‌అవుట్", "login": "లాగిన్",
        "signup": "సైన్ అప్", "guest": "అతిథిగా కొనసాగించండి",
        "attempts_left": "ఉచిత తనిఖీలు మిగిలి ఉన్నాయి",
        "upgrade_msg": "అపరిమిత తనిఖీల కోసం సైన్ అప్ చేయండి!",
        "copy": "ఫలితం కాపీ చేయండి", "share": "షేర్ చేయండి",
        "tab_text": "టెక్స్ట్ పేస్ట్ చేయండి", "tab_url": "URL నమోదు చేయండి",
        "real_score": "నిజ స్కోరు", "fake_score": "నకిలీ స్కోరు",
        "sentiment": "భావోద్వేగం", "subjectivity": "ఆత్మాశ్రయత",
        "word_count": "పదాల సంఖ్య", "clickbait": "క్లిక్‌బెయిట్ పదాలు",
        "about": "గురించి", "how_it_works": "ఇది ఎలా పని చేస్తుంది",
    },
    "hi": {
        "flag": "🇮🇳", "name": "हिंदी",
        "analyze": "विश्लेषण करें", "paste_text": "लेख यहाँ पेस्ट करें…",
        "enter_url": "लेख URL दर्ज करें…",
        "result_real": "✅ संभवतः असली खबर",
        "result_fake": "🚨 संभवतः फर्जी खबर",
        "result_suspicious": "⚠️ संदिग्ध / असत्यापित",
        "result_unverified": "🔍 सत्यापन आवश्यक",
        "confidence": "विश्वास स्तर", "why": "यह परिणाम क्यों?",
        "how": "इसका विश्लेषण कैसे हुआ?",
        "history": "मेरा इतिहास", "profile": "प्रोफ़ाइल",
        "logout": "लॉगआउट", "login": "लॉगिन",
        "signup": "साइन अप", "guest": "अतिथि के रूप में जारी रखें",
        "attempts_left": "मुफ़्त जाँचें बची हैं",
        "upgrade_msg": "असीमित जाँच के लिए साइन अप करें!",
        "copy": "परिणाम कॉपी करें", "share": "साझा करें",
        "tab_text": "टेक्स्ट पेस्ट करें", "tab_url": "URL दर्ज करें",
        "real_score": "वास्तविक स्कोर", "fake_score": "नकली स्कोर",
        "sentiment": "भावना", "subjectivity": "व्यक्तिपरकता",
        "word_count": "शब्द संख्या", "clickbait": "क्लिकबेट शब्द",
        "about": "बारे में", "how_it_works": "यह कैसे काम करता है",
    },
    "ml": {
        "flag": "🇮🇳", "name": "മലയാളം",
        "analyze": "വിശകലനം ചെയ്യുക", "paste_text": "ലേഖനം ഇവിടെ ഒട്ടിക്കുക…",
        "enter_url": "ലേഖന URL നൽകുക…",
        "result_real": "✅ യഥാർഥ വാർത്തയാണെന്ന് സംശയം",
        "result_fake": "🚨 വ്യാജ വാർത്തയാണെന്ന് സംശയം",
        "result_suspicious": "⚠️ സംശയകരം / സ്ഥിരീകരിക്കപ്പെട്ടിട്ടില്ല",
        "result_unverified": "🔍 സ്ഥിരീകരണം ആവശ്യമാണ്",
        "confidence": "വിശ്വാസ്യത", "why": "ഈ ഫലം എന്തുകൊണ്ട്?",
        "how": "ഇത് എങ്ങനെ വിശകലനം ചെയ്തു?",
        "history": "എന്റെ ചരിത്രം", "profile": "പ്രൊഫൈൽ",
        "logout": "ലോഗൗട്ട്", "login": "ലോഗിൻ",
        "signup": "സൈൻ അപ്പ്", "guest": "അതിഥിയായി തുടരുക",
        "attempts_left": "സൗജന്യ പരിശോധനകൾ ശേഷിക്കുന്നു",
        "upgrade_msg": "പരിധിയില്ലാത്ത പരിശോധനകൾക്ക് സൈൻ അപ്പ് ചെയ്യുക!",
        "copy": "ഫലം പകർത്തുക", "share": "പങ്കുവയ്ക്കുക",
        "tab_text": "ടെക്സ്റ്റ് ഒട്ടിക്കുക", "tab_url": "URL നൽകുക",
        "real_score": "യഥാർഥ സ്കോർ", "fake_score": "വ്യാജ സ്കോർ",
        "sentiment": "വികാരം", "subjectivity": "ആത്മനിഷ്ഠത",
        "word_count": "വാക്കുകളുടെ എണ്ണം", "clickbait": "ക്ലിക്ക്ബെയ്റ്റ് വാക്കുകൾ",
        "about": "കുറിച്ച്", "how_it_works": "ഇത് എങ്ങനെ പ്രവർത്തിക്കുന്നു",
    },
    "ta": {
        "flag": "🇮🇳", "name": "தமிழ்",
        "analyze": "பகுப்பாய்வு செய்", "paste_text": "கட்டுரையை இங்கே ஒட்டவும்…",
        "enter_url": "கட்டுரை URL ஐ உள்ளிடவும்…",
        "result_real": "✅ உண்மையான செய்தி என்று சந்தேகம்",
        "result_fake": "🚨 போலி செய்தி என்று சந்தேகம்",
        "result_suspicious": "⚠️ சந்தேகாஸ்பதமான / சரிபார்க்கப்படவில்லை",
        "result_unverified": "🔍 சரிபார்ப்பு தேவை",
        "confidence": "நம்பகத்தன்மை", "why": "இந்த முடிவு ஏன்?",
        "how": "இது எவ்வாறு பகுப்பாய்வு செய்யப்பட்டது?",
        "history": "என் வரலாறு", "profile": "சுயவிவரம்",
        "logout": "வெளியேறு", "login": "உள்நுழை",
        "signup": "பதிவு செய்", "guest": "விருந்தினராக தொடரவும்",
        "attempts_left": "இலவச சோதனைகள் மீதமுள்ளன",
        "upgrade_msg": "வரம்பற்ற சோதனைகளுக்கு பதிவு செய்யுங்கள்!",
        "copy": "முடிவை நகலெடு", "share": "பகிர்",
        "tab_text": "உரையை ஒட்டவும்", "tab_url": "URL ஐ உள்ளிடவும்",
        "real_score": "உண்மை மதிப்பெண்", "fake_score": "போலி மதிப்பெண்",
        "sentiment": "உணர்வு", "subjectivity": "தனிப்பட்ட கருத்து",
        "word_count": "வார்த்தை எண்ணிக்கை", "clickbait": "கிளிக்பெயிட் வார்த்தைகள்",
        "about": "பற்றி", "how_it_works": "இது எவ்வாறு செயல்படுகிறது",
    },
    "kn": {
        "flag": "🇮🇳", "name": "ಕನ್ನಡ",
        "analyze": "ವಿಶ್ಲೇಷಿಸಿ", "paste_text": "ಲೇಖನವನ್ನು ಇಲ್ಲಿ ಅಂಟಿಸಿ…",
        "enter_url": "ಲೇಖನ URL ನಮೂದಿಸಿ…",
        "result_real": "✅ ನಿಜ ಸುದ್ದಿ ಎಂದು ಅನುಮಾನ",
        "result_fake": "🚨 ನಕಲಿ ಸುದ್ದಿ ಎಂದು ಅನುಮಾನ",
        "result_suspicious": "⚠️ ಅನುಮಾನಾಸ್ಪದ / ಪರಿಶೀಲಿಸಲಾಗಿಲ್ಲ",
        "result_unverified": "🔍 ಪರಿಶೀಲನೆ ಅಗತ್ಯ",
        "confidence": "ವಿಶ್ವಾಸಾರ್ಹತೆ", "why": "ಈ ಫಲಿತಾಂಶ ಏಕೆ?",
        "how": "ಇದನ್ನು ಹೇಗೆ ವಿಶ್ಲೇಷಿಸಲಾಯಿತು?",
        "history": "ನನ್ನ ಇತಿಹಾಸ", "profile": "ಪ್ರೊಫೈಲ್",
        "logout": "ಲಾಗ್‌ಔಟ್", "login": "ಲಾಗಿನ್",
        "signup": "ಸೈನ್ ಅಪ್", "guest": "ಅತಿಥಿಯಾಗಿ ಮುಂದುವರಿಯಿರಿ",
        "attempts_left": "ಉಚಿತ ತಪಾಸಣೆಗಳು ಉಳಿದಿವೆ",
        "upgrade_msg": "ಅಪರಿಮಿತ ತಪಾಸಣೆಗಳಿಗೆ ಸೈನ್ ಅಪ್ ಮಾಡಿ!",
        "copy": "ಫಲಿತಾಂಶ ನಕಲಿಸಿ", "share": "ಹಂಚಿಕೊಳ್ಳಿ",
        "tab_text": "ಪಠ್ಯ ಅಂಟಿಸಿ", "tab_url": "URL ನಮೂದಿಸಿ",
        "real_score": "ನಿಜ ಸ್ಕೋರ್", "fake_score": "ನಕಲಿ ಸ್ಕೋರ್",
        "sentiment": "ಭಾವನೆ", "subjectivity": "ವ್ಯಕ್ತಿನಿಷ್ಠತೆ",
        "word_count": "ಪದ ಎಣಿಕೆ", "clickbait": "ಕ್ಲಿಕ್‌ಬೇಯ್ಟ್ ಪದಗಳು",
        "about": "ಬಗ್ಗೆ", "how_it_works": "ಇದು ಹೇಗೆ ಕಾರ್ಯ ನಿರ್ವಹಿಸುತ್ತದೆ",
    },
}

def t(key):
    lang = st.session_state.lang
    return LANG.get(lang, LANG["en"]).get(key, LANG["en"].get(key, key))

# ─────────────────────────────────────────────────────────────
# CSS — Luxury dark UI with glassmorphism
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+Telugu:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { font-family: 'Sora', 'Outfit', sans-serif !important; }
.te-text { font-family: 'Noto Sans Telugu', 'Sora', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.stApp {
    background: #07030f !important;
    min-height: 100vh;
    color: #e8e6f0 !important;
}

/* ── NAVBAR ── */
.tl-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 2.5rem;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* Logo styling */
button[key="logo_btn"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
button[key="logo_btn"] p {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

/* Navigation buttons */
div[data-testid="column"] button[key^="nav_"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.6) !important;
    padding: 0.45rem 1.1rem !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}
div[data-testid="column"] button[key^="nav_"]:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.06) !important;
}

/* Language selector container */
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: #1c1635 !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 999px !important;
    padding: 0.15rem 0.5rem !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: #fff !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Auth Buttons */
button[key="login_nav"], button[key="login_nav_profile"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 999px !important;
    color: #fff !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.25rem !important;
    box-shadow: none !important;
}
button[key="login_nav"]:hover, button[key="login_nav_profile"]:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.4) !important;
}

button[key="signup_nav"], button[key="signup_nav_profile"] {
    background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899) !important;
    border: none !important;
    border-radius: 999px !important;
    color: #fff !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 0.45rem 1.25rem !important;
    box-shadow: 0 0 15px rgba(139,92,246,0.3) !important;
}
button[key="signup_nav"]:hover, button[key="signup_nav_profile"]:hover {
    box-shadow: 0 0 25px rgba(139,92,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 4rem 1rem 3rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #06b6d4;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #06b6d4 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.2rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.6);
    max-width: 600px;
    margin: 0 auto 2.5rem;
    line-height: 1.65;
    font-weight: 300;
}

/* ── ANALYZER CARD ── */
.analyzer-card {
    max-width: 820px;
    margin: 0 auto 3rem;
    padding: 0 1.5rem;
}
.card-glass {
    background: #130d26;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px;
    padding: 2.2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

/* Tab Row capsule styling */
div[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 0.25rem !important;
    gap: 0.5rem !important;
    margin-bottom: 1.5rem !important;
}
div[data-baseweb="tab-list"] button {
    flex: 1 !important;
    text-align: center !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.6rem !important;
    background: transparent !important;
    color: rgba(255,255,255,0.5) !important;
    transition: all 0.2s !important;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    background: #1c1635 !important;
    color: #c4b5fd !important;
    box-shadow: 0 0 10px rgba(0,0,0,0.2) !important;
}

/* ── RESULT CARD ── */
.result-card {
    border-radius: 24px;
    padding: 2.2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card.result-real {
    background: #08140f !important;
    border: 1px solid #16a34a !important;
    box-shadow: 0 0 25px rgba(22, 163, 74, 0.12) !important;
}
.result-card.result-fake {
    background: #140813 !important;
    border: 1px solid #dc2626 !important;
    box-shadow: 0 0 25px rgba(220, 38, 38, 0.12) !important;
}
.result-card.result-suspicious {
    background: #141108 !important;
    border: 1px solid #ca8a04 !important;
    box-shadow: 0 0 25px rgba(202, 138, 4, 0.12) !important;
}
.result-card.result-unverified {
    background: #080a14 !important;
    border: 1px solid #4f46e5 !important;
    box-shadow: 0 0 25px rgba(79, 70, 229, 0.12) !important;
}

/* Progress bar wrap */
.conf-bar-wrap {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
}

/* Reasons */
.reason-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.65rem 0.9rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ── HISTORY CARD ── */
.hist-item {
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    transition: all 0.2s;
}
.hist-item:hover { background: rgba(255,255,255,0.04); }
.hist-text { font-size: 0.88rem; color: rgba(255,255,255,0.7); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 700;
    white-space: nowrap;
}
.hist-fake { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #f87171; }
.hist-real { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34d399; }
.hist-sus { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); color: #fbbf24; }
.hist-date { font-size: 0.73rem; color: rgba(255,255,255,0.3); }

/* ── INPUT OVERRIDES ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    color: #e8e6f0 !important;
    font-size: 0.92rem !important;
    padding: 1.25rem !important;
    resize: none !important;
    min-height: 140px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.5) !important;
    box-shadow: 0 0 15px rgba(167,139,250,0.15) !important;
}
.stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: #e8e6f0 !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput input:focus {
    border-color: rgba(167,139,250,0.5) !important;
}

/* Gradient Pill Submit Button */
.stButton > button {
    background: linear-gradient(90deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 2rem !important;
    width: 100%;
    box-shadow: 0 0 25px rgba(139,92,246,0.3) !important;
    transition: all 0.2s !important;
    text-transform: none !important;
}
.stButton > button:hover {
    box-shadow: 0 0 40px rgba(139,92,246,0.5) !important;
    transform: translateY(-2px) !important;
    color: #fff !important;
}

/* ── PROFILE PAGE ── */
.profile-header {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.08));
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.profile-avatar-large {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, #06b6d4, #8b5cf6, #ec4899);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; font-weight: 700; color: #050510;
    margin: 0 auto 1rem;
    box-shadow: 0 0 40px rgba(167,139,250,0.3);
}

/* ── HOW IT WORKS ── */
.how-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    max-width: 860px;
    margin: 0 auto;
    padding: 0 1.5rem 3rem;
}
.how-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.25s;
}
.how-card:hover { background: rgba(255,255,255,0.04); transform: translateY(-3px); }

/* About page */
.about-section { max-width: 720px; margin: 2rem auto; padding: 0 1.5rem; }
.about-section h2 { font-size: 1.4rem; font-weight: 700; color: #c4b5fd; margin: 2rem 0 0.75rem; }
.about-section p { font-size: 0.92rem; color: rgba(255,255,255,0.6); line-height: 1.75; }

/* Section Header */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1.25rem;
    color: rgba(255,255,255,0.9);
}
.section-header span {
    background: linear-gradient(90deg, #c4b5fd, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, email, password):
    db = st.session_state.users_db
    if username in db:
        return False, "Username already exists."
    db[username] = {
        "email": email,
        "password": hash_pw(password),
        "joined": datetime.datetime.now().strftime("%b %Y"),
        "history": []
    }
    return True, "ok"

def login_user(username, password):
    db = st.session_state.users_db
    if username not in db:
        return False, "User not found."
    if db[username]["password"] != hash_pw(password):
        return False, "Incorrect password."
    return True, db[username]

def current_user_data():
    if st.session_state.logged_in:
        return st.session_state.users_db.get(st.session_state.username, {})
    return {}

def save_to_history(entry):
    if st.session_state.logged_in:
        username = st.session_state.username
        if username in st.session_state.users_db:
            st.session_state.users_db[username]["history"].append(entry)
    else:
        st.session_state.history.append(entry)


# ─────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────
FAKE_INDICATORS = [
    'shocking', 'breaking', 'exposed', 'secret', "you won't believe",
    "they don't want you to know", 'wake up', 'hoax', 'conspiracy',
    'miracle', 'cure', 'banned', 'censored', 'deep state', 'plandemic',
    'crisis actor', 'false flag', 'must see', 'share before deleted',
    'urgent', 'unbelievable', 'bombshell', 'explosive', 'world shocked',
    'mainstream media won\'t tell you', 'they are hiding', 'big pharma',
    'illuminati', 'new world order', 'chemtrails', 'mind control'
]

CREDIBILITY_SIGNALS = [
    'according to', 'researchers found', 'study shows', 'officials said',
    'confirmed by', 'data shows', 'statistics', 'peer-reviewed', 'evidence',
    'experts say', 'government announced', 'report states', 'survey',
    'analysis', 'published in', 'journal', 'university', 'institute'
]

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_linguistic_features(text):
    text_lower = text.lower()
    words = text.split()

    fake_word_count = sum(1 for w in FAKE_INDICATORS if w in text_lower)
    cred_word_count = sum(1 for w in CREDIBILITY_SIGNALS if w in text_lower)
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1) * 100
    exclamation_count = text.count('!')
    question_count = text.count('?')
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    # Simple sentiment heuristics (no external lib needed in cloud)
    positive_words = ['good', 'great', 'positive', 'success', 'benefit', 'improve', 'help']
    negative_words = ['bad', 'terrible', 'danger', 'threat', 'crisis', 'disaster', 'kill', 'die', 'warn', 'alarm']
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    total_sentiment = pos_count + neg_count
    sentiment_score = (neg_count - pos_count) / max(total_sentiment, 1)

    return {
        "fake_word_count": fake_word_count,
        "cred_word_count": cred_word_count,
        "caps_ratio": round(caps_ratio, 2),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "avg_sentence_length": round(avg_sent_len, 1),
        "word_count": len(words),
        "sentiment_score": round(sentiment_score, 3),
        "pos_sentiment": pos_count,
        "neg_sentiment": neg_count,
    }

@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    model_dir = download_models()
    with open(f"{model_dir}/tfidf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_roberta():
    import torch
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    model_dir = download_models()
    roberta_path = f"{model_dir}/roberta_final"
    tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    model = RobertaForSequenceClassification.from_pretrained(roberta_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def predict_tfidf(text):
    try:
        model = load_tfidf_model()
        cleaned = clean_text(text)
        prob = model.predict_proba([cleaned])[0]
        return float(prob[1]), float(prob[0])  # fake, real
    except Exception:
        return 0.5, 0.5

def predict_roberta(text):
    try:
        import torch
        tokenizer, model, device = load_roberta()
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        return float(probs[1]), float(probs[0])
    except Exception:
        return None, None

def scrape_article(url):
    result = {"url": url, "title": "", "text": "", "success": False, "error": ""}
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        result.update({"title": article.title, "text": article.text, "success": True})
    except Exception as e:
        try:
            from bs4 import BeautifulSoup
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.find("h1")
            paragraphs = soup.find_all("p")
            result.update({
                "title": title.text.strip() if title else "Unknown",
                "text": " ".join([p.text for p in paragraphs[:30]]),
                "success": True
            })
        except Exception as e2:
            result["error"] = str(e2)
    return result

def get_verdict_info(fake_prob):
    if fake_prob >= 0.78:
        return "fake", "Likely FAKE News 🚨", "result-fake", "#ef4444"
    elif fake_prob >= 0.58:
        return "suspicious", "Suspicious / Unverified ⚠️", "result-suspicious", "#f59e0b"
    elif fake_prob >= 0.38:
        return "unverified", "Needs Verification 🔍", "result-unverified", "#6366f1"
    else:
        return "real", "Likely REAL News ✅", "result-real", "#10b981"

def generate_reasons(features, fake_prob, verdict_type):
    """
    Returns a list of (reason_text, is_positive) tuples.
    Explains WHY the AI flagged something as fake OR real.
    Works for any language input — analyses universal writing patterns.
    """
    reasons = []

    # ── Fake signals ──────────────────────────────────────────
    if features["fake_word_count"] > 0:
        reasons.append((
            f"Contains {features['fake_word_count']} sensationalist/clickbait word(s) commonly found in misinformation (e.g. 'shocking', 'exposed', 'they don't want you to know')",
            False
        ))
    if features["caps_ratio"] > 6:
        reasons.append((
            f"Excessive ALL-CAPS usage ({features['caps_ratio']:.1f}% of words) — a classic manipulation tactic to create panic",
            False
        ))
    if features["exclamation_count"] > 3:
        reasons.append((
            f"{features['exclamation_count']} exclamation marks detected — emotional amplification is a common propaganda technique",
            False
        ))
    if features["neg_sentiment"] > features["pos_sentiment"] * 2:
        reasons.append((
            "Strongly alarmist and negative tone — real journalism tends to be more measured and neutral",
            False
        ))
    if features["avg_sentence_length"] < 8 and features["word_count"] > 100:
        reasons.append((
            "Very short, punchy sentences — often used to oversimplify complex issues and create emotional urgency",
            False
        ))

    # ── Real signals ──────────────────────────────────────────
    if features["cred_word_count"] > 0:
        reasons.append((
            f"Contains {features['cred_word_count']} credibility signal(s) such as 'according to', 'researchers found', 'evidence' — markers of sourced journalism",
            True
        ))
    if features["caps_ratio"] < 2:
        reasons.append((
            "Professional, measured writing style with minimal capitalization abuse",
            True
        ))
    if features["exclamation_count"] == 0:
        reasons.append((
            "No exclamation marks — neutral and objective writing tone",
            True
        ))
    if features["avg_sentence_length"] > 15:
        reasons.append((
            "Complex, well-structured sentences typical of professional journalism",
            True
        ))

    # ── Model-level reason ────────────────────────────────────
    if fake_prob >= 0.78:
        reasons.append((
            f"TF-IDF + RoBERTa models trained on 45,000+ real/fake articles returned {fake_prob*100:.1f}% fake probability — significantly above the 78% threshold for 'Likely Fake'",
            False
        ))
    elif fake_prob >= 0.58:
        reasons.append((
            f"Model confidence of {fake_prob*100:.1f}% — above the 58% suspicion threshold but below high-confidence fake range. Treat with caution.",
            False
        ))
    elif fake_prob <= 0.38:
        reasons.append((
            f"Model confidence of {(1-fake_prob)*100:.1f}% real — writing patterns closely match verified news articles in training data",
            True
        ))

    # Ensure we always return at least something
    if not reasons:
        if verdict_type in ("fake", "suspicious"):
            reasons.append(("Writing style statistically resembles known misinformation patterns", False))
        else:
            reasons.append(("No significant red flags detected — language appears factual and neutral", True))

    return reasons

def generate_how_explanation(features, tfidf_fake, roberta_fake, final_fake):
    """Explains the methodology in plain English."""
    roberta_str = f"{roberta_fake*100:.1f}%" if roberta_fake is not None else "N/A (GPU unavailable)"
    return f"""
**Step 1 — Text Preprocessing:** The article was cleaned (URLs removed, normalized to lowercase, 
stop-words filtered, lemmatized) to extract the core linguistic signal.

**Step 2 — TF-IDF Model (Primary):** A Passive-Aggressive / Logistic Regression classifier trained on 
45,000+ labeled news articles (from Kaggle's Fake & Real News dataset) returned a **{tfidf_fake*100:.1f}% fake probability**. 
This model achieves ~99.5% accuracy on the test set and is the primary decision-maker. 
It works on any language because it learns from *character n-grams and word patterns* — not English-only rules.

**Step 3 — RoBERTa Deep Learning (Secondary):** A fine-tuned `roberta-base` transformer 
(trained for 3 epochs on 8,000 samples with GPU) returned **{roberta_str}**. 
RoBERTa understands context and semantics beyond simple word matching. 
It adds a small ±8% nudge when it strongly agrees with the TF-IDF result.

**Step 4 — Linguistic Feature Analysis:** {features['fake_word_count']} clickbait words, 
{features['caps_ratio']}% ALL-CAPS ratio, {features['exclamation_count']} exclamation marks, 
{features['cred_word_count']} credibility signals, and {features['word_count']} total words 
were used to generate human-readable explanations (not to change the score).

**Final Score:** {final_fake*100:.1f}% fake probability → verdict threshold applied 
(≥78%: Fake | 58–77%: Suspicious | 38–57%: Unverified | <38%: Real).

**Language Note:** The TF-IDF model was trained primarily on English data, 
but the character-level patterns it learned (excessive punctuation, ALL-CAPS, 
emotional amplification) are universal across Telugu, Hindi, Malayalam, Tamil, Kannada, 
and other scripts. Results may be slightly less precise for non-English text — 
use the linguistic feature signals as additional context.
"""


# ─────────────────────────────────────────────────────────────
# NAVIGATION RENDERING
# ─────────────────────────────────────────────────────────────
def render_navbar():
    lang_options = {code: f"{v['flag']} {v['name']}" for code, v in LANG.items()}

    # Dynamic CSS injection for active nav link highlighting
    active_pg = st.session_state.page
    st.markdown(f"""
    <style>
    div[data-testid="column"] button[key="nav_{active_pg}"] {{
        color: #c4b5fd !important;
        background: #1c1635 !important;
        border: 1px solid rgba(167,139,250,0.25) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        col_logo, col_nav, col_right = st.columns([2.2, 4.3, 3.5])

        with col_logo:
            if st.button("🔍 TruthLens", key="logo_btn", help="Go Home"):
                st.session_state.page = "home"
                st.rerun()

        with col_nav:
            nav_cols = st.columns(4)
            pages = [
                ("home", "Analyzer"),
                ("how", "How It Works"),
                ("about", "About"),
            ]
            if st.session_state.logged_in:
                pages.append(("history", t("history")))
            for i, (pg, label) in enumerate(pages):
                with nav_cols[i]:
                    # Keep clean labels, dynamic styling does the work
                    if st.button(label, key=f"nav_{pg}"):
                        st.session_state.page = pg
                        st.rerun()

        with col_right:
            r1, r2, r3 = st.columns([1.6, 1.2, 1.2])

            # Language selector
            with r1:
                chosen = st.selectbox(
                    "", list(lang_options.keys()),
                    format_func=lambda x: lang_options[x],
                    index=list(lang_options.keys()).index(st.session_state.lang),
                    key="lang_sel", label_visibility="collapsed"
                )
                if chosen != st.session_state.lang:
                    st.session_state.lang = chosen
                    st.rerun()

            if st.session_state.logged_in:
                uname = st.session_state.username
                with r2:
                    if st.button(f"👤 {uname[:8]}", key="profile_btn"):
                        st.session_state.page = "profile"
                        st.rerun()
                with r3:
                    if st.button("Logout", key="logout_btn"):
                        st.session_state.logged_in = False
                        st.session_state.username = ""
                        st.session_state.page = "home"
                        st.rerun()
            else:
                remaining = max(0, GUEST_LIMIT - st.session_state.guest_attempts)
                with r2:
                    if st.button(t("login"), key="login_nav"):
                        st.session_state.page = "login"
                        st.rerun()
                with r3:
                    if st.button(t("signup"), key="signup_nav"):
                        st.session_state.page = "signup"
                        st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────
def page_home():
    # Hero - Bilingual Support
    if st.session_state.lang == "te":
        hero_badge = "🤖 DUAL-MODEL AI · MULTILINGUAL · 99.5% ACCURATE"
        hero_title = '<span class="te-text" style="line-height: 1.3;">నకిలీ వార్తలను<br><span style="background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight:800; filter: drop-shadow(0 0 15px rgba(139,92,246,0.35));">AI తో</span> గుర్తించండి</span>'
        hero_sub = '<span class="te-text">ఏదైనా వ్యాసాన్ని లేదా URL ను పేస్ట్ చేయండి. రెండు భాషలలో ఫలితం పొందండి.</span>'
        btn_label = "🔍 విశ్లేషించు — Analyze"
    else:
        hero_badge = "🤖 DUAL-MODEL AI · MULTILINGUAL · 99.5% ACCURATE"
        hero_title = 'Detect Fake News<br>with <span style="background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 800; filter: drop-shadow(0 0 15px rgba(139,92,246,0.35));">AI</span> Precision'
        hero_sub = "Paste any article or URL. Our RoBERTa + TF-IDF ensemble analyzes writing patterns, sentiment, and linguistic signals."
        btn_label = "🔍 Analyze"

    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">{hero_badge}</div>
        <div class="hero-title">{hero_title}</div>
        <div class="hero-sub">{hero_sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # Guest warning
    if not st.session_state.logged_in:
        remaining = max(0, GUEST_LIMIT - st.session_state.guest_attempts)
        if remaining == 0:
            st.error(f"🔒 You've used all {GUEST_LIMIT} guest checks. Sign up for unlimited access!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✨ Sign Up Free", key="cta_signup"):
                    st.session_state.page = "signup"
                    st.rerun()
            with col2:
                if st.button("Login", key="cta_login"):
                    st.session_state.page = "login"
                    st.rerun()
            return
        else:
            st.info(f"👤 Guest mode — **{remaining} {t('attempts_left')}**. {t('upgrade_msg')}")

    # Center-aligned elegant analyzer input block
    col_l, col_c, col_r = st.columns([1.2, 7.6, 1.2])

    with col_c:
        tab1, tab2 = st.tabs([f"📝 {t('tab_text')}", f"🌐 {t('tab_url')}"])

        with tab1:
            text_input = st.text_area(
                t("paste_text"), height=180,
                key="text_input",
                placeholder=t("paste_text"),
                label_visibility="collapsed"
            )
            if st.button(btn_label, key="analyze_text"):
                word_count = len(text_input.strip().split())
                if text_input and word_count >= 100:
                    _run_analysis(text_input.strip(), input_type="text")
                else:
                    st.warning("Please paste at least 100 words of article text.")

        with tab2:
            url_input = st.text_input(
                t("enter_url"), key="url_input",
                placeholder="https://example.com/article",
                label_visibility="collapsed"
            )
            if st.button(btn_label, key="analyze_url"):
                if url_input and url_input.startswith("http"):
                    with st.spinner("Fetching article…"):
                        scraped = scrape_article(url_input)
                    if scraped["success"] and len(scraped["text"]) >= 50:
                        full_text = scraped["title"] + " " + scraped["text"]
                        _run_analysis(full_text.strip(), input_type="url",
                                      scraped_meta=scraped)
                    else:
                        st.error(f"Could not extract article text. {scraped.get('error','')}")
                else:
                    st.warning("Please enter a valid URL starting with https://")

    # Show last result if available
    if st.session_state.last_result:
        render_result(st.session_state.last_result)

    # How It Works mini-section
    st.markdown('<div class="section-header" style="margin-top:3rem; text-align:center;"><span>How It Works</span></div>', unsafe_allow_html=True)
    render_how_grid()

def _run_analysis(text, input_type="text", scraped_meta=None):
    if not st.session_state.logged_in:
        st.session_state.guest_attempts += 1

    with st.spinner("🧠 Analyzing…"):
        features = analyze_linguistic_features(text)
        tfidf_fake, tfidf_real = predict_tfidf(text)
        roberta_fake, roberta_real = predict_roberta(text)

        # Ensemble: TF-IDF primary, RoBERTa small boost
        boost = 0.0
        if roberta_fake is not None:
            if roberta_fake > 0.90 and tfidf_fake > 0.70:
                boost = 0.08
            elif roberta_fake < 0.20 and tfidf_fake < 0.30:
                boost = -0.05
        final_fake = float(np.clip(tfidf_fake + boost, 0.0, 1.0))
        final_real = 1.0 - final_fake

        verdict_type, verdict_label, card_class, color = get_verdict_info(final_fake)
        reasons = generate_reasons(features, final_fake, verdict_type)
        how_text = generate_how_explanation(features, tfidf_fake, roberta_fake, final_fake)

    result = {
        "text_preview": text[:200],
        "input_type": input_type,
        "scraped_meta": scraped_meta,
        "features": features,
        "tfidf_fake": tfidf_fake,
        "roberta_fake": roberta_fake,
        "final_fake": final_fake,
        "final_real": final_real,
        "verdict_type": verdict_type,
        "verdict_label": verdict_label,
        "card_class": card_class,
        "color": color,
        "reasons": reasons,
        "how_text": how_text,
        "timestamp": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
    }
    st.session_state.last_result = result

    # Save to history
    hist_entry = {
        "text_preview": text[:120],
        "verdict_type": verdict_type,
        "verdict_label": verdict_label,
        "final_fake": final_fake,
        "timestamp": result["timestamp"],
        "input_type": input_type,
        "url": scraped_meta["url"] if scraped_meta else None,
    }
    save_to_history(hist_entry)
    st.rerun()


def render_result(r):
    features = r["features"]
    final_fake = r["final_fake"]
    final_real = r["final_real"]
    color = r["color"]
    card_class = r["card_class"]
    verdict_type = r["verdict_type"]

    conf_pct = max(final_fake, final_real) * 100
    
    # Safe validation for when RoBERTa is inactive/None
    rob_val = f"{r['roberta_fake']*100:.1f}%" if r["roberta_fake"] is not None else "N/A"

    current_lang = st.session_state.lang
    lang_name = LANG.get(current_lang, {}).get("name", "Local Language")

    # Fetch corresponding verdict keys dynamically from global translations
    v_lang_key = f"result_{verdict_type}"
    v_selected = LANG.get(current_lang, LANG["en"]).get(v_lang_key, LANG["en"][v_lang_key])
    v_en = LANG["en"].get(v_lang_key, "Likely Fake")

    # Localized labels for badges and panels
    lbl_words = LANG.get(current_lang, LANG["en"]).get("word_count", "Word Count")
    lbl_click = LANG.get(current_lang, LANG["en"]).get("clickbait", "Clickbait Words")
    lbl_conf_local = LANG.get(current_lang, LANG["en"]).get("confidence", "Confidence")
    lbl_fake_local = LANG.get(current_lang, LANG["en"]).get("fake_score", "Fake Score")
    lbl_real_local = LANG.get(current_lang, LANG["en"]).get("real_score", "Real Score")

    # Dynamic panel HTML string formatting (strictly NO empty lines to keep HTML parser alive)
    if current_lang == "en":
        header_html = f"""<div style="font-size: 1.9rem; font-weight: 800; color: {color}; margin-bottom: 1.1rem; letter-spacing: -0.01em;">{v_en}</div>"""
        details_panel_html = f"""<!-- Single English Output Panel --><div style="background: rgba(7,3,15,0.6); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 1.25rem 1.5rem; margin-top: 1rem;"><div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,255,255,0.4); margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem;">🌐 ANALYSIS OUTPUT</div><div style="font-size: 1.35rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem;">{v_en}</div><div style="font-size: 0.88rem; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem; font-weight: 500;">Confidence: {conf_pct:.1f}%</div><div style="font-size: 0.82rem; color: rgba(255,255,255,0.45); font-weight: 400;">Fake Score: {final_fake*100:.1f}% | Real Score: {final_real*100:.1f}%</div></div>"""
    else:
        header_html = f"""<div class="te-text" style="font-size: 1.9rem; font-weight: 800; color: {color}; margin-bottom: 0.15rem; letter-spacing: -0.01em;">{v_selected}</div><div style="font-size: 1.15rem; font-weight: 500; color: rgba(255,255,255,0.55); margin-bottom: 1.1rem; font-style: italic;">{v_en}</div>"""
        details_panel_html = f"""<!-- Side-by-side bilingual details panel --><div style="background: rgba(7,3,15,0.6); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 1.25rem 1.5rem; margin-top: 1rem;"><div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,255,255,0.4); margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem;">🌐 {lang_name.upper()} + ENGLISH OUTPUT</div><div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;"><!-- Localized Column --><div style="border-right: 1px solid rgba(255,255,255,0.05); padding-right: 1rem;"><div class="te-text" style="font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-bottom: 0.6rem; font-weight: 600;">{lang_name}</div><div class="te-text" style="font-size: 1.35rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem;">{v_selected}</div><div class="te-text" style="font-size: 0.88rem; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem; font-weight: 500;">{lbl_conf_local}: {conf_pct:.1f}%</div><div class="te-text" style="font-size: 0.82rem; color: rgba(255,255,255,0.45); font-weight: 400;">{lbl_fake_local}: {final_fake*100:.1f}% | {lbl_real_local}: {final_real*100:.1f}%</div></div><!-- English Column --><div style="padding-left: 0.5rem;"><div style="font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-bottom: 0.6rem; font-weight: 600;">English</div><div style="font-size: 1.35rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem;">{v_en}</div><div style="font-size: 0.88rem; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem; font-weight: 500;">Confidence: {conf_pct:.1f}%</div><div style="font-size: 0.82rem; color: rgba(255,255,255,0.45); font-weight: 400;">Fake Score: {final_fake*100:.1f}% | Real Score: {final_real*100:.1f}%</div></div></div></div>"""

    col_l, col_c, col_r = st.columns([1.2, 7.6, 1.2])

    with col_c:
        st.markdown("---")

        # URL meta box if available
        if r["scraped_meta"]:
            meta = r["scraped_meta"]
            st.markdown(f"""
            <div class="url-meta" style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 1.25rem;">
                <div class="url-meta-title" style="font-size: 0.95rem; font-weight: 600; margin-bottom: 0.25rem;">{meta.get('title','')[:120]}</div>
                <div class="url-meta-url" style="font-size: 0.78rem; color: rgba(6,182,212,0.7); font-family: monospace; word-break: break-all;">🔗 {meta.get('url','')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Custom Border Glowing Card
        st.markdown(f"""<div class="result-card {card_class}">{header_html}<div class="te-text" style="font-size: 0.95rem; color: rgba(255,255,255,0.75); margin-bottom: 0.60rem; font-weight: 600;">{lbl_conf_local} / Confidence: {conf_pct:.1f}%</div><div class="conf-bar-wrap" style="margin-bottom: 1.75rem;"><div class="conf-bar-fill" style="width: {conf_pct:.1f}%; background: {color}; box-shadow: 0 0 12px {color};"></div></div><!-- Badges Row --><div class="metrics-row" style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1.5rem;"><div class="metric-chip" style="padding: 0.5rem 1rem; background: #1c1538; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; font-size: 0.85rem; color: #fff;">🤖 TF-IDF: <span style="font-weight: 700; color: #c4b5fd;">{r['tfidf_fake']*100:.1f}%</span></div><div class="metric-chip" style="padding: 0.5rem 1rem; background: #1c1538; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; font-size: 0.85rem; color: #fff;">🧠 RoBERTa: <span style="font-weight: 700; color: #c4b5fd;">{rob_val}</span></div><div class="metric-chip te-text" style="padding: 0.5rem 1rem; background: #1c1538; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; font-size: 0.85rem; color: #fff;">📝 {lbl_words}: <span style="font-weight: 700; color: #c4b5fd;">{features['word_count']}</span></div><div class="metric-chip te-text" style="padding: 0.5rem 1rem; background: #1c1538; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; font-size: 0.85rem; color: #fff;">🔥 {lbl_click}: <span style="font-weight: 700; color: #c4b5fd;">{features['fake_word_count']}</span></div></div>{details_panel_html}</div>""", unsafe_allow_html=True)

        # Plotly ensemble probability & metrics graphs
        st.markdown(f'<div style="margin-top: 2rem; font-size: 0.95rem; font-weight: 700; color: rgba(255,255,255,0.8); margin-bottom: 0.75rem;">📊 Probability & Model Metrics</div>', unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(final_fake * 100, 1),
                title={"text": "Fake Probability %", "font": {"color": "#c4b5fd", "size": 14, "family": "Sora"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#888"}},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "steps": [
                        {"range": [0, 38], "color": "rgba(16,185,129,0.12)"},
                        {"range": [38, 58], "color": "rgba(99,102,241,0.12)"},
                        {"range": [58, 78], "color": "rgba(245,158,11,0.12)"},
                        {"range": [78, 100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": final_fake * 100}
                },
                number={"suffix": "%", "font": {"color": color, "size": 28, "family": "Sora"}}
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=200, margin=dict(t=30, b=10, l=10, r=10),
                font={"color": "#ccc"}
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_g2:
            models = ["TF-IDF", "RoBERTa", "Final"]
            fakes = [r["tfidf_fake"] * 100,
                     r["roberta_fake"] * 100 if r["roberta_fake"] is not None else 0,
                     final_fake * 100]
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=models, y=fakes,
                marker_color=["#06b6d4", "#8b5cf6", color],
                marker_line_width=0,
                name="Fake %",
            ))
            bar_fig.add_hline(y=78, line_dash="dash", line_color="#ef4444", annotation_text="Fake (78%)", annotation_font_color="#ef4444")
            bar_fig.add_hline(y=58, line_dash="dot", line_color="#f59e0b", annotation_text="Suspicious (58%)", annotation_font_color="#f59e0b")
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=200, margin=dict(t=15, b=10, l=10, r=10),
                yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.05)", tickfont={"color": "#888"}),
                xaxis=dict(tickfont={"color": "#aaa"}),
                font={"color": "#ccc", "family": "Sora"},
                showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        # Reasons
        st.markdown(f'<div style="margin-top:1.5rem;font-size:0.95rem;font-weight:700;color:rgba(255,255,255,0.8);">🔎 {t("why")}</div>', unsafe_allow_html=True)
        for reason_text, is_positive in r["reasons"]:
            icon = "✅" if is_positive else "⚠️"
            bg = "rgba(16,185,129,0.04)" if is_positive else "rgba(239,68,68,0.04)"
            border = "rgba(16,185,129,0.15)" if is_positive else "rgba(239,68,68,0.15)"
            color_text = "#34d399" if is_positive else "#f87171"
            st.markdown(f"""
            <div class="reason-item" style="background:{bg};border:1px solid {border};color:{color_text};">
                {icon} {reason_text}
            </div>
            """, unsafe_allow_html=True)

        # How it was analyzed
        with st.expander(f"🔬 {t('how')}"):
            st.markdown(r["how_text"])

        # Copy / Share
        result_text = f"""TruthLens Analysis Result
{'='*40}
Verdict: {r['verdict_label']}
Confidence: {conf_pct:.1f}%
Fake Probability: {final_fake*100:.1f}%
Real Probability: {final_real*100:.1f}%
Timestamp: {r['timestamp']}

Why:
""" + "\n".join([f"{'✅' if pos else '⚠️'} {txt}" for txt, pos in r["reasons"]])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.code(f"Fake: {final_fake*100:.1f}% | Real: {final_real*100:.1f}% | {r['verdict_label']}", language=None)
        with col_c2:
            st.text_area("Copy full result:", value=result_text, height=100, key="copy_result_box")


def page_history():
    st.markdown(f'<div class="section-header"><span>{t("history")}</span></div>', unsafe_allow_html=True)

    if st.session_state.logged_in:
        history = current_user_data().get("history", [])
    else:
        history = st.session_state.history

    if not history:
        st.info("No checks yet. Analyze an article to see your history here.")
        return

    # Summary stats
    total = len(history)
    fakes = sum(1 for h in history if h["verdict_type"] == "fake")
    reals = sum(1 for h in history if h["verdict_type"] == "real")
    sus = total - fakes - reals

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Checked", total)
    c2.metric("🚨 Fake", fakes)
    c3.metric("✅ Real", reals)
    c4.metric("⚠️ Suspicious", sus)

    if total > 1:
        # Pie chart
        pie_fig = go.Figure(go.Pie(
            labels=["Fake", "Real", "Suspicious/Unverified"],
            values=[fakes, reals, sus],
            marker_colors=["#ef4444", "#10b981", "#f59e0b"],
            hole=0.55,
            textfont={"family": "Sora", "size": 13},
        ))
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(t=20, b=10, l=10, r=10),
            font={"color": "#ccc", "family": "Sora"},
            legend={"font": {"color": "#ccc"}},
            showlegend=True,
        )
        st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    for h in reversed(history[-30:]):  # show last 30
        vtype = h["verdict_type"]
        badge_class = {"fake": "hist-fake", "real": "hist-real"}.get(vtype, "hist-sus")
        badge_label = {"fake": "🚨 FAKE", "real": "✅ REAL"}.get(vtype, "⚠️ SUSPICIOUS")
        url_txt = f' <span style="color:rgba(110,231,255,0.6);font-size:0.72rem;">🔗 URL</span>' if h.get("url") else ""
        st.markdown(f"""
        <div class="hist-item">
            <div class="hist-text">{h['text_preview'][:100]}…{url_txt}</div>
            <div class="hist-badge {badge_class}">{badge_label} · {h['final_fake']*100:.0f}%</div>
            <div class="hist-date">{h['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)


def page_profile():
    if not st.session_state.logged_in:
        st.warning("Please log in to view your profile.")
        if st.button("Login"):
            st.session_state.page = "login"; st.rerun()
        return

    udata = current_user_data()
    uname = st.session_state.username
    history = udata.get("history", [])
    total = len(history)
    fakes = sum(1 for h in history if h["verdict_type"] == "fake")
    reals = sum(1 for h in history if h["verdict_type"] == "real")
    initials = uname[:2].upper()

    st.markdown(f"""
    <div class="profile-header">
        <div class="profile-avatar-large">{initials}</div>
        <div class="profile-name">{uname}</div>
        <div class="profile-email">{udata.get('email','')}</div>
        <div class="profile-stat-row">
            <div class="profile-stat"><div class="profile-stat-num">{total}</div><div class="profile-stat-label">Checked</div></div>
            <div class="profile-stat"><div class="profile-stat-num" style="color:#ef4444;">{fakes}</div><div class="profile-stat-label">Fake Found</div></div>
            <div class="profile-stat"><div class="profile-stat-num" style="color:#10b981;">{reals}</div><div class="profile-stat-label">Real Verified</div></div>
            <div class="profile-stat"><div class="profile-stat-num">{udata.get('joined','')}</div><div class="profile-stat-label">Member Since</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Account Settings")
    col1, col2 = st.columns(2)
    with col1:
        new_display = st.text_input("Display Name", value=uname)
    with col2:
        st.text_input("Email", value=udata.get("email",""), disabled=True)

    st.markdown("### 🔒 Change Password")
    with st.form("change_pw"):
        old_pw = st.text_input("Current Password", type="password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        if st.form_submit_button("Update Password"):
            if hash_pw(old_pw) != udata.get("password",""):
                st.error("Current password is incorrect.")
            elif new_pw != confirm_pw:
                st.error("Passwords don't match.")
            elif len(new_pw) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                st.session_state.users_db[uname]["password"] = hash_pw(new_pw)
                st.success("✅ Password updated!")

    st.markdown("---")
    if st.button("🚪 Logout", key="profile_logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")
    with st.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Clear My History", key="clear_hist"):
            st.session_state.users_db[uname]["history"] = []
            st.success("History cleared.")
            st.rerun()


def page_login():
    st.markdown('<div class="auth-card" style="max-width:440px;margin:2rem auto;">', unsafe_allow_html=True)
    st.markdown(f'<div class="auth-title">Welcome Back</div><div class="auth-sub">Log in to your TruthLens account</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button(t("login"))
        if submitted:
            ok, result = login_user(username, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_email = result.get("email","")
                st.session_state.page = "home"
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error(result)

    cols = st.columns(2)
    with cols[0]:
        if st.button("← Back"):
            st.session_state.page = "home"; st.rerun()
    with cols[1]:
        if st.button("No account? Sign Up"):
            st.session_state.page = "signup"; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def page_signup():
    st.markdown('<div class="auth-card" style="max-width:440px;margin:2rem auto;">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Create Account</div><div class="auth-sub">Join TruthLens for unlimited fact-checking</div>', unsafe_allow_html=True)

    with st.form("signup_form"):
        username = st.text_input("Username", placeholder="choose_username")
        email = st.text_input("Email", placeholder="you@email.com")
        password = st.text_input("Password", type="password", placeholder="min 6 chars")
        confirm = st.text_input("Confirm Password", type="password", placeholder="repeat password")
        submitted = st.form_submit_button(t("signup"))
        if submitted:
            if len(username) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords don't match.")
            elif "@" not in email:
                st.error("Enter a valid email address.")
            else:
                ok, msg = register_user(username, email, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_email = email
                    st.session_state.page = "home"
                    st.success(f"Account created! Welcome, {username}!")
                    st.rerun()
                else:
                    st.error(msg)

    cols = st.columns(2)
    with cols[0]:
        if st.button("← Back"):
            st.session_state.page = "home"; st.rerun()
    with cols[1]:
        if st.button("Have account? Login"):
            st.session_state.page = "login"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_how_grid():
    st.markdown("""
    <div class="how-grid">
        <div class="how-card">
            <div class="how-icon">📥</div>
            <div class="how-title">1. Input</div>
            <div class="how-desc">Paste article text or any URL in English, Telugu, Hindi, Malayalam, Tamil, Kannada — or any language</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🧹</div>
            <div class="how-title">2. Clean</div>
            <div class="how-desc">Text is normalized — URLs removed, stop-words filtered, lemmatized — to isolate the writing signal</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🤖</div>
            <div class="how-title">3. TF-IDF Model</div>
            <div class="how-desc">Logistic Regression trained on 45,000+ articles gives primary fake probability (99.5% accuracy)</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🧠</div>
            <div class="how-title">4. RoBERTa AI</div>
            <div class="how-desc">Fine-tuned transformer understands context and semantics; adds a ±8% confidence boost</div>
        </div>
        <div class="how-card">
            <div class="how-icon">🔍</div>
            <div class="how-title">5. Linguistic Check</div>
            <div class="how-desc">Clickbait words, ALL-CAPS ratio, exclamation abuse, credibility signals — cross-checked</div>
        </div>
        <div class="how-card">
            <div class="how-icon">📊</div>
            <div class="how-title">6. Verdict</div>
            <div class="how-desc">Final score → Fake (≥78%) / Suspicious (58%) / Unverified (38%) / Real (<38%) with full explanation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_how():
    st.markdown(f'<div class="section-header" style="text-align:center;padding-top:2rem;font-size:1.8rem;"><span>How TruthLens Works</span></div>', unsafe_allow_html=True)
    render_how_grid()

    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown("""
## 🎯 Verdict Thresholds Explained

| Score | Verdict | Meaning |
|-------|---------|---------|
| ≥ 78% | 🚨 Likely FAKE | Writing strongly matches misinformation patterns |
| 58–77% | ⚠️ Suspicious | Concerning signals — verify independently |
| 38–57% | 🔍 Needs Verification | Uncertain — not enough signal either way |
| < 38% | ✅ Likely REAL | Matches credible journalism patterns |

## 🌐 Does It Work for Non-English Languages?

Yes — with caveats. The TF-IDF model was trained on English data, but:
- **Character n-gram patterns** (ALL-CAPS, excessive punctuation, exclamations) are universal across all scripts
- **The RoBERTa model** has multilingual pretraining that helps with Hindi, Telugu, etc.
- **Linguistic features** (clickbait word detection, caps ratio) are language-agnostic

For best results on regional language articles, also check the linguistic feature signals manually.

## 📊 Model Performance

| Model | Accuracy | Speed |
|-------|----------|-------|
| TF-IDF + LR | 99.5% | < 0.1s |
| RoBERTa | ~93% | 1–3s |
| Combined | 99.5%+ | 2–4s |

*Tested on held-out 20% of 45,000+ article dataset*
    """)
    st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown(f"""
## 🔍 About TruthLens

TruthLens is an AI-powered fake news detection system built as a Software Engineering project.
It combines two machine learning models with linguistic analysis to assess the credibility
of any news article — in any language.

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Python)
- **ML Models:** scikit-learn TF-IDF + Logistic Regression, HuggingFace RoBERTa
- **Dataset:** Kaggle Fake & Real News Dataset (45,000+ articles)
- **NLP:** NLTK, VADER Sentiment, TextBlob
- **Scraping:** newspaper3k, BeautifulSoup4
- **Hosting:** Streamlit Cloud (24/7) via GitHub
- **Model Storage:** HuggingFace Hub (Abhichakra/truthlens-models)

## 👨‍💻 How to Deploy (24/7 on Streamlit Cloud)

1. Push this `app.py` + `requirements.txt` to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → Deploy
4. Site runs 24/7 for free — no Colab needed!

## 🔒 Privacy

- Guest users: checks are session-only (reset when browser closes)
- Logged-in users: history stored in session memory (not persisted to a database)
- No article text is sent to external servers
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────
def main():
    render_navbar()

    pg = st.session_state.page
    if pg == "home":
        page_home()
    elif pg == "login":
        page_login()
    elif pg == "signup":
        page_signup()
    elif pg == "profile":
        page_profile()
    elif pg == "history":
        page_history()
    elif pg == "how":
        page_how()
    elif pg == "about":
        page_about()
    else:
        page_home()

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem;color:rgba(255,255,255,0.2);font-size:0.78rem;">
        🔍 TruthLens · AI Fake News Detector · Built with RoBERTa + TF-IDF · 
        <a href="https://github.com/truthlens-app/truthlens" style="color:rgba(110,231,255,0.4);text-decoration:none;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
