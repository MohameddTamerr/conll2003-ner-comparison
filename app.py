from pathlib import Path
import pandas as pd
import html
from inference import MODEL_NAMES, MODEL_FILES, load_model, predict, extract_entities
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="NER Demo | CoNLL-2003",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="locked"
)

ICONS = {
    'brain': '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M12 5v13"/>',
    'network': '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    'sparkles': '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/><path d="M19 3v4M21 5h-4M5 19v2M6 20H4"/>',
    'home': '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    'play': '<polygon points="6 3 20 12 6 21 6 3"/>',
    'chart': '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    'info': '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    'person': '<circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 0 0-16 0"/>',
    'org': '<rect width="16" height="20" x="4" y="2" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M8 10h.01"/><path d="M16 10h.01"/><path d="M8 14h.01"/><path d="M16 14h.01"/>',
    'pin': '<path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/>',
    'tag': '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>',
    'layers': '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    'file': '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    'nodes': '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    'globe': '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    'chip': '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    'check_circle': '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    'text': '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/>',
    'settings': '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
}

def icon(name, size=24):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root {
    --ink: #23174b;
    --purple: #6541a4;
    --purple-light: #7b55c6;
    --purple-dark: #502e8f;
    --surface: #ffffff;
}

html, body, [data-testid="stApp"] {
    font-family: Inter, Arial, sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 98% 1%, #efeaff 0%, transparent 40%), #fbfaff;
}

[data-testid="stHeader"] {
    background: transparent;
    height: 2.5rem;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    max-width: 1450px;
    padding: 2rem clamp(1rem, 3.2vw, 3.6rem) 4rem;
}

/* =========================================================
   STATIC NON-COLLAPSIBLE SIDEBAR
   Completely remove sidebar open/close buttons and controls
   ========================================================= */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarHeader"] button,
button[aria-label*="sidebar" i],
button[aria-label*="Sidebar" i],
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stSidebarResizer"] {
    display: none !important;
    pointer-events: none !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #322256 0%, #281a49 70%, #442a6c 100%) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: white;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
}

/* Sidebar Brand Header */
.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    color: #ffffff;
    font-weight: 800;
    font-size: clamp(20px, 1.5vw, 25px);
    padding: 4px 6px 28px;
    letter-spacing: -0.4px;
}

.brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(255, 255, 255, 0.06) 100%);
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 4px 14px rgba(18, 10, 36, 0.35);
    color: #e4d7ff;
    flex-shrink: 0;
}

/* =========================================================
   BUTTON HOVER & PRESS (ACTIVE) ANIMATIONS
   ========================================================= */

/* Primary Action Buttons (Cards, Analyze, etc.) */
div[data-testid="stMain"] button[kind="primary"],
.st-key-try_card button,
.st-key-compare_card button {
    background: linear-gradient(135deg, #7449c7 0%, #572fa1 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    letter-spacing: 0.2px !important;
    padding: 0.75rem 1.4rem !important;
    min-height: 2.9rem !important;
    box-shadow: 0 4px 14px -2px rgba(87, 47, 161, 0.38), 0 2px 4px rgba(0, 0, 0, 0.06) !important;
    cursor: pointer !important;
    position: relative !important;
    transform: translateY(0) scale(1) !important;
    transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.2s ease,
                background 0.2s ease,
                filter 0.2s ease !important;
}

/* Primary Button Hover */
div[data-testid="stMain"] button[kind="primary"]:hover,
.st-key-try_card button:hover,
.st-key-compare_card button:hover {
    background: linear-gradient(135deg, #8154d8 0%, #6336b1 100%) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.38) !important;
    transform: translateY(-2.5px) scale(1.015) !important;
    box-shadow: 0 10px 24px -3px rgba(87, 47, 161, 0.52), 0 4px 10px -2px rgba(87, 47, 161, 0.25) !important;
    filter: brightness(1.05) !important;
}

/* Primary Button Press / Active */
div[data-testid="stMain"] button[kind="primary"]:active,
.st-key-try_card button:active,
.st-key-compare_card button:active {
    transform: translateY(2px) scale(0.975) !important;
    box-shadow: 0 2px 6px rgba(87, 47, 161, 0.35) !important;
    filter: brightness(0.93) !important;
    transition: transform 0.07s ease, box-shadow 0.07s ease !important;
}

/* Primary Button Icon Slide on Hover */
div[data-testid="stMain"] button[kind="primary"] [data-testid="stIconMaterial"],
div[data-testid="stMain"] button[kind="primary"] .material-symbols-rounded,
.st-key-try_card button [data-testid="stIconMaterial"],
.st-key-compare_card button [data-testid="stIconMaterial"] {
    transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}

div[data-testid="stMain"] button[kind="primary"]:hover [data-testid="stIconMaterial"],
div[data-testid="stMain"] button[kind="primary"]:hover .material-symbols-rounded,
.st-key-try_card button:hover [data-testid="stIconMaterial"],
.st-key-compare_card button:hover [data-testid="stIconMaterial"] {
    transform: translateX(3px) !important;
}

/* Secondary Buttons (Quick Sample Pills & Action Badges) */
div[data-testid="stMain"] button[kind="secondary"] {
    background: #ffffff !important;
    color: #442e78 !important;
    border: 1.5px solid #ded5f4 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 0.9rem !important;
    box-shadow: 0 1px 3px rgba(35, 23, 75, 0.04) !important;
    transform: translateY(0) scale(1) !important;
    transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.18s ease,
                background 0.18s ease,
                border-color 0.18s ease !important;
}

div[data-testid="stMain"] button[kind="secondary"]:hover {
    background: #f4effe !important;
    color: #321c64 !important;
    border-color: #8f6cc9 !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 16px -2px rgba(101, 65, 164, 0.18) !important;
}

div[data-testid="stMain"] button[kind="secondary"]:active {
    transform: translateY(1.5px) scale(0.975) !important;
    box-shadow: 0 1px 3px rgba(101, 65, 164, 0.2) !important;
    background: #ece3fb !important;
    transition: transform 0.07s ease !important;
}

/* Sidebar Navigation Buttons */
[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-bottom: 8px !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 13px !important;
    padding: 0.85rem 1.15rem !important;
    font-size: 0.98rem !important;
    min-height: 3.25rem !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    background: transparent !important;
    color: #e3d9fa !important;
    font-weight: 600 !important;
    letter-spacing: 0.15px !important;
    transform: translateX(0) scale(1) !important;
    transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1),
                background 0.18s ease,
                color 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: rgba(255, 255, 255, 0.14) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 12px rgba(18, 9, 36, 0.25) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button:hover [data-testid="stIconMaterial"] {
    transform: scale(1.14) !important;
    transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button:active {
    transform: translateX(2px) scale(0.98) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    transition: transform 0.07s ease !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #7e57cf 0%, #5e37a3 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 4px 16px rgba(25, 12, 54, 0.45) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8a64db 0%, #6940b2 100%) !important;
    transform: translateX(4px) scale(1.01) !important;
    box-shadow: 0 6px 20px rgba(25, 12, 54, 0.55) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:active {
    transform: translateX(2px) scale(0.98) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button [data-testid="stIconMaterial"],
[data-testid="stSidebar"] div[data-testid="stButton"] button .material-symbols-rounded {
    font-size: 1.32rem !important;
    width: 1.32rem !important;
    height: 1.32rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: inherit !important;
    flex-shrink: 0 !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"] button p {
    margin: 0 !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: inherit !important;
}

/* =========================================================
   HOME PAGE: HERO & CARDS
   ========================================================= */
.hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(240px, 32%);
    align-items: center;
    gap: 20px;
    min-height: 270px;
    margin-bottom: 16px;
}
.hero-text { min-width: 0; }
.hero h1 {
    color: var(--ink);
    font-size: clamp(30px, 3.1vw, 48px);
    letter-spacing: -1.8px;
    line-height: 1.1;
    margin: 0;
    font-weight: 800;
}
.hero .subtitle {
    font-size: clamp(22px, 2vw, 33px);
    margin: 9px 0 21px;
    color: #2b205b;
    font-weight: 400;
}
.hero .intro {
    font-size: clamp(14px, 1.05vw, 17px);
    line-height: 1.7;
    color: #3c355d;
    max-width: 830px;
    margin: 0;
}
.hero-image {
    width: 100%;
    height: auto;
    max-height: 320px;
    object-fit: contain;
}

/* Entity Badges */
.entity-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 15px;
    margin: 20px 0 30px;
}
.entity {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 18px 16px;
    border-radius: 19px;
    min-height: 85px;
    min-width: 0;
    border: 1px solid rgba(0, 0, 0, 0.03);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.entity:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
}
.entity .label {
    font-size: clamp(16px, 1.3vw, 21px);
    font-weight: 800;
    line-height: 1.2;
}
.entity .small {
    font-size: clamp(10px, .9vw, 13px);
    margin-top: 5px;
    overflow-wrap: anywhere;
}
.entity svg { flex-shrink: 0; }
.entity.per { background: #f0edff; color: #6347a2; border-color: #dfd7f7; }
.entity.org { background: #e0f0ff; color: #245b9d; border-color: #c9e4fc; }
.entity.loc { background: #e1f8ed; color: #207856; border-color: #c7f0db; }
.entity.misc { background: #fff0db; color: #85420f; border-color: #f7ddbe; }

/* Interactive Action Cards on Home Page */
.st-key-try_card, .st-key-compare_card {
    border-radius: 22px !important;
    padding: 28px 30px !important;
    min-height: 220px !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    border: 1.5px solid transparent !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.st-key-try_card {
    background: linear-gradient(145deg, #f3effe 0%, #ebe4fb 100%) !important;
    border-color: #dfd4f8 !important;
}

.st-key-compare_card {
    background: linear-gradient(145deg, #eaf0ff 0%, #e1e9fc 100%) !important;
    border-color: #d1dffb !important;
}

.st-key-try_card:hover, .st-key-compare_card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 16px 32px -8px rgba(78, 48, 140, 0.15) !important;
}

.st-key-try_card:hover { border-color: #c6b2f0 !important; }
.st-key-compare_card:hover { border-color: #b3caff !important; }

.action-title {
    display: flex;
    align-items: center;
    gap: 14px;
    color: #24164b;
    font-size: clamp(20px, 1.7vw, 25px);
    font-weight: 800;
    margin: 0 0 10px;
}
.action-title svg {
    color: #6541a4;
    flex-shrink: 0;
}
.action-description {
    margin: 0 0 20px 0;
    color: #3b3060;
    font-size: 15px;
    line-height: 1.55;
    max-width: 420px;
}

/* Model cards */
.section-title {
    color: #271e54;
    font-weight: 800;
    font-size: clamp(22px, 1.7vw, 27px);
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 18px;
}
.section-title svg { color: #6541a4; flex-shrink: 0; }
.models {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 13px;
}
.model {
    border: 1px solid #e5e0fa;
    border-radius: 16px;
    padding: 24px 8px 19px;
    text-align: center;
    background: #fff;
    min-height: 143px;
    color: #2c205b;
    min-width: 0;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.model:hover {
    transform: translateY(-2.5px);
    border-color: #cfc2f0;
    box-shadow: 0 8px 20px rgba(85, 50, 145, 0.07);
}
.model .model-icon {
    color: #754fb5;
    margin: 0 auto 12px;
    display: flex;
    justify-content: center;
}
.model strong {
    display: block;
    font-size: clamp(13px, 1.06vw, 16px);
}
.model span {
    display: block;
    font-size: 12px;
    line-height: 1.45;
    margin-top: 5px;
    overflow-wrap: break-word;
}

/* Headers and Badges */
.page-header h1 {
    color: #23194f;
    font-size: clamp(30px, 3vw, 42px);
    letter-spacing: -1px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.page-header h1 svg { color: var(--purple); flex-shrink: 0; }
.page-header p { color: #655e7b; margin-bottom: 22px; font-size: 1.05rem; }

.legend {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 16px 0 24px;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid transparent;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.pill:hover {
    transform: translateY(-1.5px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
}
.pill svg { flex-shrink: 0; }
.pill.PER { background: #eee4ff; color: #5f3d9b; border-color: #dfd3f8; }
.pill.ORG { background: #e1f1ff; color: #265c9e; border-color: #cce5fd; }
.pill.LOC { background: #daf5e7; color: #17764e; border-color: #bee8d2; }
.pill.MISC { background: #fff0d8; color: #975112; border-color: #f7dcba; }

/* =========================================================
   TRY NER PAGE: FORM CONTROLS & DUAL PANELS
   ========================================================= */

/* Symmetrical Elevated Panels */
.st-key-analysis_panel, .st-key-result_panel {
    padding: 28px 30px !important;
    border: 1.5px solid #ece4f8 !important;
    border-radius: 22px !important;
    background: #ffffff !important;
    box-shadow: 0 6px 24px -4px rgba(50, 24, 96, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    min-height: 520px !important;
    display: flex !important;
    flex-direction: column !important;
    transition: all 0.25s ease !important;
}

.st-key-analysis_panel:hover, .st-key-result_panel:hover {
    border-color: #d8c8f2 !important;
    box-shadow: 0 10px 32px -4px rgba(50, 24, 96, 0.09) !important;
}

.panel-title {
    display: flex;
    align-items: center;
    gap: 11px;
    font-size: 1.25rem;
    font-weight: 800;
    color: #24164b;
    margin-bottom: 14px;
    letter-spacing: -0.3px;
}

.panel-title svg {
    color: #6541a4;
    flex-shrink: 0;
}

.sample-label {
    font-size: 0.84rem;
    font-weight: 600;
    color: #5c537a;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Light Theme Selectbox & BaseWeb Overrides */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #251a53 !important;
    border: 1.5px solid #d9cef2 !important;
    border-radius: 12px !important;
    min-height: 2.85rem !important;
    box-shadow: 0 1px 3px rgba(37, 26, 83, 0.03) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #8c68c9 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #6541a4 !important;
    box-shadow: 0 0 0 3px rgba(101, 65, 164, 0.16) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #251a53 !important;
    font-weight: 500 !important;
}

/* Dropdown popover menu */
div[data-baseweb="popover"], div[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid #dcd3f0 !important;
    box-shadow: 0 12px 30px -5px rgba(37, 26, 83, 0.16) !important;
}

div[data-baseweb="menu"] [role="option"] {
    color: #251a53 !important;
    background-color: transparent !important;
    border-radius: 9px !important;
    padding: 10px 14px !important;
    margin: 3px 6px !important;
    font-size: 0.94rem !important;
    transition: background 0.15s ease !important;
}

div[data-baseweb="menu"] [role="option"]:hover,
div[data-baseweb="menu"] [aria-selected="true"] {
    background-color: #f1ecfc !important;
    color: #4f2c94 !important;
    font-weight: 600 !important;
}

/* Light Theme Textarea */
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #24164b !important;
    border: 1.5px solid #d9cef2 !important;
    border-radius: 12px !important;
    font-family: inherit !important;
    font-size: 0.98rem !important;
    padding: 14px 16px !important;
    line-height: 1.6 !important;
    box-shadow: 0 1px 3px rgba(37, 26, 83, 0.03) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stTextArea"] textarea:hover {
    border-color: #8c68c9 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #6541a4 !important;
    box-shadow: 0 0 0 3px rgba(101, 65, 164, 0.16) !important;
    outline: none !important;
}

/* Form Labels & Captions with High Contrast */
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    color: #251a53 !important;
    font-weight: 650 !important;
    font-size: 0.93rem !important;
    margin-bottom: 6px !important;
}

[data-testid="stMain"] .stCaption,
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: #584f74 !important;
    font-size: 0.86rem !important;
    line-height: 1.5 !important;
}

/* Clean Empty State Illustration for Results */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 38px 24px;
    flex: 1;
    min-height: 380px;
    border-radius: 18px;
    background: linear-gradient(180deg, #faf8ff 0%, #f4f0fc 100%);
    border: 1.5px dashed #ded3f4;
}

.empty-icon {
    width: 62px;
    height: 62px;
    border-radius: 50%;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6c41b8;
    box-shadow: 0 6px 18px rgba(108, 65, 184, 0.15);
    margin-bottom: 16px;
}

.empty-state h3 {
    color: #24164b;
    font-size: 1.22rem;
    font-weight: 750;
    margin: 0 0 8px;
}

.empty-state p {
    color: #5c537a;
    font-size: 0.92rem;
    line-height: 1.6;
    max-width: 360px;
    margin: 0 0 20px;
}

.empty-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
}

.chip-item {
    font-size: 0.8rem;
    font-weight: 700;
    padding: 5px 11px;
    border-radius: 8px;
    letter-spacing: 0.2px;
}

.chip-item.PER { background: #eee4ff; color: #5f3d9b; }
.chip-item.ORG { background: #e1f1ff; color: #265c9e; }
.chip-item.LOC { background: #daf5e7; color: #17764e; }
.chip-item.MISC { background: #fff0d8; color: #975112; }

/* Status Bar & Result Token Stream */
.result-status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f0e9fa;
}

.count-badge {
    background: linear-gradient(135deg, #eee5ff 0%, #e4d7fb 100%);
    color: #502b94;
    font-weight: 750;
    font-size: 0.9rem;
    padding: 6px 13px;
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid #d9c7f6;
}

.model-badge {
    color: #584d76;
    font-size: 0.86rem;
    font-weight: 600;
}

.tokens {
    background: #faf8ff;
    border: 1px solid #ebe4f7;
    padding: 22px 24px;
    border-radius: 16px;
    line-height: 3;
    font-size: 17px;
    min-height: 100px;
    margin-bottom: 18px;
}

.token {
    display: inline-block;
    margin: 3px 5px;
    padding: 0 8px;
    border-radius: 7px;
    color: #241e46;
    font-weight: 500;
    transition: transform 0.15s ease;
}

.token:hover {
    transform: translateY(-1px);
}

.token.PER {
    background: #ebddff;
    color: #55279b;
    font-weight: 700;
    border: 1px solid #d4c0f5;
    box-shadow: 0 2px 5px rgba(85, 39, 155, 0.1);
}
.token.ORG {
    background: #d8edff;
    color: #125791;
    font-weight: 700;
    border: 1px solid #b8ddfd;
    box-shadow: 0 2px 5px rgba(18, 87, 145, 0.1);
}
.token.LOC {
    background: #d5f7e5;
    color: #0d7445;
    font-weight: 700;
    border: 1px solid #b5ebd0;
    box-shadow: 0 2px 5px rgba(13, 116, 69, 0.1);
}
.token.MISC {
    background: #ffeed3;
    color: #8c4505;
    font-weight: 700;
    border: 1px solid #fcdbaf;
    box-shadow: 0 2px 5px rgba(140, 69, 5, 0.1);
}

/* =========================================================
   COMPARE MODELS & ABOUT SECTIONS
   ========================================================= */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: #ffffff;
    border: 1.5px solid #ece4f8;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 4px 18px -2px rgba(50, 24, 96, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px -3px rgba(50, 24, 96, 0.1);
}

.kpi-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}

.kpi-icon.gold { background: #fff8e6; color: #b58100; border: 1px solid #fae4a8; }
.kpi-icon.purple { background: #f3edfc; color: #6941b8; border: 1px solid #dfd1f5; }
.kpi-icon.blue { background: #eaf3ff; color: #1e63b8; border: 1px solid #c8dffa; }
.kpi-label { font-size: 0.86rem; color: #5f547c; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: clamp(26px, 2.2vw, 34px); font-weight: 800; color: #24164d; margin: 4px 0; }
.kpi-sub { font-size: 0.86rem; color: #72688f; }

.about-card {
    background: #ffffff;
    border: 1.5px solid #ece4f8;
    border-radius: 20px;
    padding: 26px 28px;
    box-shadow: 0 4px 20px -2px rgba(50, 24, 96, 0.05);
    margin-bottom: 20px;
}

.about-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2rem;
    font-weight: 800;
    color: #24164b;
    margin-bottom: 12px;
}
.about-header svg { color: var(--purple); }

.stat-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.stat-pill {
    background: #f4effd;
    color: #553396;
    font-size: 0.86rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid #ded3f4;
}

@media(max-width: 1100px) {
    .models { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .entity-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .kpi-grid { grid-template-columns: 1fr; }
}

@media(max-width: 760px) {
    .block-container { padding: 1.4rem 1rem 3rem; }
    .hero { grid-template-columns: 1fr; gap: 0; }
    .hero-image { max-width: 270px; justify-self: center; }
    .models { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .st-key-try_card, .st-key-compare_card { min-height: auto; padding: 20px 18px !important; }
    .st-key-analysis_panel, .st-key-result_panel { min-height: auto !important; padding: 18px !important; }
    .tokens { font-size: 15px; padding: 14px; }
}

@media(max-width: 430px) {
    .entity-row { gap: 9px; }
    .entity { padding: 11px 9px; gap: 7px; }
    .entity svg { width: 25px; height: 25px; }
    .entity .small { font-size: 10px; }
    .models { gap: 8px; }
    .model { padding: 16px 6px; }
    .hero .intro { font-size: 14px; }
}
</style>
''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading saved model...')
def cached_model(name, directory):
    return load_model(name, directory)

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if 'sentence' not in st.session_state:
    st.session_state.sentence = 'Bill Gates founded Microsoft in the United States.'

def go(page):
    st.session_state.page = page
    st.rerun()

NAV_ITEMS = [
    ('Home', ':material/home:'),
    ('Try NER', ':material/play_circle:'),
    ('Compare Models', ':material/bar_chart:'),
    ('About', ':material/info:'),
]

with st.sidebar:
    st.markdown(f'<div class="brand"><div class="brand-icon">{icon("brain", 24)}</div><span>NER Demo</span></div>', unsafe_allow_html=True)
    for page_name, nav_icon in NAV_ITEMS:
        st.button(
            page_name,
            icon=nav_icon,
            key=f'nav_{page_name}',
            type='primary' if st.session_state.page == page_name else 'secondary',
            on_click=go,
            args=(page_name,),
            use_container_width=True
        )

page = st.session_state.page
if page == 'Home':
    hero_svg = (ROOT / 'assets' / 'ner_illustration.svg').read_text(encoding='utf-8')
    import base64
    hero_uri = 'data:image/svg+xml;base64,' + base64.b64encode(hero_svg.encode()).decode()
    st.markdown(f'''<div class="hero"><div class="hero-text"><h1>Named Entity Recognition (NER)</h1><div class="subtitle">on CoNLL-2003 Dataset</div><p class="intro">Identify and classify named entities in English text (PER, ORG, LOC, MISC) using multiple NLP models and compare their performance.</p></div><img class="hero-image" src="{hero_uri}" alt="Vector illustration of entity tags and a magnifying glass"></div>''', unsafe_allow_html=True)
    
    labels = [
        ('per', 'person', 'PER', 'Person'),
        ('org', 'org', 'ORG', 'Organization'),
        ('loc', 'pin', 'LOC', 'Location'),
        ('misc', 'tag', 'MISC', 'Miscellaneous')
    ]
    st.markdown('<div class="entity-row">' + ''.join(f'<div class="entity {cls}">{icon(sym, 32)}<div><div class="label">{label}</div><div class="small">{description}</div></div></div>' for cls, sym, label, description in labels) + '</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        with st.container(key='try_card'):
            st.markdown(f'<div><div class="action-title">{icon("sparkles", 32)} Try NER</div><div class="action-description">Enter any sentence and inspect predicted named entities in real-time.</div></div>', unsafe_allow_html=True)
            if st.button('Start Now', key='go_ner', icon=':material/arrow_forward:'):
                go('Try NER')
    with col2:
        with st.container(key='compare_card'):
            st.markdown(f'<div><div class="action-title">{icon("chart", 32)} Compare Models</div><div class="action-description">Upload evaluation CSV and view precision, recall, and span F1 scores.</div></div>', unsafe_allow_html=True)
            if st.button('View Results', key='go_compare', icon=':material/arrow_forward:'):
                go('Compare Models')
                
    st.markdown(f'<div class="section-title">{icon("layers", 30)} Trained Models</div>', unsafe_allow_html=True)
    models = [
        ('file', 'N-gram', '+ Logistic Regression'),
        ('nodes', 'Word2Vec', '+ Classifier'),
        ('globe', 'GloVe', '+ Classifier'),
        ('chip', 'LSTM', '+ Combined Features'),
        ('layers', 'LSTM', '+ Embedding Layer'),
    ]
    st.markdown('<div class="models">' + ''.join(f'<div class="model"><div class="model-icon">{icon(sym, 32)}</div><strong>{title}</strong><span>{detail}</span></div>' for sym, title, detail in models) + '</div>', unsafe_allow_html=True)

elif page == 'Try NER':
    st.markdown(f'<div class="page-header"><h1>{icon("sparkles", 36)} Try NER</h1><p>Explore entity recognition using your five saved models.</p></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="legend">'
        f'<span class="pill PER">{icon("person", 16)} PER · Person</span>'
        f'<span class="pill ORG">{icon("org", 16)} ORG · Organization</span>'
        f'<span class="pill LOC">{icon("pin", 16)} LOC · Location</span>'
        f'<span class="pill MISC">{icon("tag", 16)} MISC · Miscellaneous</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    left, right = st.columns([1, 1], gap='large')
    with left:
        with st.container(key='analysis_panel'):
            st.markdown(f'<div class="panel-title">{icon("text", 22)} Text analysis</div>', unsafe_allow_html=True)
            
            # Interactive Quick Sample Sentence Chips
            st.markdown('<div class="sample-label">⚡ Quick sample sentences:</div>', unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3, gap="small")
            with sc1:
                if st.button("Tech Founders", key="sample_tech", use_container_width=True):
                    st.session_state.sentence = "Bill Gates founded Microsoft in the United States."
                    st.session_state.pop('last_prediction', None)
                    st.rerun()
            with sc2:
                if st.button("Apple in CA", key="sample_apple", use_container_width=True):
                    st.session_state.sentence = "Steve Jobs and Steve Wozniak built Apple in California."
                    st.session_state.pop('last_prediction', None)
                    st.rerun()
            with sc3:
                if st.button("UN in Geneva", key="sample_un", use_container_width=True):
                    st.session_state.sentence = "The United Nations convened a global summit in Geneva."
                    st.session_state.pop('last_prediction', None)
                    st.rerun()
            
            sentence = st.text_area(
                'Your sentence',
                key='sentence',
                height=130,
                placeholder="Type or paste a sentence to detect named entities..."
            )
            chosen = st.selectbox('Choose a model', MODEL_NAMES)
            analyze = st.button('Analyze Text', type='primary', icon=':material/neurology:', use_container_width=True)
            st.caption('Inference loads saved artifacts from the models/ directory. Tokenization matches the model preprocessing.')
            
    with right:
        with st.container(key='result_panel'):
            st.markdown(f'<div class="panel-title">{icon("check_circle", 22)} Prediction results</div>', unsafe_allow_html=True)
            if analyze:
                if not sentence.strip():
                    st.warning('Please enter a sentence to analyze.')
                    st.session_state.pop('last_prediction', None)
                else:
                    try:
                        loaded = cached_model(chosen, str(ROOT / 'models'))
                        words, tags = predict(sentence, chosen, ROOT / 'models', loaded)
                        spans = extract_entities(words, tags)
                        st.session_state.last_prediction = (words, tags, spans, chosen)
                    except Exception as exc:
                        st.session_state.pop('last_prediction', None)
                        st.error(f'Unable to run model: {exc}')
                        
            if 'last_prediction' in st.session_state:
                words, tags, spans, previous_model = st.session_state.last_prediction
                label_by_token = {i: span['Type'] for span in spans for i in range(span['Start'], span['End'])}
                
                # Result Status Bar
                st.markdown(
                    f'<div class="result-status-bar">'
                    f'<div class="count-badge">{icon("sparkles", 18)} {len(spans)} Entities Detected</div>'
                    f'<div class="model-badge">Model: <strong>{previous_model}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Highlighted Tokens Flow
                markup = ' '.join(f'<span class="token {label_by_token.get(i, "")}">{html.escape(w)}</span>' for i, w in enumerate(words))
                st.markdown(f'<div class="tokens">{markup}</div>', unsafe_allow_html=True)
                
                if spans:
                    st.dataframe(pd.DataFrame([{'Entity': x['Entity'], 'Type': x['Type']} for x in spans]), hide_index=True, use_container_width=True)
                else:
                    st.info('No named entities detected by this model.')
                    
                with st.expander('Show token-level BIO labels'):
                    st.dataframe(pd.DataFrame({'Token': words, 'Predicted label': tags}), hide_index=True, use_container_width=True)
            else:
                # Polished Empty State Illustration
                st.markdown(
                    f'<div class="empty-state">'
                    f'<div class="empty-icon">{icon("sparkles", 32)}</div>'
                    f'<h3>Ready for Analysis</h3>'
                    f'<p>Enter text on the left or click a quick sample, then press <strong>Analyze Text</strong> to view named entities in real-time.</p>'
                    f'<div class="empty-chips">'
                    f'<span class="chip-item PER">PER Person</span>'
                    f'<span class="chip-item ORG">ORG Organization</span>'
                    f'<span class="chip-item LOC">LOC Location</span>'
                    f'<span class="chip-item MISC">MISC Miscellaneous</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

elif page == 'Compare Models':
    st.markdown(f'<div class="page-header"><h1>{icon("chart", 36)} Compare Models</h1><p>Explore validation metrics and benchmark performance across all 5 models on CoNLL-2003.</p></div>', unsafe_allow_html=True)
    
    # Benchmark KPIs
    st.markdown(f'''
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon gold">{icon("sparkles", 24)}</div>
            <div class="kpi-label">Top F1-Score</div>
            <div class="kpi-value">81.96%</div>
            <div class="kpi-sub">LSTM + Combined Features</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon purple">{icon("check_circle", 24)}</div>
            <div class="kpi-label">Best Precision</div>
            <div class="kpi-value">85.90%</div>
            <div class="kpi-sub">Highest Entity Confidence</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon blue">{icon("chart", 24)}</div>
            <div class="kpi-label">Top Baseline</div>
            <div class="kpi-value">81.07%</div>
            <div class="kpi-sub">N-gram + Logistic Regression</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load default benchmark from notebook/ner_validation_results.csv if present
    default_csv = ROOT / 'notebook' / 'ner_validation_results.csv'
    current_df = None
    if default_csv.exists():
        try:
            current_df = pd.read_csv(default_csv)
            current_df.columns = [c.lower() for c in current_df.columns]
        except Exception:
            current_df = None
            
    csv = st.file_uploader('Upload custom results CSV (optional)', type='csv')
    if csv:
        try:
            user_df = pd.read_csv(csv)
            expected = ['model', 'precision', 'recall', 'f1']
            if not all(k in [c.lower() for c in user_df.columns] for k in expected):
                st.error('Expected columns: model, precision, recall, f1')
            else:
                user_df.columns = [c.lower() for c in user_df.columns]
                user_df[expected[1:]] = user_df[expected[1:]].apply(pd.to_numeric, errors='coerce')
                current_df = user_df
                st.success('Custom evaluation metrics loaded successfully!')
        except Exception as exc:
            st.error(f'Unable to read uploaded CSV: {exc}')
            
    if current_df is not None:
        plot_df = current_df.copy()
        plot_df[['precision', 'recall', 'f1']] = plot_df[['precision', 'recall', 'f1']].apply(pd.to_numeric, errors='coerce')
        plot_df = plot_df.sort_values(by='f1', ascending=False)
        
        st.markdown('#### Benchmark Comparison Chart')
        st.bar_chart(plot_df.set_index('model')[['precision', 'recall', 'f1']])
        
        st.markdown('#### Model Leaderboard')
        display_df = plot_df.copy()
        display_df['precision'] = display_df['precision'].apply(lambda x: f'{x*100:.2f}%')
        display_df['recall'] = display_df['recall'].apply(lambda x: f'{x*100:.2f}%')
        display_df['f1'] = display_df['f1'].apply(lambda x: f'{x*100:.2f}%')
        display_df.columns = ['Model Architecture', 'Precision', 'Recall', 'Span F1-Score']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.markdown(f'<div class="page-header"><h1>{icon("info", 36)} About this project</h1><p>CoNLL-2003 Named Entity Recognition experimental comparison.</p></div>', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="about-card">
        <div class="about-header">{icon("file", 24)} CoNLL-2003 Shared Task Benchmark</div>
        <p style="color: #4b3e6e; line-height: 1.6; margin: 0 0 10px;">
            CoNLL-2003 is the canonical benchmark dataset for English Named Entity Recognition. It evaluates models on extracting and classifying real-world entities into four fundamental classes: <strong>PER</strong> (Person), <strong>ORG</strong> (Organization), <strong>LOC</strong> (Location), and <strong>MISC</strong> (Miscellaneous).
        </p>
        <div class="stat-pills">
            <span class="stat-pill">14,987 Training Sentences</span>
            <span class="stat-pill">3,466 Validation Sentences</span>
            <span class="stat-pill">3,684 Test Sentences</span>
            <span class="stat-pill">Span-level F1 Evaluation</span>
        </div>
    </div>
    
    <div class="about-card">
        <div class="about-header">{icon("layers", 24)} Compared Architectures</div>
        <p style="color: #4b3e6e; line-height: 1.6; margin: 0 0 10px;">
            This project systematically investigates the evolution of representation learning for sequence tagging:
        </p>
        <ul style="color: #4b3e6e; line-height: 1.8; margin-bottom: 0; padding-left: 20px;">
            <li><strong>N-gram + Logistic Regression:</strong> High-dimensional sparse contextual character and word n-gram features.</li>
            <li><strong>Word2Vec + Classifier:</strong> Continuous bag-of-words & Skip-gram distributed representations.</li>
            <li><strong>GloVe + Classifier:</strong> Global log-bilinear matrix factorization word representations.</li>
            <li><strong>Bi-LSTM + Combined Features:</strong> Bidirectional recurrence over concatenated dense embeddings and n-gram vectors (Best F1: 81.96%).</li>
            <li><strong>Bi-LSTM + Learnable Embedding:</strong> End-to-end differentiable token embedding layer trained jointly.</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
