# ruff: noqa: E501  (HTML/CSS string literals; black does not reflow strings)
"""AquaHealth AI design system — dark cinematic aquatic UI, no external assets.

Everything is inline CSS/SVG/HTML rendered through st.markdown: background layers (water gradient,
particles), midground (technical grid, scan rings, glow), foreground (fish, glass panels, controls).
Product-style navigation, technical frames with corner brackets, luminous buttons. Motion is disabled
under `prefers-reduced-motion`. The 3D-style centerpiece is decorative; the app works without it.
"""

from __future__ import annotations

import html

PALETTE = {
    "bg": "#05090f",
    "text": "#e8f3ff",
    "muted": "#8aa3c0",
    "cyan": "#22d3ee",
    "blue": "#3b82f6",
    "violet": "#8b5cf6",
    "healthy": "#34d399",
    "moderate": "#f59e0b",
    "high": "#fb7185",
    "low": "#60a5fa",
}

CSS = """
<style>
:root {
  --bg:#05090f; --text:#e8f3ff; --muted:#8aa3c0; --cyan:#22d3ee; --blue:#3b82f6; --violet:#8b5cf6;
  --healthy:#34d399; --moderate:#f59e0b; --high:#fb7185; --low:#60a5fa;
  --line:rgba(120,200,255,.18); --line-strong:rgba(34,211,238,.55);
  --glass:rgba(10,18,32,.66); --glass2:rgba(14,26,46,.5);
}
/* ---------- background (layer 1) ---------- */
.stApp {
  background:
    radial-gradient(1100px 620px at 15% -10%, rgba(34,211,238,.14) 0%, transparent 60%),
    radial-gradient(900px 560px at 95% 5%, rgba(139,92,246,.16) 0%, transparent 55%),
    radial-gradient(1000px 700px at 50% 115%, rgba(59,130,246,.16) 0%, transparent 60%),
    linear-gradient(180deg,#05090f 0%,#081120 55%,#05090f 100%);
  color:var(--text);
}
.stApp::before { content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.55;
  background-image: radial-gradient(rgba(165,243,252,.35) 1px, transparent 1.6px),
                    radial-gradient(rgba(139,92,246,.25) 1px, transparent 1.6px);
  background-size: 140px 140px, 220px 220px; background-position: 0 0, 60px 80px; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
.block-container { max-width:1240px; padding-top:1.2rem; padding-bottom:3rem; }
h1,h2,h3,h4 { color:var(--text); letter-spacing:-.01em; }
p,li { color:var(--text); }
.aq-muted { color:var(--muted); }
.aq-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.16em; color:var(--muted); }
.aq-label.accent { color:var(--cyan); }
.aq-num { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--cyan); font-size:.78rem; letter-spacing:.14em; }
.aq-h1 { font-size:clamp(2.4rem,5.2vw,4.6rem); font-weight:800; line-height:1.0; margin:.3rem 0 .8rem 0;
  background:linear-gradient(90deg,#ffffff 0%,#bff6ff 55%,#c4b5fd 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.aq-h2 { font-size:clamp(1.6rem,2.6vw,2.3rem); font-weight:800; margin:0 0 .4rem 0; }
.aq-sub { color:var(--muted); font-size:1.08rem; max-width:58ch; line-height:1.55; }
.aq-section { margin-top:34px; }
/* ---------- frames + glass (layer 3) ---------- */
.aq-frame { position:relative; background:var(--glass); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border:1px solid var(--line); border-radius:12px; padding:22px 24px; box-shadow:0 30px 70px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.04); }
.aq-frame::before, .aq-frame::after { content:""; position:absolute; width:18px; height:18px; border-color:var(--line-strong); border-style:solid; pointer-events:none; }
.aq-frame::before { left:-1px; top:-1px; border-width:2px 0 0 2px; border-radius:12px 0 0 0; }
.aq-frame::after { right:-1px; bottom:-1px; border-width:0 2px 2px 0; border-radius:0 0 12px 0; }
.aq-frame.tight { padding:14px 16px; }
.aq-frame.center { text-align:center; }
.aq-frame.glow-cyan { box-shadow:0 0 0 1px rgba(34,211,238,.28), 0 30px 70px rgba(34,211,238,.10); }
.aq-frame.glow-healthy { box-shadow:0 0 0 1px rgba(52,211,153,.4), 0 30px 70px rgba(52,211,153,.12); }
.aq-frame.glow-high { box-shadow:0 0 0 1px rgba(251,113,133,.4), 0 30px 70px rgba(251,113,133,.12); }
.aq-frame.glow-moderate { box-shadow:0 0 0 1px rgba(245,158,11,.4), 0 30px 70px rgba(245,158,11,.12); }
.aq-frame.glow-low { box-shadow:0 0 0 1px rgba(96,165,250,.4), 0 30px 70px rgba(96,165,250,.12); }
.aq-badge { display:inline-block; padding:8px 18px; border-radius:6px; font-weight:800; letter-spacing:.12em; font-size:.82rem; color:#04101c; text-transform:uppercase; }
.aq-badge.LOW { background:var(--low); } .aq-badge.MODERATE { background:var(--moderate); }
.aq-badge.HIGH { background:var(--high); } .aq-badge.HEALTHY { background:var(--healthy); }
.aq-chip { display:inline-block; padding:5px 12px; border-radius:6px; border:1px solid var(--line); background:rgba(255,255,255,.03); color:var(--muted); font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; margin:0 8px 8px 0; }
.aq-note { background:rgba(34,211,238,.07); border:1px solid rgba(34,211,238,.3); border-radius:10px; padding:14px 18px; color:var(--text); }
.aq-warn { background:rgba(245,158,11,.10); border:1px solid rgba(245,158,11,.45); border-radius:10px; padding:14px 18px; color:#ffe1b0; }
.aq-err { background:rgba(251,113,133,.10); border:1px solid rgba(251,113,133,.45); border-radius:10px; padding:14px 18px; color:#ffd0d8; }
.aq-disclaimer { font-size:.84rem; color:var(--muted); }
.aq-big { font-size:clamp(2rem,3.6vw,3.2rem); font-weight:800; line-height:1.05; margin:6px 0 10px 0; }
.aq-pct { font-size:clamp(3rem,6vw,5.2rem); font-weight:800; line-height:1; letter-spacing:-.02em; }
/* ---------- nav ---------- */
.aq-brand { display:flex; align-items:center; gap:12px; padding:6px 0; }
.aq-brand .logo { width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,var(--cyan),var(--violet)); box-shadow:0 0 22px rgba(34,211,238,.5); }
.aq-brand b { letter-spacing:.2em; font-size:.9rem; text-transform:uppercase; }
.st-key-aq-nav .stButton > button { width:100%; min-height:44px; border-radius:8px; border:1px solid transparent; background:transparent; color:var(--muted);
  font-weight:700; letter-spacing:.14em; font-size:.74rem; text-transform:uppercase; }
.st-key-aq-nav .stButton > button:hover { color:var(--text); border-color:var(--line); background:rgba(255,255,255,.03); box-shadow:none; }
.st-key-aq-nav .stButton > button[kind="primary"] { background:rgba(34,211,238,.10); color:var(--cyan); border:1px solid var(--line-strong); box-shadow:inset 0 -2px 0 var(--cyan); }
.aq-navline { height:1px; background:linear-gradient(90deg,transparent,var(--line-strong),transparent); margin:6px 0 22px 0; }
/* ---------- buttons ---------- */
.stButton > button { min-height:50px; border-radius:8px; padding:.6rem 1.6rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; font-size:.8rem;
  border:1px solid var(--line); background:rgba(14,26,46,.7); color:var(--text); transition:all .15s ease; }
.stButton > button:hover { border-color:var(--cyan); box-shadow:0 0 22px rgba(34,211,238,.3); transform:translateY(-1px); color:var(--text); }
.stButton > button:focus-visible { outline:2px solid var(--cyan); outline-offset:2px; }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,var(--blue),var(--cyan)); border:none; color:#04101c; box-shadow:0 12px 34px rgba(34,211,238,.32); }
.stButton > button[kind="primary"]:hover { box-shadow:0 14px 40px rgba(34,211,238,.5); color:#04101c; }
.stButton > button[kind="tertiary"] { background:transparent; border:1px solid transparent; color:var(--muted); min-height:40px; }
.stButton > button[kind="tertiary"]:hover { color:var(--cyan); box-shadow:none; transform:none; }
/* ---------- uploader ---------- */
[data-testid="stFileUploader"] { background:rgba(10,18,32,.55); border:1px dashed rgba(34,211,238,.5); border-radius:12px; padding:26px 18px; }
[data-testid="stFileUploader"] section { background:transparent; }
[data-testid="stFileUploader"] section > button { border-radius:8px; border:1px solid var(--line-strong); background:rgba(34,211,238,.1); color:var(--cyan); font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
[data-testid="stImage"] img { border-radius:10px; }
[data-testid="stExpander"] { background:var(--glass2); border:1px solid var(--line); border-radius:10px; }
[data-testid="stTable"] { color:var(--text); }
/* ---------- probability bars ---------- */
.aq-row { display:grid; grid-template-columns:minmax(150px,1.3fr) 4fr 74px; gap:14px; align-items:center; margin:10px 0; font-size:.95rem; }
.aq-row.top { color:#bff6ff; font-weight:700; }
.aq-bar { height:12px; border-radius:3px; background:rgba(255,255,255,.07); overflow:hidden; position:relative; }
.aq-bar > span { display:block; height:100%; background:linear-gradient(90deg,var(--blue),var(--cyan)); }
.aq-bar.top > span { background:linear-gradient(90deg,var(--cyan),#bff6ff); box-shadow:0 0 14px rgba(34,211,238,.7); }
.aq-bar.healthy > span { background:linear-gradient(90deg,#059669,var(--healthy)); box-shadow:0 0 14px rgba(52,211,153,.6); }
/* ---------- feature strip / class cards ---------- */
.aq-grid4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
.aq-grid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.aq-feature { position:relative; background:var(--glass2); border:1px solid var(--line); border-radius:10px; padding:18px 18px 16px 18px; }
.aq-feature::before { content:""; position:absolute; left:0; top:14px; bottom:14px; width:2px; background:linear-gradient(180deg,var(--cyan),var(--violet)); border-radius:2px; }
.aq-feature .aq-num { display:block; margin-bottom:6px; }
.aq-feature b { display:block; font-size:1.02rem; margin-bottom:4px; }
.aq-feature small { color:var(--muted); line-height:1.4; }
/* ---------- timeline ---------- */
.aq-timeline { position:relative; padding-left:34px; }
.aq-timeline::before { content:""; position:absolute; left:11px; top:8px; bottom:8px; width:2px; background:linear-gradient(180deg,var(--cyan),var(--violet)); box-shadow:0 0 12px rgba(34,211,238,.6); }
.aq-tl { position:relative; margin:0 0 16px 0; background:var(--glass2); border:1px solid var(--line); border-radius:10px; padding:14px 18px; }
.aq-tl::before { content:""; position:absolute; left:-30px; top:18px; width:14px; height:14px; border-radius:50%; background:var(--bg); border:2px solid var(--cyan); box-shadow:0 0 14px var(--cyan); }
.aq-tl .aq-num { display:block; margin-bottom:4px; }
.aq-tl b { display:block; font-size:1.05rem; letter-spacing:.04em; }
.aq-tl small { color:var(--muted); }
/* ---------- hero + centerpiece (layers 2/3) ---------- */
.aq-hero { text-align:center; padding:26px 12px 0 12px; }
.aq-scene { position:relative; height:520px; margin:6px auto 0 auto; perspective:1100px; overflow:hidden; border-radius:16px;
  background:radial-gradient(700px 320px at 50% 62%, rgba(34,211,238,.12), transparent 70%); }
.aq-scene.small { height:260px; }
.aq-water { position:absolute; inset:0; background:
  radial-gradient(600px 240px at 50% 100%, rgba(59,130,246,.22), transparent 70%),
  linear-gradient(180deg, rgba(5,9,15,0) 0%, rgba(8,17,32,.6) 100%); }
.aq-grid { position:absolute; left:-20%; right:-20%; bottom:-10%; height:70%; background-image:
  linear-gradient(rgba(34,211,238,.10) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.10) 1px, transparent 1px);
  background-size:44px 44px; transform:perspective(700px) rotateX(64deg); transform-origin:50% 100%;
  -webkit-mask-image:linear-gradient(180deg, transparent 0%, #000 45%, #000 100%); mask-image:linear-gradient(180deg, transparent 0%, #000 45%, #000 100%); }
.aq-orb { position:absolute; left:50%; top:52%; width:520px; height:520px; margin:-260px 0 0 -260px; border-radius:50%;
  background:radial-gradient(circle at 42% 38%, rgba(165,243,252,.28), rgba(34,211,238,.14) 35%, rgba(59,130,246,.06) 55%, transparent 70%); filter:blur(6px); }
.aq-ring { position:absolute; left:50%; top:56%; border-radius:50%; border:1px solid rgba(34,211,238,.45);
  box-shadow:0 0 30px rgba(34,211,238,.25), inset 0 0 30px rgba(34,211,238,.12); transform:rotateX(66deg); animation:aq-spin 16s linear infinite; }
.aq-ring::after { content:""; position:absolute; inset:-1px; border-radius:50%; border-top:3px solid var(--cyan); filter:drop-shadow(0 0 10px var(--cyan)); }
.aq-ring.r1 { width:420px; height:420px; margin:-210px 0 0 -210px; }
.aq-ring.r2 { width:600px; height:600px; margin:-300px 0 0 -300px; animation-duration:26s; animation-direction:reverse; border-color:rgba(139,92,246,.4); }
.aq-ring.r2::after { border-top-color:var(--violet); filter:drop-shadow(0 0 10px var(--violet)); }
.aq-ring.r3 { width:250px; height:250px; margin:-125px 0 0 -125px; animation-duration:9s; border-color:rgba(165,243,252,.5); }
.aq-fish { position:absolute; left:50%; top:46%; width:min(560px,80%); height:auto; transform:translate(-50%,-50%);
  filter:drop-shadow(0 30px 40px rgba(0,0,0,.6)) drop-shadow(0 0 30px rgba(34,211,238,.35)); animation:aq-swim 8s ease-in-out infinite; }
.aq-scene.small .aq-fish { width:min(300px,70%); }
.aq-scene.small .aq-ring.r1 { width:260px; height:260px; margin:-130px 0 0 -130px; }
.aq-scene.small .aq-ring.r2 { width:360px; height:360px; margin:-180px 0 0 -180px; }
.aq-scene.small .aq-ring.r3 { display:none; }
.aq-scene.small .aq-orb { width:320px; height:320px; margin:-160px 0 0 -160px; }
.aq-scanline { position:absolute; left:12%; right:12%; height:2px; top:20%; background:linear-gradient(90deg,transparent,var(--cyan),transparent); box-shadow:0 0 16px var(--cyan); animation:aq-scan 3.2s ease-in-out infinite; }
.aq-bubble { position:absolute; bottom:-10px; border-radius:50%; background:rgba(165,243,252,.45); box-shadow:0 0 12px rgba(165,243,252,.6); animation:aq-rise 9s linear infinite; }
.aq-bubble.b1 { left:24%; width:8px; height:8px; } .aq-bubble.b2 { left:36%; width:5px; height:5px; animation-delay:3s; animation-duration:11s; }
.aq-bubble.b3 { left:64%; width:11px; height:11px; animation-delay:5s; animation-duration:13s; } .aq-bubble.b4 { left:78%; width:6px; height:6px; animation-delay:1.5s; animation-duration:10s; }
.aq-bubble.b5 { left:50%; width:4px; height:4px; animation-delay:7s; animation-duration:12s; }
.aq-node { position:absolute; padding:8px 12px; border-radius:6px; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--text);
  background:rgba(10,18,32,.85); border:1px solid var(--line-strong); backdrop-filter:blur(8px); animation:aq-float 6s ease-in-out infinite; }
.aq-node .aq-num { display:block; font-size:.62rem; margin-bottom:2px; }
.aq-node.n1 { left:6%; top:16%; } .aq-node.n2 { right:6%; top:22%; animation-delay:2s; } .aq-node.n3 { left:10%; bottom:16%; animation-delay:4s; } .aq-node.n4 { right:8%; bottom:20%; animation-delay:1s; }
.aq-corner { position:absolute; width:26px; height:26px; border-color:var(--line-strong); border-style:solid; }
.aq-corner.tl { left:10px; top:10px; border-width:2px 0 0 2px; } .aq-corner.tr { right:10px; top:10px; border-width:2px 2px 0 0; }
.aq-corner.bl { left:10px; bottom:10px; border-width:0 0 2px 2px; } .aq-corner.br { right:10px; bottom:10px; border-width:0 2px 2px 0; }
.aq-analyzing { position:relative; height:320px; border-radius:12px; overflow:hidden; background:var(--glass); border:1px solid var(--line-strong); }
.aq-analyzing .aq-fish { display:none; }
.aq-analyzing .aq-ring.r1 { width:220px; height:220px; margin:-110px 0 0 -110px; animation-duration:4s; }
.aq-analyzing .aq-ring.r2 { width:320px; height:320px; margin:-160px 0 0 -160px; animation-duration:7s; }
.aq-analyzing .aq-ring.r3 { width:120px; height:120px; margin:-60px 0 0 -60px; animation-duration:2.5s; }
.aq-analyzing .aq-orb { width:360px; height:360px; margin:-180px 0 0 -180px; }
.aq-analyzing .caption { position:absolute; left:0; right:0; top:22px; text-align:center; }
.aq-stages { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.aq-stages span { padding:7px 12px; border-radius:6px; border:1px solid var(--line); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.aq-gauge { width:170px; height:170px; }
.aq-footer { margin-top:44px; padding:18px 4px; border-top:1px solid var(--line); color:var(--muted); font-size:.85rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
@keyframes aq-spin { from { transform:rotateX(66deg) rotateZ(0deg);} to { transform:rotateX(66deg) rotateZ(360deg);} }
@keyframes aq-swim { 0%,100% { transform:translate(-50%,-50%) rotate(-2deg);} 50% { transform:translate(calc(-50% - 18px),calc(-50% + 12px)) rotate(3deg);} }
@keyframes aq-rise { 0% { transform:translateY(0); opacity:0;} 12% { opacity:.9;} 100% { transform:translateY(-520px); opacity:0;} }
@keyframes aq-float { 0%,100% { transform:translateY(0);} 50% { transform:translateY(-10px);} }
@keyframes aq-scan { 0%,100% { top:14%;} 50% { top:86%;} }
@media (prefers-reduced-motion: reduce) { .aq-ring,.aq-fish,.aq-bubble,.aq-node,.aq-scanline { animation:none; } }
@media (max-width: 1000px) { .aq-grid4 { grid-template-columns:repeat(2,minmax(0,1fr)); } .aq-scene { height:420px; } }
@media (max-width: 680px) { .aq-grid4,.aq-grid3 { grid-template-columns:1fr; } .aq-scene { height:300px; } .aq-node { display:none; } .aq-ring.r2 { display:none; } .aq-pct { font-size:2.6rem; }
  .st-key-aq-nav .stButton > button { min-height:34px; font-size:.68rem; } .st-key-aq-nav [data-testid="stVerticalBlock"], .st-key-aq-nav [data-testid="stHorizontalBlock"] { gap:4px !important; } .aq-h1 { font-size:2.2rem; } }
</style>
"""

FISH_SVG = """
<svg class="aq-fish" viewBox="0 0 560 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="stylised fish under AI scan">
  <defs>
    <linearGradient id="aqBody" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#d9fbff"/><stop offset=".45" stop-color="#67e8f9"/><stop offset=".8" stop-color="#3b82f6"/><stop offset="1" stop-color="#1e3a8a"/>
    </linearGradient>
    <linearGradient id="aqBelly" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="rgba(255,255,255,0)"/><stop offset="1" stop-color="rgba(191,246,255,.55)"/>
    </linearGradient>
    <linearGradient id="aqFin" x1="0" x2="1"><stop offset="0" stop-color="#a5f3fc"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient>
    <radialGradient id="aqEye" cx=".4" cy=".4" r=".6"><stop offset="0" stop-color="#ffffff"/><stop offset=".5" stop-color="#0ea5e9"/><stop offset="1" stop-color="#04101c"/></radialGradient>
    <pattern id="aqScales" width="26" height="26" patternUnits="userSpaceOnUse" patternTransform="rotate(10)">
      <path d="M0 13 Q13 -4 26 13" stroke="rgba(4,16,28,.28)" stroke-width="2" fill="none"/>
    </pattern>
  </defs>
  <ellipse cx="290" cy="250" rx="220" ry="18" fill="rgba(34,211,238,.18)"/>
  <path d="M410 150 L 545 62 L 520 150 L 545 238 Z" fill="url(#aqFin)" opacity=".92"/>
  <path d="M205 82 L 285 18 L 322 86 Z" fill="url(#aqFin)" opacity=".95"/>
  <path d="M225 220 L 280 286 L 330 214 Z" fill="url(#aqFin)" opacity=".95"/>
  <path d="M60 150 C 120 40, 330 30, 428 150 C 330 270, 120 262, 60 150 Z" fill="url(#aqBody)"/>
  <path d="M60 150 C 120 40, 330 30, 428 150 C 330 270, 120 262, 60 150 Z" fill="url(#aqScales)"/>
  <path d="M90 170 C 160 250, 300 250, 400 175 C 300 236, 170 236, 90 170 Z" fill="url(#aqBelly)"/>
  <path d="M175 96 Q 215 150 175 204" stroke="rgba(4,16,28,.32)" stroke-width="4" fill="none"/>
  <path d="M230 80 Q 275 150 230 220" stroke="rgba(4,16,28,.26)" stroke-width="4" fill="none"/>
  <path d="M290 76 Q 340 150 290 224" stroke="rgba(4,16,28,.2)" stroke-width="4" fill="none"/>
  <path d="M235 150 L 300 122 L 300 178 Z" fill="rgba(191,246,255,.35)"/>
  <circle cx="120" cy="128" r="17" fill="url(#aqEye)"/><circle cx="126" cy="122" r="5" fill="#ffffff"/>
  <path d="M62 150 Q 78 166 96 160" stroke="rgba(4,16,28,.35)" stroke-width="3" fill="none"/>
  <path d="M60 150 C 120 40, 330 30, 428 150" stroke="rgba(255,255,255,.35)" stroke-width="2" fill="none"/>
</svg>
"""


def scene_html(small: bool = False, nodes: bool = True) -> str:
    """The 3D-style aquatic centerpiece: water, grid plane, orb, rings, fish, bubbles, nodes."""
    node_html = (
        '<div class="aq-node n1"><span class="aq-num">01</span>AI screening</div>'
        '<div class="aq-node n2"><span class="aq-num">02</span>8-class model</div>'
        '<div class="aq-node n3"><span class="aq-num">03</span>Visual explanation</div>'
        '<div class="aq-node n4"><span class="aq-num">04</span>Risk + recommendation</div>'
        if nodes
        else ""
    )
    return (
        f'<div class="aq-scene{" small" if small else ""}" aria-hidden="true"><div class="aq-water"></div>'
        '<div class="aq-grid"></div><div class="aq-orb"></div>'
        '<div class="aq-ring r2"></div><div class="aq-ring r1"></div><div class="aq-ring r3"></div>'
        f"{FISH_SVG}"
        '<span class="aq-bubble b1"></span><span class="aq-bubble b2"></span><span class="aq-bubble b3"></span>'
        '<span class="aq-bubble b4"></span><span class="aq-bubble b5"></span>'
        f"{node_html}"
        '<span class="aq-corner tl"></span><span class="aq-corner tr"></span><span class="aq-corner bl"></span><span class="aq-corner br"></span>'
        "</div>"
    )


def hero_html(kicker: str, title_html: str, sub: str) -> str:
    """Hero copy block (centered). `title_html` may contain <br>; other text is escaped."""
    return (
        f'<div class="aq-hero"><div class="aq-label accent">{html.escape(kicker)}</div>'
        f'<div class="aq-h1">{title_html}</div><p class="aq-sub" style="margin:0 auto;">{html.escape(sub)}</p></div>'
    )


def page_header(number: str, kicker: str, title: str, sub: str) -> str:
    return (
        f'<div style="margin:6px 0 18px 0;"><span class="aq-num">{html.escape(number)}</span>'
        f'<div class="aq-label" style="margin-top:6px">{html.escape(kicker)}</div>'
        f'<div class="aq-h2">{html.escape(title)}</div><p class="aq-sub">{html.escape(sub)}</p></div>'
    )


def analyzing_html() -> str:
    """Scanning visual + the real pipeline stages. No fake progress percentages."""
    stages = [
        "Image received",
        "Quality check",
        "CLAHE · resize · crop",
        "EfficientNet-B0 inference",
        "Confidence & risk",
        "Grad-CAM explanation",
        "Recommendation",
    ]
    chips = "".join(f"<span>{s}</span>" for s in stages)
    return (
        '<div class="aq-analyzing" role="status" aria-live="polite"><div class="aq-water"></div><div class="aq-grid"></div>'
        '<div class="aq-orb"></div><div class="aq-ring r2"></div><div class="aq-ring r1"></div><div class="aq-ring r3"></div>'
        '<div class="aq-scanline"></div><div class="caption"><div class="aq-label accent">AI visual analysis</div>'
        '<div style="font-weight:800;font-size:1.3rem;margin-top:4px;">Scanning image…</div></div>'
        '<span class="aq-corner tl"></span><span class="aq-corner tr"></span><span class="aq-corner bl"></span><span class="aq-corner br"></span></div>'
        f'<div class="aq-stages">{chips}</div>'
    )


def gauge_svg(confidence: float, colour_var: str) -> str:
    """Radial confidence gauge; the arc length is the actual confidence."""
    value = max(0.0, min(1.0, float(confidence)))
    radius, circumference = 62, 2 * 3.141592653589793 * 62
    dash = circumference * value
    label = ">99.9%" if 0.9995 <= value < 1.0 else f"{value:.1%}"
    return (
        f'<svg class="aq-gauge" viewBox="0 0 150 150" role="img" aria-label="confidence {label}">'
        '<circle cx="75" cy="75" r="62" stroke="rgba(255,255,255,.08)" stroke-width="10" fill="none"/>'
        f'<circle cx="75" cy="75" r="{radius}" stroke="var({colour_var})" stroke-width="10" fill="none" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circumference:.1f}" '
        'transform="rotate(-90 75 75)" style="filter: drop-shadow(0 0 10px var('
        + colour_var
        + '))"/>'
        f'<text x="75" y="83" text-anchor="middle" font-size="24" font-weight="800" fill="#e8f3ff">{label}</text></svg>'
    )


def confidence_label(confidence: float) -> str:
    value = max(0.0, min(1.0, float(confidence)))
    return ">99.9%" if 0.9995 <= value < 1.0 else f"{value:.1%}"
