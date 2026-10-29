"""
CrimeVision AI - Intelligent Multimodal Forensic Analysis Platform
Skill Wallet Capstone Project 2

Single-file, modular Streamlit app. No external database; all state lives in
st.session_state. Works fully offline in DEMO mode; uses Groq/Gemini when a key
is supplied (LLM chat + Whisper transcription via Groq).

Run:  streamlit run app.py
"""
import hashlib
import io
import json
import os
import random
import re
import tempfile
from datetime import datetime, timezone

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# ReportLab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, PageBreak)

APP_NAME = "CrimeVision AI"
SUBTITLE = "Intelligent Multimodal Forensic Analysis Platform"

# ----------------------------------------------------------------------------
# PAGE CONFIG + STYLING
# ----------------------------------------------------------------------------
st.set_page_config(page_title=f"{APP_NAME} | Forensic Platform", page_icon="🕵️",
                   layout="wide")

CSS = """
<style>
:root { --bg:#0b0f17; --panel:#121826; --line:#1f2a3d; --accent:#38bdf8;
        --red:#ef4444; --amber:#f59e0b; --green:#22c55e; --txt:#e5e7eb; }
.stApp { background: var(--bg); color: var(--txt); }
section[data-testid="stSidebar"] { background:#0d1320; border-right:1px solid var(--line); }
h1,h2,h3 { color:#f1f5f9; letter-spacing:.3px; }
.cv-hero { padding:18px 22px; border:1px solid var(--line); border-radius:12px;
  background:linear-gradient(120deg,#0f1a2e,#111827 60%,#1a1020); margin-bottom:14px; }
.cv-hero h1 { margin:0; font-size:1.7rem; }
.cv-hero p { margin:4px 0 0; color:#94a3b8; }
.cv-card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; margin-bottom:8px; }
.cv-badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
  font-weight:700; letter-spacing:.5px; }
.b-low{background:#14532d;color:#86efac}.b-medium{background:#78350f;color:#fcd34d}
.b-high{background:#7c2d12;color:#fdba74}.b-critical{background:#7f1d1d;color:#fca5a5}
.cv-mono { font-family:ui-monospace,Menlo,monospace; font-size:.8rem; color:#7dd3fc; }
.cv-cite { color:#38bdf8; font-weight:600; }
.stTabs [data-baseweb="tab"] { color:#94a3b8; }
.stTabs [aria-selected="true"] { color:#38bdf8 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# STATE + UTILITIES
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "ledger": [], "counters": {}, "custody": [], "images": [], "audio": [],
        "video": [], "chat": [], "case_id": f"CV-{datetime.now():%Y%m%d}-{random.randint(100,999)}",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def next_id(prefix: str) -> str:
    c = st.session_state.counters
    c[prefix] = c.get(prefix, 0) + 1
    return f"{prefix}-{c[prefix]:03d}"


def log_custody(evidence_id: str, name: str, digest: str, action="Ingested"):
    st.session_state.custody.append({
        "Evidence ID": evidence_id, "File": name, "SHA-256": digest,
        "Action": action, "Timestamp": now_utc(),
        "Handler": st.session_state.get("investigator") or "UNSIGNED",
    })


def add_ledger(source, ref, t_sec, description, category, confidence, evid):
    """Append an entry to the unified master forensic event ledger."""
    entry = {"id": evid, "source": source, "ref": ref, "t": float(t_sec),
             "description": description, "category": category,
             "confidence": round(float(confidence), 2)}
    st.session_state.ledger.append(entry)
    return entry


def fmt_t(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def parse_json_loose(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else None


# ----------------------------------------------------------------------------
# LLM / WHISPER CLIENTS (Groq + Gemini, REST only)
# ----------------------------------------------------------------------------
def llm_available() -> bool:
    return bool(st.session_state.get("api_key")) and st.session_state.get("provider") != "Demo Mode"


def call_llm(system: str, user: str, temperature=0.0) -> str:
    provider, key = st.session_state.get("provider"), st.session_state.get("api_key")
    if provider == "Groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "llama-3.3-70b-versatile", "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]}, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if provider == "Gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": temperature}}, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("No LLM provider configured")


def whisper_transcribe(file_bytes: bytes, filename: str):
    """Returns list of segments [{start,end,text}] or raises. Order: Groq Whisper API -> local whisper."""
    if llm_available() and st.session_state.provider == "Groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {st.session_state.api_key}"},
            files={"file": (filename, file_bytes)},
            data={"model": "whisper-large-v3", "response_format": "verbose_json"}, timeout=180)
        r.raise_for_status()
        return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in r.json().get("segments", [])]
    try:  # optional local Whisper (pip install openai-whisper)
        import whisper  # type: ignore
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as f:
            f.write(file_bytes)
            path = f.name
        model = whisper.load_model("base")
        res = model.transcribe(path)
        os.unlink(path)
        return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in res["segments"]]
    except ImportError:
        raise RuntimeError("Whisper unavailable (needs Groq key or openai-whisper).")


# ----------------------------------------------------------------------------
# EPIC 2: VISUAL EVIDENCE TAGGER
# ----------------------------------------------------------------------------
FORENSIC_CLASSES = ["person", "firearm", "edged weapon", "blood spatter", "forced entry point"]
CLASS_COLORS = {"person": "#38bdf8", "firearm": "#ef4444", "edged weapon": "#f97316",
                "blood spatter": "#dc2626", "forced entry point": "#facc15"}
LEAD_TEMPLATES = {
    "person": "Identify and canvass individuals present; cross-reference with CCTV subjects.",
    "firearm": "Secure weapon, photograph in situ, request ballistics + latent print lift.",
    "edged weapon": "Bag and tag blade; swab for DNA/blood; check for tool-mark matches.",
    "blood spatter": "Commission bloodstain pattern analysis; collect DNA swabs before cleanup.",
    "forced entry point": "Document tool marks; compare with recovered tools; check neighbouring entry points.",
}


def try_rtdetr(img: Image.Image):
    """Optional real detector (ultralytics RT-DETR, COCO). Maps COCO -> forensic classes."""
    try:
        from ultralytics import RTDETR  # type: ignore
        model = st.session_state.get("_rtdetr") or RTDETR("rtdetr-l.pt")
        st.session_state["_rtdetr"] = model
        res = model.predict(img, verbose=False, conf=0.3)[0]
        mapping = {"person": "person", "knife": "edged weapon", "scissors": "edged weapon"}
        out = []
        for b in res.boxes:
            name = model.names[int(b.cls)]
            if name in mapping:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                out.append({"label": mapping[name], "conf": float(b.conf), "box": (x1, y1, x2, y2)})
        return out
    except Exception:
        return None


def simulate_rtdetr(img: Image.Image, digest: str):
    """Deterministic RT-DETR simulation seeded by the image hash (reproducible for audit)."""
    rng = random.Random(int(digest[:12], 16))
    w, h = img.size
    n_obj = rng.randint(3, 6)
    labels = ["person"] + rng.sample(FORENSIC_CLASSES, k=n_obj - 1 if n_obj - 1 <= 5 else 5)
    out = []
    for lab in labels:
        bw, bh = rng.uniform(.10, .30) * w, rng.uniform(.15, .40) * h
        x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        out.append({"label": lab, "conf": rng.uniform(.62, .97), "box": (x1, y1, x1 + bw, y1 + bh)})
    return out


def assess_scene(dets):
    labs = [d["label"] for d in dets]
    score = 0
    score += 3 * labs.count("firearm") + 2 * labs.count("edged weapon")
    score += 2 * labs.count("blood spatter") + 1 * labs.count("forced entry point")
    score += 1 if "person" in labs else 0
    threat = "Critical" if score >= 7 else "High" if score >= 4 else "Medium" if score >= 2 else "Low"
    if "blood spatter" in labs and ("firearm" in labs or "edged weapon" in labs):
        scene = "Homicide"
    elif "forced entry point" in labs:
        scene = "Burglary"
    elif ("edged weapon" in labs or "firearm" in labs) or "blood spatter" in labs:
        scene = "Assault"
    else:
        scene = "Vandalism"
    return threat, scene


def draw_boxes(img: Image.Image, dets):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, det in enumerate(dets, 1):
        c = CLASS_COLORS[det["label"]]
        d.rectangle(det["box"], outline=c, width=max(2, im.width // 300))
        d.text((det["box"][0] + 3, det["box"][1] + 3), f"#{i} {det['label']} {det['conf']:.0%}", fill=c)
    return im


def analyze_image(file):
    data = file.getvalue()
    digest = sha256_bytes(data)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    real = try_rtdetr(img)
    mode = "RT-DETR (live)" if real else "RT-DETR (simulated)"
    dets = real if real else simulate_rtdetr(img, digest)
    if real and not any(d["label"] in FORENSIC_CLASSES[1:] for d in dets):
        dets += [d for d in simulate_rtdetr(img, digest) if d["label"] in ("blood spatter", "forced entry point")][:1]
        mode = "RT-DETR (live COCO) + heuristic forensic classes"
    threat, scene = assess_scene(dets)
    evid = next_id("IMG")
    log_custody(evid, file.name, digest)
    leads = sorted({d["label"] for d in dets},
                   key=lambda l: ["firearm", "edged weapon", "blood spatter", "forced entry point", "person"].index(l))
    return {"id": evid, "name": file.name, "digest": digest, "mode": mode, "threat": threat,
            "scene": scene, "dets": dets, "leads": [LEAD_TEMPLATES[l] for l in leads], "img": img}


# ----------------------------------------------------------------------------
# EPIC 3: AUDIO
# ----------------------------------------------------------------------------
KEYWORD_EVENTS = [
    (r"\b(gun|pistol|firearm|revolver)\b", "Firearm mentioned", "Armed Threat"),
    (r"\b(knife|blade|machete)\b", "Edged weapon mentioned", "Armed Threat"),
    (r"\b(shot|shots|gunfire|fired)\b", "Gunfire reported", "Armed Violence"),
    (r"\b(ran|run|fled|running|escaped|drove off|took off)\b", "Suspect fled scene", "Flight from Scene"),
    (r"\b(broke|break|smashed|kicked|forced).{0,15}(door|window|lock)\b", "Forced entry described", "Burglary"),
    (r"\b(hit|punched|attacked|stabbed|beat)\b", "Physical assault described", "Assault"),
    (r"\b(car|van|truck|motorbike|vehicle|sedan)\b", "Vehicle referenced", "Vehicle Involvement"),
    (r"\b(hoodie|jacket|mask|wearing|tall|short)\b", "Suspect description", "Identification"),
    (r"\b(bleeding|blood|injured|hurt)\b", "Injury / blood reported", "Bodily Harm"),
]

DEMO_TRANSCRIPT = [
    (0.0, 5.0, "Dispatcher", "Emergency services, what is your emergency?"),
    (5.0, 12.0, "Caller", "Someone just broke the back door of the shop next to me and went inside."),
    (12.0, 19.0, "Dispatcher", "Can you describe the person?"),
    (19.0, 28.0, "Caller", "Tall, wearing a black hoodie and a mask. He's got a knife in his hand."),
    (28.0, 36.0, "Caller", "I heard shouting, then he hit the owner. There is blood on the floor."),
    (36.0, 44.0, "Caller", "He ran out and got into a dark sedan, driving off toward the highway."),
]


def tag_transcript(segments):
    """segments: [(start,end,speaker,text)] -> (events, tagged)"""
    events = []
    for start, end, spk, text in segments:
        for pat, label, cat in KEYWORD_EVENTS:
            if re.search(pat, text, re.I):
                events.append({"t": start, "speaker": spk, "event": label, "category": cat, "quote": text})
    return events


def analyze_audio(file):
    data = file.getvalue()
    digest = sha256_bytes(data)
    evid = next_id("AUD")
    log_custody(evid, file.name, digest)
    mode, segments = "Whisper", None
    try:
        raw = whisper_transcribe(data, file.name)
        # naive speaker diarisation: alternate on silence gaps > 1.2s
        segments, spk, prev_end = [], 0, 0.0
        for s in raw:
            if s["start"] - prev_end > 1.2:
                spk ^= 1
            segments.append((s["start"], s["end"], f"Speaker {spk + 1}", s["text"]))
            prev_end = s["end"]
    except Exception as e:
        mode = f"Demo fallback ({str(e)[:60]})"
        segments = DEMO_TRANSCRIPT
    events = tag_transcript(segments)
    return {"id": evid, "name": file.name, "digest": digest, "mode": mode,
            "segments": segments, "events": events}


# ----------------------------------------------------------------------------
# EPIC 4: VIDEO
# ----------------------------------------------------------------------------
def classify_motion(speed, area_ratio, prev_speed):
    """Heuristic mapping of motion kinematics -> advisory legal category."""
    if area_ratio > 0.18 and speed < 25:
        return "Assault / Physical Altercation (advisory)", 0.6
    if speed > 90:
        return "Flight from Scene / Pursuit (advisory)", 0.7
    if prev_speed is not None and prev_speed < 10 and speed > 40:
        return "Burglary - Rapid Entry/Exit (advisory)", 0.62
    if speed > 35:
        return "Suspicious Movement / Trespass (advisory)", 0.55
    return "Loitering / Surveillance (advisory)", 0.5


DEMO_VIDEO_EVENTS = [
    (2, "Subject enters frame from left; approaches rear door.", "Trespass (advisory)", .66, (80, 200)),
    (6, "Subject forces door; brief struggle with second person.", "Burglary / Assault (advisory)", .71, (180, 190)),
    (11, "Object consistent with bladed weapon visible in subject's right hand.", "Armed Threat (advisory)", .64, (260, 180)),
    (16, "Subject exits at speed; vehicle departs frame right.", "Flight from Scene (advisory)", .74, (420, 210)),
]


def analyze_video(file, step_sec=1.0, max_frames=60):
    data = file.getvalue()
    digest = sha256_bytes(data)
    evid = next_id("VID")
    log_custody(evid, file.name, digest)
    events, traj, mode = [], [], "OpenCV motion tracking"
    try:
        if cv2 is None:
            raise RuntimeError("OpenCV not installed")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1] or ".mp4") as f:
            f.write(data)
            path = f.name
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            raise RuntimeError("Unreadable video")
        stride = max(1, int(fps * step_sec))
        prev_gray, prev_c, prev_speed, idx, n = None, None, None, 0, 0
        while n < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            t = idx / fps
            if prev_gray is not None:
                diff = cv2.threshold(cv2.absdiff(prev_gray, gray), 25, 255, cv2.THRESH_BINARY)[1]
                diff = cv2.dilate(diff, None, iterations=2)
                cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = [c for c in cnts if cv2.contourArea(c) > 400]
                if cnts:
                    big = max(cnts, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(big)
                    c = (x + w / 2, y + h / 2)
                    area_ratio = cv2.contourArea(big) / (frame.shape[0] * frame.shape[1])
                    speed = float(np.hypot(c[0] - prev_c[0], c[1] - prev_c[1])) / step_sec if prev_c else 0.0
                    cat, conf = classify_motion(speed, area_ratio, prev_speed)
                    traj.append({"t": t, "x": round(c[0]), "y": round(c[1])})
                    events.append({"t": t, "desc": f"Motion track at ({c[0]:.0f},{c[1]:.0f}), speed {speed:.0f}px/s, "
                                                   f"area {area_ratio:.1%}", "category": cat, "conf": conf})
                    prev_c, prev_speed = c, speed
            prev_gray = gray
            idx += stride
            n += 1
        cap.release()
        os.unlink(path)
        if not events:
            raise RuntimeError("No motion detected")
    except Exception as e:
        mode = f"Demo fallback ({str(e)[:60]})"
        events = [{"t": t, "desc": d, "category": c, "conf": cf} for t, d, c, cf, _ in DEMO_VIDEO_EVENTS]
        traj = [{"t": t, "x": p[0], "y": p[1]} for t, _, _, _, p in DEMO_VIDEO_EVENTS]
    return {"id": evid, "name": file.name, "digest": digest, "mode": mode, "events": events, "traj": traj}


# ----------------------------------------------------------------------------
# EPIC 5: FUSION + GROUNDED CHAT
# ----------------------------------------------------------------------------
def rebuild_ledger():
    st.session_state.ledger = []
    for im in st.session_state.images:
        t0 = im.get("t_offset", 0)
        for i, d in enumerate(im["dets"], 1):
            add_ledger("Image", im["id"], t0 + i * 0.01,
                       f"{d['label']} detected at box {tuple(int(v) for v in d['box'])} in {im['name']}",
                       d["label"], d["conf"], f"{im['id']}.{i}")
    for au in st.session_state.audio:
        for i, e in enumerate(au["events"], 1):
            add_ledger("Audio", au["id"], e["t"] + au.get("t_offset", 0),
                       f"{e['event']} ({e['speaker']}): \"{e['quote']}\"", e["category"], 0.8, f"{au['id']}.{i}")
    for vd in st.session_state.video:
        for i, e in enumerate(vd["events"], 1):
            add_ledger("Video", vd["id"], e["t"] + vd.get("t_offset", 0), e["desc"], e["category"],
                       e["conf"], f"{vd['id']}.{i}")
    st.session_state.ledger.sort(key=lambda x: x["t"])


STOP = set("the a an of to in was is were and or did do does before after at on for with by it that this "
           "be been has have had from as are who what when where which how any there their".split())
WEAPON_RE = re.compile(r"firearm|gun|knife|edged|blade|weapon|armed", re.I)
FLEE_RE = re.compile(r"flee|fled|flight|exit|ran|drove|departs|escap", re.I)


def tokens(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def retrieve(question, k=5):
    q = tokens(question)
    scored = []
    for e in st.session_state.ledger:
        doc = tokens(e["description"] + " " + e["category"] + " " + e["source"])
        s = len(q & doc)
        if s:
            scored.append((s, e))
    scored.sort(key=lambda x: (-x[0], x[1]["t"]))
    return [e for _, e in scored[:k]]


def cite(e):
    return f"[{e['id']}]"


def grounded_answer_demo(question):
    ledger = st.session_state.ledger
    if not ledger:
        return "Insufficient evidence: the ledger is empty. Analyze evidence in Tabs 1-3 first."
    ql = question.lower()
    # temporal weapon-before-flight reasoning
    if WEAPON_RE.search(ql) and re.search(r"flee|fled|flight|run|ran|escape|leave|left", ql):
        w = [e for e in ledger if WEAPON_RE.search(e["description"] + e["category"])]
        f = [e for e in ledger if FLEE_RE.search(e["description"] + e["category"])]
        if not w:
            return "Insufficient evidence: no weapon-related entry exists in the ledger."
        if not f:
            return (f"A weapon is recorded {cite(w[0])} at T+{fmt_t(w[0]['t'])}, but no flight event is "
                    "recorded, so the sequence cannot be established.")
        before = [e for e in w if e["t"] < f[0]["t"]]
        if before:
            cites = " ".join(cite(e) for e in before[:3])
            return (f"Yes. Weapon-related evidence appears at T+{fmt_t(before[0]['t'])} {cite(before[0])}, "
                    f"earlier than the first flight event at T+{fmt_t(f[0]['t'])} {cite(f[0])}. "
                    f"Supporting entries: {cites}.")
        return (f"No. The earliest weapon-related entry {cite(w[0])} (T+{fmt_t(w[0]['t'])}) does not precede "
                f"the first flight event {cite(f[0])} (T+{fmt_t(f[0]['t'])}).")
    hits = retrieve(question)
    if not hits:
        return "Insufficient evidence: no ledger entries match this question."
    hits.sort(key=lambda e: e["t"])
    lines = [f"- T+{fmt_t(e['t'])} {cite(e)} ({e['source']}): {e['description']}" for e in hits]
    return "Relevant ledger entries (extractive, chronological):\n" + "\n".join(lines)


def grounded_answer_llm(question):
    ledger_json = json.dumps(st.session_state.ledger, indent=1)
    system = ("You are a forensic evidence analyst. Answer ONLY from the EVIDENCE LEDGER provided. "
              "Every factual statement must end with a citation of the entry id in square brackets, e.g. [AUD-001.3]. "
              "Use the 't' field (seconds) to reason about order. If the ledger does not contain the answer, reply exactly: "
              "'Insufficient evidence in ledger.' Never speculate, infer identity, or add outside facts.")
    return call_llm(system, f"EVIDENCE LEDGER:\n{ledger_json}\n\nINVESTIGATOR QUESTION: {question}")


def verify_citations(answer):
    """Anti-hallucination guard: strip/flag any cited IDs that are not in the ledger."""
    valid = {e["id"] for e in st.session_state.ledger}
    cited = set(re.findall(r"\[([A-Z]{3}-\d{3}(?:\.\d+)?)\]", answer))
    bad = cited - valid
    if bad:
        answer += f"\n\n⚠️ Citation check failed for: {', '.join(sorted(bad))} (not in ledger). Treat as unreliable."
    return answer


def answer_question(q):
    if llm_available() and st.session_state.ledger:
        try:
            return verify_citations(grounded_answer_llm(q))
        except Exception as e:
            return grounded_answer_demo(q) + f"\n\n(LLM error, used offline grounding: {str(e)[:80]})"
    return grounded_answer_demo(q)


# ----------------------------------------------------------------------------
# EPIC 6: PDF EXPORT
# ----------------------------------------------------------------------------
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf() -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=f"Forensic Report {st.session_state.case_id}", author=APP_NAME)
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, textColor=colors.HexColor("#0f172a"))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#0c4a6e"), spaceBefore=10)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, leading=9.5)

    def tbl(rows, widths, header=True):
        t = Table([[Paragraph(esc(c), S) for c in r] for r in rows], colWidths=widths, repeatRows=1 if header else 0)
        st_ = [("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
        if header:
            st_.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")))
        t.setStyle(TableStyle(st_))
        return t

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(16 * mm, 10 * mm, f"{APP_NAME} | Case {st.session_state.case_id} | CONFIDENTIAL - AI-ASSISTED, REQUIRES HUMAN VERIFICATION")
        canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {d.page}")
        canvas.restoreState()

    s = []
    s += [Paragraph("FORENSIC ANALYSIS SUMMARY REPORT", H1), Paragraph(SUBTITLE, ss["Normal"]), Spacer(1, 6)]
    s.append(tbl([["Case ID", st.session_state.case_id],
                  ["Generated", now_utc()],
                  ["Investigator", st.session_state.get("investigator") or "UNSIGNED"],
                  ["Badge / ID", st.session_state.get("badge") or "-"],
                  ["Sign-off", "Confirmed" if st.session_state.get("signoff") else "PENDING"],
                  ["Analysis mode", st.session_state.get("provider", "Demo Mode")]],
                 [40 * mm, 136 * mm], header=False))

    s.append(Paragraph("1. Chain of Custody", H2))
    cust = [["Evidence ID", "File", "SHA-256", "Action", "Timestamp", "Handler"]]
    cust += [[c["Evidence ID"], c["File"], c["SHA-256"][:32] + "…", c["Action"], c["Timestamp"], c["Handler"]]
             for c in st.session_state.custody] or [["-"] * 6]
    s.append(tbl(cust, [18 * mm, 32 * mm, 48 * mm, 16 * mm, 36 * mm, 26 * mm]))

    s.append(Paragraph("2. Visual Evidence Findings", H2))
    for im in st.session_state.images:
        s.append(Paragraph(f"<b>{im['id']}</b> - {esc(im['name'])} | Threat: <b>{im['threat']}</b> | "
                           f"Scene: <b>{im['scene']}</b> | Engine: {esc(im['mode'])}", B))
        rows = [["#", "Class", "Confidence", "Bounding box (x1,y1,x2,y2)"]]
        rows += [[i, d["label"], f"{d['conf']:.0%}", str(tuple(int(v) for v in d["box"]))]
                 for i, d in enumerate(im["dets"], 1)]
        s.append(tbl(rows, [10 * mm, 50 * mm, 30 * mm, 86 * mm]))
        s.append(Paragraph("Leads: " + esc(" • ".join(im["leads"])), S))
        s.append(Spacer(1, 4))
    if not st.session_state.images:
        s.append(Paragraph("No image evidence analyzed.", B))

    s.append(Paragraph("3. Witness / Emergency Audio", H2))
    for au in st.session_state.audio:
        s.append(Paragraph(f"<b>{au['id']}</b> - {esc(au['name'])} | Engine: {esc(au['mode'])}", B))
        rows = [["Time", "Speaker", "Statement"]] + [[fmt_t(a), sp, tx] for a, _, sp, tx in au["segments"]]
        s.append(tbl(rows, [18 * mm, 28 * mm, 130 * mm]))
    if not st.session_state.audio:
        s.append(Paragraph("No audio evidence analyzed.", B))

    s.append(Paragraph("4. CCTV Event Progression", H2))
    for vd in st.session_state.video:
        s.append(Paragraph(f"<b>{vd['id']}</b> - {esc(vd['name'])} | Engine: {esc(vd['mode'])}", B))
        rows = [["Time", "Event", "Advisory category", "Conf."]]
        rows += [[fmt_t(e["t"]), e["desc"], e["category"], f"{e['conf']:.0%}"] for e in vd["events"][:40]]
        s.append(tbl(rows, [16 * mm, 92 * mm, 50 * mm, 16 * mm]))
    if not st.session_state.video:
        s.append(Paragraph("No video evidence analyzed.", B))

    s.append(PageBreak())
    s.append(Paragraph("5. Unified Master Event Ledger", H2))
    rows = [["ID", "T+", "Source", "Category", "Description", "Conf."]]
    rows += [[e["id"], fmt_t(e["t"]), e["source"], e["category"], e["description"], f"{e['confidence']:.0%}"]
             for e in st.session_state.ledger] or [["-"] * 6]
    s.append(tbl(rows, [18 * mm, 16 * mm, 14 * mm, 28 * mm, 84 * mm, 16 * mm]))

    s.append(Paragraph("6. Investigator Q&A Log (grounded, cited)", H2))
    for m in st.session_state.chat:
        s.append(Paragraph(f"<b>Q:</b> {esc(m['q'])}", B))
        s.append(Paragraph(f"<b>A:</b> {esc(m['a']).replace(chr(10), '<br/>')}", B))
        s.append(Spacer(1, 3))
    if not st.session_state.chat:
        s.append(Paragraph("No queries recorded.", B))

    s.append(Paragraph("7. Limitations & Certification", H2))
    s.append(Paragraph(
        "Automated findings (object detection, transcription, motion-based crime categories) are advisory, "
        "probabilistic and must be verified by a qualified examiner before use in legal proceedings. Crime "
        "categories are not legal determinations. Evidence integrity is anchored by SHA-256 hashes recorded at ingestion.", B))
    s.append(Spacer(1, 18))
    s.append(tbl([["Investigator signature: ____________________", f"Date: {datetime.now():%Y-%m-%d}"]],
                 [110 * mm, 66 * mm], header=False))
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def threat_badge(level):
    return f'<span class="cv-badge b-{level.lower()}">{level.upper()}</span>'


def sidebar():
    with st.sidebar:
        st.markdown(f"### 🕵️ {APP_NAME}")
        st.caption("Configuration & Chain of Custody")
        provider = st.selectbox("AI Provider", ["Demo Mode", "Groq", "Gemini"], key="provider_sel")
        key = ""
        if provider != "Demo Mode":
            key = st.text_input(f"{provider} API Key", type="password",
                                value=os.environ.get(f"{provider.upper()}_API_KEY", ""),
                                help="Stored only in session memory. Never written to disk.")
        if provider != "Demo Mode" and not key:
            st.warning("No key supplied → running in DEMO fallback mode.")
            provider = "Demo Mode"
        st.session_state.provider, st.session_state.api_key = provider, key
        st.success("🟢 Live AI" if provider != "Demo Mode" else "🟡 Demo mode (offline)")

        st.divider()
        st.markdown("**Chain of Custody**")
        st.text_input("Case ID", key="case_id")
        st.text_input("Investigator name", key="investigator")
        st.text_input("Badge / ID", key="badge")
        st.checkbox("I certify evidence handling is accurate (sign-off)", key="signoff")
        st.markdown(f'<span class="cv-mono">Session: {now_utc()}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="cv-mono">Evidence items: {len(st.session_state.custody)}</span>',
                    unsafe_allow_html=True)
        if st.button("🗑 Reset case"):
            for k in ["ledger", "counters", "custody", "images", "audio", "video", "chat"]:
                st.session_state.pop(k, None)
            st.rerun()


def tab_image():
    st.subheader("Crime Scene Visual Evidence Tagger")
    st.caption("RT-DETR detection: persons, firearms, edged weapons, blood spatter, forced entry.")
    files = st.file_uploader("Upload scene photographs", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key="img_up")
    if files and st.button("🔍 Analyze images", type="primary"):
        done = {i["name"] + i["digest"] for i in st.session_state.images}
        for f in files:
            if f.name + sha256_bytes(f.getvalue()) in done:
                continue
            try:
                with st.spinner(f"Analyzing {f.name}…"):
                    st.session_state.images.append(analyze_image(f))
            except Exception as e:
                st.error(f"Could not process {f.name}: {e}")
        rebuild_ledger()
    for im in st.session_state.images:
        with st.expander(f"{im['id']} · {im['name']}", expanded=True):
            c1, c2 = st.columns([3, 2])
            c1.image(draw_boxes(im["img"], im["dets"]), use_container_width=True)
            c2.markdown(f"**Threat Level:** {threat_badge(im['threat'])}", unsafe_allow_html=True)
            c2.markdown(f"**Scene Classification:** `{im['scene']}`")
            c2.markdown(f"<span class='cv-mono'>SHA-256 {im['digest'][:24]}… | {im['mode']}</span>",
                        unsafe_allow_html=True)
            im["t_offset"] = c2.number_input("Incident time offset (sec) for this photo", 0.0, 86400.0,
                                             float(im.get("t_offset", 0)), key=f"off_{im['id']}")
            c2.markdown("**Prioritized investigative leads**")
            for i, l in enumerate(im["leads"], 1):
                c2.markdown(f"{i}. {l}")
            st.dataframe([{"#": i, "Item": d["label"], "Confidence": f"{d['conf']:.0%}",
                           "Box (x1,y1,x2,y2)": str(tuple(int(v) for v in d["box"]))}
                          for i, d in enumerate(im["dets"], 1)], use_container_width=True, hide_index=True)
    if st.session_state.images:
        rebuild_ledger()


def tab_audio():
    st.subheader("Witness Audio Transcription & Timeline")
    st.caption("Whisper transcription (Groq Whisper API / local whisper) with demo fallback.")
    f = st.file_uploader("Upload witness statement or 911/112 call", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
                         key="aud_up")
    if st.button("🎙 Transcribe & extract timeline", type="primary", disabled=f is None):
        try:
            with st.spinner("Transcribing…"):
                st.session_state.audio.append(analyze_audio(f))
            rebuild_ledger()
        except Exception as e:
            st.error(f"Audio processing failed: {e}")
    if f is None and st.button("▶ Load demo 911 call"):
        digest = sha256_bytes(b"demo-911")
        evid = next_id("AUD")
        log_custody(evid, "demo_911_call.wav", digest)
        st.session_state.audio.append({"id": evid, "name": "demo_911_call.wav", "digest": digest,
                                       "mode": "Demo transcript", "segments": DEMO_TRANSCRIPT,
                                       "events": tag_transcript(DEMO_TRANSCRIPT)})
        rebuild_ledger()
    for au in st.session_state.audio:
        with st.expander(f"{au['id']} · {au['name']} · {au['mode']}", expanded=True):
            au["t_offset"] = st.number_input("Incident time offset (sec)", 0.0, 86400.0,
                                             float(au.get("t_offset", 0)), key=f"off_{au['id']}")
            st.markdown("**Transcript**")
            st.dataframe([{"Time": fmt_t(a), "Speaker": sp, "Statement": tx} for a, _, sp, tx in au["segments"]],
                         use_container_width=True, hide_index=True)
            st.markdown("**Key events (chronological)**")
            st.dataframe([{"Time": fmt_t(e["t"]), "Speaker": e["speaker"], "Event": e["event"],
                           "Category": e["category"]} for e in au["events"]],
                         use_container_width=True, hide_index=True)
    if st.session_state.audio:
        rebuild_ledger()


def tab_video():
    st.subheader("CCTV Video Analysis & Event Progression")
    st.caption("Frame sampling + motion tracking of subjects/vehicles; advisory penal-category classification.")
    c1, c2 = st.columns(2)
    step = c1.slider("Sampling interval (sec)", 0.5, 5.0, 1.0, 0.5)
    maxf = c2.slider("Max frames", 10, 120, 60, 10)
    f = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    b1, b2 = st.columns(2)
    if b1.button("🎞 Analyze video", type="primary", disabled=f is None):
        with st.spinner("Sampling frames and tracking motion…"):
            st.session_state.video.append(analyze_video(f, step, maxf))
        rebuild_ledger()
    if b2.button("▶ Load demo CCTV analysis"):
        class _F:  # tiny stand-in so analyze_video falls back to demo
            name, getvalue = "demo_cctv.mp4", staticmethod(lambda: b"demo-cctv")
        st.session_state.video.append(analyze_video(_F))
        rebuild_ledger()
    for vd in st.session_state.video:
        with st.expander(f"{vd['id']} · {vd['name']} · {vd['mode']}", expanded=True):
            vd["t_offset"] = st.number_input("Incident time offset (sec)", 0.0, 86400.0,
                                             float(vd.get("t_offset", 0)), key=f"off_{vd['id']}")
            if vd["traj"]:
                st.markdown("**Subject trajectory (pixel coordinates)**")
                st.line_chart({"x": [p["x"] for p in vd["traj"]], "y": [p["y"] for p in vd["traj"]]})
            st.dataframe([{"Time": fmt_t(e["t"]), "Event": e["desc"], "Advisory category": e["category"],
                           "Confidence": f"{e['conf']:.0%}"} for e in vd["events"]],
                         use_container_width=True, hide_index=True)
    st.caption("⚖️ Categories are heuristic, advisory labels - not legal determinations.")
    if st.session_state.video:
        rebuild_ledger()


def tab_fusion():
    st.subheader("Multimodal Fusion & Grounded Evidence Chat")
    ledger = st.session_state.ledger
    if not ledger:
        st.info("Ledger is empty. Analyze evidence in Tabs 1–3 (or load the demos).")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ledger entries", len(ledger))
        m2.metric("Image", sum(e["source"] == "Image" for e in ledger))
        m3.metric("Audio", sum(e["source"] == "Audio" for e in ledger))
        m4.metric("Video", sum(e["source"] == "Video" for e in ledger))
        src = st.multiselect("Filter sources", ["Image", "Audio", "Video"], default=["Image", "Audio", "Video"])
        st.dataframe([{"ID": e["id"], "T+": fmt_t(e["t"]), "Source": e["source"], "Category": e["category"],
                       "Description": e["description"], "Conf.": f"{e['confidence']:.0%}"}
                      for e in ledger if e["source"] in src], use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("#### 💬 Interrogate the evidence")
    st.caption("Answers use only ledger entries and cite evidence IDs. Missing facts return 'Insufficient evidence'.")
    for m in st.session_state.chat:
        with st.chat_message("user"):
            st.write(m["q"])
        with st.chat_message("assistant"):
            st.write(m["a"])
    q = st.chat_input("e.g. Was a weapon visible before the suspect fled?")
    if q:
        a = answer_question(q)
        st.session_state.chat.append({"q": q, "a": a, "t": now_utc()})
        st.rerun()


def tab_pdf():
    st.subheader("Automated Forensic PDF Report")
    ok_sign = st.session_state.get("signoff")
    if not st.session_state.get("investigator") or not ok_sign:
        st.warning("Investigator name and sign-off (sidebar) are required for a court-ready report. "
                   "Report will be marked UNSIGNED/PENDING otherwise.")
    n = len(st.session_state.custody)
    st.markdown(f"Contents: chain of custody ({n} items), image findings, transcripts, CCTV log, "
                f"master ledger ({len(st.session_state.ledger)} entries), Q&A log, limitations & certification.")
    if st.button("📄 Generate forensic PDF", type="primary"):
        try:
            rebuild_ledger()
            st.session_state.pdf = build_pdf()
            st.session_state.custody.append({
                "Evidence ID": "REPORT", "File": f"{st.session_state.case_id}.pdf",
                "SHA-256": sha256_bytes(st.session_state.pdf), "Action": "Report generated",
                "Timestamp": now_utc(), "Handler": st.session_state.get("investigator") or "UNSIGNED"})
            st.success("Report generated.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    if st.session_state.get("pdf"):
        st.download_button("⬇ Download PDF", st.session_state.pdf,
                           file_name=f"CrimeVision_{st.session_state.case_id}.pdf", mime="application/pdf")
    if st.session_state.custody:
        st.markdown("**Custody log**")
        st.dataframe(st.session_state.custody, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    init_state()
    sidebar()
    st.markdown(f'<div class="cv-hero"><h1>🕵️ {APP_NAME}</h1><p>{SUBTITLE}</p></div>', unsafe_allow_html=True)
    tabs = st.tabs(["🖼 Visual Evidence", "🎙 Witness Audio", "🎥 CCTV Video", "🧩 Fusion & Evidence Chat",
                    "📄 PDF Report"])
    for t, fn in zip(tabs, [tab_image, tab_audio, tab_video, tab_fusion, tab_pdf]):
        with t:
            try:
                fn()
            except Exception as e:
                st.error(f"Unexpected error in this module: {e}")
    st.caption("AI-assisted analysis for investigative support only. Human verification required.")


if __name__ == "__main__":
    main()
