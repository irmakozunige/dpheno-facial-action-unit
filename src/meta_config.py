"""
meta_config.py
--------------
Shared configuration for the meta-analysis pipeline:
  - LLM client (Gemini 3.5 Flash Lite)
  - AU reference table with clinical annotations
  - Extraction prompt
  - Rate limiting helpers

Imported by:  01_meta_analysis.py
              meta_retrieval.py
"""
import os, json, time, textwrap, re, math, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm
import requests
from Bio import Entrez

# ── LLM client ───────────────────────────────────────────────────────────────
# Default: Gemini 2.0 Flash (free, 1500 req/day, 1M context)
# Fallback: any OpenAI-compatible endpoint (local vLLM, Ollama, etc.)
#
# Setup:
#   export GEMINI_API_KEY=...   # free at aistudio.google.com
#   OR
#   export LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3  # Ollama
#   export LLM_BASE_URL=http://localhost:8000/v1   LLM_MODEL=Qwen/Qwen2.5-72B  # vLLM

LLM_BACKEND      = os.environ.get("LLM_BACKEND", "gemini")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
LLM_BASE_URL     = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL        = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_API_KEY      = os.environ.get("LLM_API_KEY", GEMINI_API_KEY)

# Gemini free tier limits (enforced by the wrapper below)
# gemini-3.5-flash-lite:  1500 req/day,  15 req/min,  1M tokens/min
# gemini-3.5-flash-lite: 1500 req/day, 15 req/min
# gemini-3.5-flash-lite:     50 req/day,    2 req/min   (use only for narrative summary)
GEMINI_RPM        = int(os.environ.get("GEMINI_RPM", "14"))   # stay under 15
GEMINI_RPD        = int(os.environ.get("GEMINI_RPD", "490"))   # stay under 500 (gemini-3.5-flash-lite limit)
_gemini_req_times: list[float] = []
_gemini_day_count: list[int]   = [0]


def _gemini_rate_gate():
    """Block until it is safe to send another Gemini request."""
    import time as _t
    now = _t.time()
    # per-minute window: drop timestamps older than 60 s
    _gemini_req_times[:] = [t for t in _gemini_req_times if now - t < 60]
    if len(_gemini_req_times) >= GEMINI_RPM:
        wait = 60 - (now - _gemini_req_times[0]) + 1
        print(f"    [Gemini] rate limit — waiting {wait:.0f}s")
        _t.sleep(wait)
    if _gemini_day_count[0] >= GEMINI_RPD:
        raise RuntimeError("Gemini daily quota reached (1500 req). Resume tomorrow.")
    _gemini_req_times.append(_t.time())
    _gemini_day_count[0] += 1


def llm_chat(prompt: str, max_tokens: int = 1200, prefer_pro: bool = False) -> str:
    """
    Unified LLM call. Routes by LLM_BACKEND:
      gemini        → official google-genai SDK (>= 2.3.0), rate-limited
      openai_compat → any OpenAI-compatible server (Ollama, vLLM, LM Studio)
    prefer_pro=True → uses gemini-3.5-flash-lite for final narrative.
    """
    import time as _t

    if LLM_BACKEND == "gemini" or (not LLM_BASE_URL and GEMINI_API_KEY):
        from google import genai as _genai
        from google.genai import types as _gtypes
        _client = _genai.Client(api_key=GEMINI_API_KEY)
        model_name = LLM_MODEL  # always use gemini-3.5-flash-lite
        for attempt in range(5):
            _gemini_rate_gate()
            try:
                response = _client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=_gtypes.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                    ),
                )
                if not response.candidates:
                    return ""
                return (response.text or "").strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    wait = 2**attempt * 10
                    print(f"    [Gemini] 429 — backing off {wait}s")
                    _t.sleep(wait)
                elif "503" in err:
                    _t.sleep(30)
                else:
                    print(f"    [Gemini] error attempt {attempt+1}: {e}")
                    _t.sleep(5)
        return ""

    else:
        # OpenAI-compatible: Ollama / vLLM / LM Studio / any local server
        from openai import OpenAI as _OAI
        c    = _OAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "EMPTY", timeout=600)
        resp = c.chat.completions.create(
            model       = LLM_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = max_tokens,
            temperature = 0.0,
        )
        text = resp.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ── Config ────────────────────────────────────────────────────────────────────
Entrez.email = "irmak.ozarslan@unige.ch"
MAX_PAPERS   = 400   # collect broadly
TOP_K        = 30    # keep only top-ranked after scoring
RESULTS_DIR  = "results/meta"
os.makedirs(RESULTS_DIR, exist_ok=True)
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")
# Gemini API key — free at https://aistudio.google.com (click "Get API key")
# Free limits: 1500 req/day, 15 req/min for Flash; 50 req/day for Pro
# GEMINI_API_KEY already read above in the LLM client block
BROWSER_UA   = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SCIHUB_MIRRORS = ["https://sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru"]

# ── Source-specific queries ───────────────────────────────────────────────────
# PubMed uses MeSH terms for precision; other APIs use free-text variants

PUBMED_QUERY = (
    '(("facial expression"[MeSH] OR "action unit"[tiab] OR "AU1"[tiab] OR '
    '"AU2"[tiab] OR "AU4"[tiab] OR "AU6"[tiab] OR "AU12"[tiab] OR '
    '"AU15"[tiab] OR "AU25"[tiab] OR "AU45"[tiab] OR '
    '"OpenFace"[tiab] OR "FACS"[tiab] OR "facial action coding"[tiab] OR '
    '"facial action unit"[tiab]) AND '
    '("schizophrenia"[MeSH] OR "psychosis"[MeSH] OR '
    '"schizoaffective disorder"[MeSH] OR "negative symptoms"[tiab] OR '
    '"blunted affect"[tiab] OR "flat affect"[tiab] OR "apathy"[tiab]))'
)

# Free-text queries for APIs that don't support MeSH
# (Semantic Scholar, OpenAlex, CORE, CrossRef, Google Scholar)
QUERIES = [
    "facial action units schizophrenia negative symptoms",
    "facial action unit AU schizophrenia blunted affect",
    "OpenFace schizophrenia negative symptoms automated",
    "FACS facial action coding schizophrenia psychosis",
    "facial expressivity flat affect schizophrenia quantitative",
    "AU12 AU6 AU4 smile schizophrenia",
    "blunted affect facial movement automated analysis",
    "facial expression apathy schizophrenia machine learning",
    "facial affect psychosis negative symptoms measurement",
    "automated facial analysis schizophrenia clinical interview",
]

# ── FACS AU reference with clinical annotations ──────────────────────────────
# clinical_direction: expected direction in schizophrenia vs HC
# clinical_scale: scale the effect relates to
# priority: HIGH = directly linked to apathy/negative symptoms
AU_REFERENCE = {
    "AU1":  {"label":"Inner Brow Raise",       "muscle":"frontalis (medial)",    "face_region":"brow",  "functional_tag":"sadness/concern",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"HIGH"},
    "AU2":  {"label":"Outer Brow Raise",        "muscle":"frontalis (lateral)",   "face_region":"brow",  "functional_tag":"surprise/attention",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"HIGH"},
    "AU4":  {"label":"Brow Lowerer",            "muscle":"corrugator supercilii", "face_region":"brow",  "functional_tag":"anger/sadness/concentration",
             "clinical_direction":"decreased",  "clinical_scale":"PANSS_N",            "priority":"HIGH"},
    "AU5":  {"label":"Upper Lid Raiser",        "muscle":"levator palpebrae",     "face_region":"eye",   "functional_tag":"surprise/fear/arousal",
             "clinical_direction":"increased",  "clinical_scale":"depression",         "priority":"HIGH",
             "note":"increased frequency in depression; arousal/vigilance marker"},
    "AU6":  {"label":"Cheek Raiser",            "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"enjoyment/Duchenne smile",
             "clinical_direction":"increased",  "clinical_scale":"depression",         "priority":"HIGH",
             "note":"increased in depression; Duchenne smile component; reduced in flat affect"},
    "AU7":  {"label":"Lid Tightener",           "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"anger/disgust",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"MEDIUM"},
    "AU9":  {"label":"Nose Wrinkler",           "muscle":"levator labii",         "face_region":"nose",  "functional_tag":"disgust",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"LOW"},
    "AU10": {"label":"Upper Lip Raiser",        "muscle":"levator labii",         "face_region":"mouth", "functional_tag":"disgust/contempt",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"LOW"},
    "AU12": {"label":"Lip Corner Puller",       "muscle":"zygomaticus major",     "face_region":"mouth", "functional_tag":"smile / positive affect",
             "clinical_direction":"decreased",  "clinical_scale":"PANSS_N",            "priority":"HIGH",
             "note":"negatively correlated with PANSS Negative summative scale; key smile biomarker"},
    "AU14": {"label":"Dimpler",                 "muscle":"buccinator",            "face_region":"mouth", "functional_tag":"amusement/subtle smile",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"MEDIUM"},
    "AU15": {"label":"Lip Corner Depressor",    "muscle":"depressor anguli oris", "face_region":"mouth", "functional_tag":"sadness/disgust",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"HIGH"},
    "AU17": {"label":"Chin Raiser",             "muscle":"mentalis",              "face_region":"mouth", "functional_tag":"doubt/sadness",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"MEDIUM"},
    "AU20": {"label":"Lip Stretcher",           "muscle":"risorius",              "face_region":"mouth", "functional_tag":"fear",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"LOW"},
    "AU23": {"label":"Lip Tightener",           "muscle":"orbicularis oris",      "face_region":"mouth", "functional_tag":"anger",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"LOW"},
    "AU24": {"label":"Lip Pressor",             "muscle":"orbicularis oris",      "face_region":"mouth", "functional_tag":"anger/determination",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"LOW"},
    "AU25": {"label":"Lips Part",               "muscle":"depressor labii",       "face_region":"mouth", "functional_tag":"speech/surprise",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"MEDIUM"},
    "AU26": {"label":"Jaw Drop",                "muscle":"masseter relaxation",   "face_region":"mouth", "functional_tag":"surprise/open mouth",
             "clinical_direction":"decreased",  "clinical_scale":"flat_affect",        "priority":"LOW"},
    "AU43": {"label":"Eyes Closed",             "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"avoidance/disengagement",
             "clinical_direction":"increased",  "clinical_scale":"negative_symptoms",  "priority":"MEDIUM"},
    "AU45": {"label":"Blink",                   "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"blink rate — dopamine/attention proxy",
             "clinical_direction":"decreased",  "clinical_scale":"schizophrenia",      "priority":"HIGH",
             "note":"AU45 occurrence tracked but not used as standalone biomarker"},
    "AU46": {"label":"Wink",                    "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"deliberate social signal",
             "clinical_direction":"decreased",  "clinical_scale":"social_cognition",   "priority":"LOW"},
    # ── Composite / head movement biomarkers (not FACS AUs but extracted by OpenFace) ──
    "HEAD_EXPR":    {"label":"Overall Expressivity Index (normalized)", "face_region":"composite",
                     "functional_tag":"mean AU intensity / baseline",
                     "clinical_direction":"decreased", "clinical_scale":"PANSS_N", "priority":"HIGH",
                     "note":"Truelove-Hill 2021 PMC8817208 — normalized composite expressivity"},
    "HEAD_MOV":     {"label":"Head Movement Magnitude",  "face_region":"head",
                     "functional_tag":"mean frame-to-frame angular displacement",
                     "clinical_direction":"decreased", "clinical_scale":"negative_symptoms", "priority":"HIGH",
                     "note":"reduced spontaneous head movement = motor poverty / apathy"},
    "HEAD_STILL":   {"label":"Head Stillness Index",     "face_region":"head",
                     "functional_tag":"proportion frames with near-zero movement",
                     "clinical_direction":"increased", "clinical_scale":"negative_symptoms", "priority":"HIGH"},
}
AU_REF_JSON = json.dumps(AU_REFERENCE, indent=2)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _paper(uid,source,fa,year,title,abstract,journal="",doi=""):
    return {"id":uid,"source":source,"first_author":fa,"year":str(year),
            "title":title,"abstract":abstract,"journal":journal,
            "doi":doi,"full_text":None,"full_text_source":None}

def _fp(title):
    return hashlib.md5(title.lower().strip().encode()).hexdigest()

def _get(url,params=None,headers=None,timeout=15,**kw):
    try:
        return requests.get(url,params=params,
                            headers={**(headers or {}),"User-Agent":BROWSER_UA},
                            timeout=timeout,**kw)
    except Exception:
        return None