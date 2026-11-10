"""
01_meta_analysis.py  (v3 — full-text + semantic AU mapping)
-------------------------------------------------------------
Sources:
  PubMed · Semantic Scholar · Google Scholar · CrossRef

Full-text retrieval (where available):
  1. Unpaywall  — open-access PDFs (no credentials needed)
  2. EZproxy    — institutional access (set EZPROXY_BASE + cookie below)
  3. Direct DOI fallback with institutional User-Agent

Per-paper extraction (Claude):
  - AUs studied
  - For each AU: direction, mean/SD SZ vs HC, frequency vs intensity
  - Semantic label  (e.g. AU12 → "Lip Corner Puller / smile")
  - Functional tag  (e.g. "positive affect expression")
  - Feature type    (intensity | frequency | duration | presence)

Outputs in results/meta/:
  raw_extractions.json
  au_long.csv               one row per paper x AU (with semantic labels)
  au_feature_vocab.csv      MASTER TABLE: AU → label → direction → feature type
  meta_results.csv          pooled Hedges g per AU
  forest_AU**.png
  bubble_summary.png
  direction_heatmap.png
  source_breakdown.png
  meta_analysis_summary.txt

Requirements:
  pip install anthropic biopython requests scholarly semanticscholar
              pandas numpy matplotlib scipy tqdm PyMuPDF
  ANTHROPIC_API_KEY env var
  Optional: EZPROXY_BASE, EZPROXY_COOKIE  env vars for institutional access
"""

import os, json, time, textwrap, re, math, hashlib, base64
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
import anthropic
from Bio import Entrez

# ── Config ──────────────────────────────────────────────────────────────────
Entrez.email  = "your@email.com"
MAX_PAPERS    = 100
RESULTS_DIR   = "results/meta"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOKEN  = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=TOKEN)

# Institutional access — set these in your environment or fill directly
# EZPROXY_BASE   = os.environ.get("EZPROXY_BASE", "https://ezproxy.youruniversity.edu/login?url=")
# EZPROXY_COOKIE = os.environ.get("EZPROXY_COOKIE", "")   # from browser DevTools after login
EZPROXY_BASE   = os.environ.get("EZPROXY_BASE", "")
EZPROXY_COOKIE = os.environ.get("EZPROXY_COOKIE", "")

QUERIES = [
    "facial action units schizophrenia",
    "facial expression flat affect schizophrenia",
    "facial action units negative symptoms",
    "apathy facial expression schizophrenia",
    "facial affect automated analysis psychosis",
    "OpenFace schizophrenia negative symptoms",
    "FACS facial coding schizophrenia",
    "blunted affect facial movement schizophrenia",
]

# ── FACS AU master reference ─────────────────────────────────────────────────
# This is the ground-truth semantic map used to standardise extraction.
# Claude is prompted to tag each AU against this table.
AU_REFERENCE = {
    "AU1":  {"label": "Inner Brow Raise",        "muscle": "frontalis (medial)",    "face_region": "brow",   "functional_tag": "sadness/concern"},
    "AU2":  {"label": "Outer Brow Raise",         "muscle": "frontalis (lateral)",   "face_region": "brow",   "functional_tag": "surprise/attention"},
    "AU4":  {"label": "Brow Lowerer",             "muscle": "corrugator supercilii", "face_region": "brow",   "functional_tag": "anger/sadness/concentration"},
    "AU5":  {"label": "Upper Lid Raiser",         "muscle": "levator palpebrae",     "face_region": "eye",    "functional_tag": "surprise/fear"},
    "AU6":  {"label": "Cheek Raiser",             "muscle": "orbicularis oculi",     "face_region": "eye",    "functional_tag": "enjoyment/Duchenne smile component"},
    "AU7":  {"label": "Lid Tightener",            "muscle": "orbicularis oculi",     "face_region": "eye",    "functional_tag": "anger/disgust"},
    "AU9":  {"label": "Nose Wrinkler",            "muscle": "levator labii",         "face_region": "nose",   "functional_tag": "disgust"},
    "AU10": {"label": "Upper Lip Raiser",         "muscle": "levator labii",         "face_region": "mouth",  "functional_tag": "disgust/contempt"},
    "AU12": {"label": "Lip Corner Puller",        "muscle": "zygomaticus major",     "face_region": "mouth",  "functional_tag": "smile / positive affect"},
    "AU14": {"label": "Dimpler",                  "muscle": "buccinator",            "face_region": "mouth",  "functional_tag": "amusement/subtle smile"},
    "AU15": {"label": "Lip Corner Depressor",     "muscle": "depressor anguli oris", "face_region": "mouth",  "functional_tag": "sadness/disgust"},
    "AU17": {"label": "Chin Raiser",              "muscle": "mentalis",              "face_region": "mouth",  "functional_tag": "doubt/sadness"},
    "AU20": {"label": "Lip Stretcher",            "muscle": "risorius",              "face_region": "mouth",  "functional_tag": "fear"},
    "AU23": {"label": "Lip Tightener",            "muscle": "orbicularis oris",      "face_region": "mouth",  "functional_tag": "anger"},
    "AU24": {"label": "Lip Pressor",              "muscle": "orbicularis oris",      "face_region": "mouth",  "functional_tag": "anger/determination"},
    "AU25": {"label": "Lips Part",                "muscle": "depressor labii",       "face_region": "mouth",  "functional_tag": "speech/surprise"},
    "AU26": {"label": "Jaw Drop",                 "muscle": "masseter relaxation",   "face_region": "mouth",  "functional_tag": "surprise/open mouth"},
    "AU43": {"label": "Eyes Closed",              "muscle": "orbicularis oculi",     "face_region": "eye",    "functional_tag": "sleep/relaxation/avoidance"},
    "AU45": {"label": "Blink",                    "muscle": "orbicularis oculi",     "face_region": "eye",    "functional_tag": "blink rate (attention/dryness)"},
    "AU46": {"label": "Wink",                     "muscle": "orbicularis oculi",     "face_region": "eye",    "functional_tag": "deliberate social signal"},
}
AU_REF_JSON = json.dumps(AU_REFERENCE, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 1. RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def _paper_dict(uid, source, first_author, year, title, abstract, journal="", doi=""):
    return {"id": uid, "source": source, "first_author": first_author,
            "year": str(year), "title": title, "abstract": abstract,
            "journal": journal, "doi": doi, "full_text": None}


def search_pubmed(query, max_results=20):
    try:
        h   = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        ids = Entrez.read(h)["IdList"]; h.close()
        if not ids: return []
        h   = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        recs= Entrez.read(h)["PubmedArticle"]; h.close()
    except Exception as e:
        print(f"    PubMed: {e}"); return []
    papers = []
    for rec in recs:
        try:
            art  = rec["MedlineCitation"]["Article"]
            title= str(art.get("ArticleTitle",""))
            abst = " ".join(str(p) for p in art.get("Abstract",{}).get("AbstractText",[]))
            year = str(art.get("Journal",{}).get("JournalIssue",{}).get("PubDate",{}).get("Year","?"))
            aus  = art.get("AuthorList",[])
            fa   = str(aus[0].get("LastName","")) if aus else ""
            pmid = str(rec["MedlineCitation"]["PMID"])
            jnl  = str(art.get("Journal",{}).get("Title",""))
            # try to get DOI
            ids2 = rec.get("PubmedData",{}).get("ArticleIdList",[])
            doi  = next((str(x) for x in ids2 if x.attributes.get("IdType")=="doi"), "")
            if abst.strip():
                papers.append(_paper_dict(f"pmid_{pmid}","PubMed",fa,year,title,abst,jnl,doi))
        except Exception: continue
    return papers


def search_semantic_scholar(query, max_results=20):
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                         params={"query":query,"limit":max_results,
                                 "fields":"title,abstract,year,authors,externalIds,venue"},
                         timeout=15)
        data = r.json().get("data",[])
    except Exception as e:
        print(f"    S2: {e}"); return []
    papers = []
    for p in data:
        abst = p.get("abstract") or ""
        if not abst.strip(): continue
        aus  = p.get("authors",[])
        fa   = aus[0].get("name","").split()[-1] if aus else ""
        doi  = (p.get("externalIds") or {}).get("DOI","")
        papers.append(_paper_dict(f"ss_{p.get('paperId','')}","SemanticScholar",
                                   fa, p.get("year","?"), p.get("title",""),
                                   abst, p.get("venue",""), doi))
    return papers


def search_google_scholar(query, max_results=10):
    try:
        from scholarly import scholarly as sch
        results = sch.search_pubs(query)
    except Exception as e:
        print(f"    GS: {e}"); return []
    papers = []
    for i, pub in enumerate(results):
        if i >= max_results: break
        try:
            bib  = pub.get("bib",{})
            abst = bib.get("abstract","") or ""
            ttl  = bib.get("title","") or ""
            if not abst.strip() or not ttl.strip(): continue
            fa   = str(bib.get("author","")).split(" and ")[0].split(",")[0].strip()
            year = str(bib.get("pub_year","?"))
            uid  = hashlib.md5(ttl.encode()).hexdigest()[:10]
            papers.append(_paper_dict(f"gs_{uid}","GoogleScholar",fa,year,ttl,abst,
                                       bib.get("venue",""),""))
            time.sleep(1.5)
        except Exception: continue
    return papers


def search_crossref(query, max_results=20):
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query":query,"rows":max_results,
                                 "select":"DOI,title,abstract,author,published,container-title",
                                 "mailto":"your@email.com"},
                         timeout=15)
        items = r.json().get("message",{}).get("items",[])
    except Exception as e:
        print(f"    CrossRef: {e}"); return []
    papers = []
    for item in items:
        abst = re.sub(r"<[^>]+>"," ", item.get("abstract","") or "").strip()
        ttls = item.get("title",[])
        ttl  = ttls[0] if ttls else ""
        if not abst or len(abst) < 80: continue
        aus  = item.get("author",[])
        fa   = aus[0].get("family","") if aus else ""
        pub  = item.get("published",{}).get("date-parts",[[None]])[0]
        year = str(pub[0]) if pub and pub[0] else "?"
        doi  = item.get("DOI","")
        jnl  = (item.get("container-title") or [""])[0]
        papers.append(_paper_dict(f"cr_{doi.replace('/','_')}","CrossRef",
                                   fa,year,ttl,abst,jnl,doi))
    return papers


def collect_all_papers():
    seen, papers = set(), []
    sources = [
        ("PubMed",          search_pubmed,          20),
        ("SemanticScholar", search_semantic_scholar, 20),
        ("GoogleScholar",   search_google_scholar,   10),
        ("CrossRef",        search_crossref,         20),
    ]
    for name, fn, n in sources:
        print(f"\n  [{name}]")
        for q in QUERIES:
            print(f"    → {q}")
            try:
                res = fn(q, max_results=n)
            except Exception as e:
                print(f"      error: {e}"); res = []
            new = 0
            for p in res:
                fp = hashlib.md5(p["title"].lower().encode()).hexdigest()
                if fp not in seen:
                    seen.add(fp); papers.append(p); new += 1
            print(f"      +{new}  (total {len(papers)})")
            time.sleep(0.4)
    return papers[:MAX_PAPERS]


# ═══════════════════════════════════════════════════════════════════════════
# 2. FULL-TEXT RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def try_unpaywall(doi: str, email: str = "your@email.com") -> str | None:
    """Try to get open-access PDF URL from Unpaywall."""
    if not doi: return None
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                         params={"email": email}, timeout=10)
        data = r.json()
        loc  = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def try_ezproxy(doi: str) -> str | None:
    """Build EZproxy URL for institutional full-text access."""
    if not EZPROXY_BASE or not doi:
        return None
    return f"{EZPROXY_BASE}https://doi.org/{doi}"


def fetch_pdf_text(url: str) -> str | None:
    """Download a PDF and extract text with PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        headers = {"User-Agent": "Mozilla/5.0 (research bot; your@email.com)"}
        cookies = {}
        if EZPROXY_COOKIE:
            # parse "name=value; name2=value2" cookie string
            for part in EZPROXY_COOKIE.split(";"):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    cookies[k.strip()] = v.strip()

        resp = requests.get(url, headers=headers, cookies=cookies,
                            timeout=30, stream=True)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type","")
        if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
            return None

        pdf_bytes = resp.content
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        # keep only results/methods sections — most relevant for AU stats
        text = extract_results_section(text)
        return text[:8000]   # cap to stay within Claude context
    except Exception:
        return None


def extract_results_section(text: str) -> str:
    """
    Heuristic: grab text between 'Results' and 'Discussion' headers.
    Falls back to full text if headers not found.
    """
    patterns = [
        r"(?i)\bresults?\b(.+?)(?=\b(?:discussion|conclusion|references)\b)",
        r"(?i)\bmethods?\b(.+?)(?=\b(?:results?|discussion)\b)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(0)[:8000]
    return text[:8000]


def enrich_with_full_text(papers: list[dict]) -> list[dict]:
    """Try to fetch full text for each paper; store in paper['full_text']."""
    print("\n  Attempting full-text retrieval...")
    for p in tqdm(papers):
        doi = p.get("doi","")
        text = None

        # 1. Unpaywall (open access)
        url = try_unpaywall(doi)
        if url:
            text = fetch_pdf_text(url)
            if text:
                p["full_text_source"] = "unpaywall"

        # 2. EZproxy (institutional)
        if not text:
            url = try_ezproxy(doi)
            if url:
                text = fetch_pdf_text(url)
                if text:
                    p["full_text_source"] = "ezproxy"

        p["full_text"] = text
        time.sleep(0.3)

    n_ft = sum(1 for p in papers if p.get("full_text"))
    print(f"    Full text retrieved for {n_ft}/{len(papers)} papers")
    return papers


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLAUDE EXTRACTION — with semantic AU labelling
# ═══════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = textwrap.dedent("""
You are a clinical research assistant doing a systematic meta-analysis on
facial Action Units (AUs) in schizophrenia.

You have access to this FACS AU reference table:
{au_ref}

Given the paper text below, extract EXACTLY this JSON — no preamble, no fences:

{{
  "aus_studied": ["AU4", "AU12"],
  "findings": {{
    "AU4": {{
      "au_label":       "Brow Lowerer",
      "face_region":    "brow",
      "functional_tag": "anger/sadness/concentration",
      "direction":      "decreased",
      "feature_type":   "intensity",
      "mean_sz":        1.2,
      "sd_sz":          0.4,
      "mean_hc":        2.1,
      "sd_hc":          0.5,
      "stat_reported":  "t(48)=3.2, p=0.002",
      "notes":          ""
    }},
    "AU12": {{
      "au_label":       "Lip Corner Puller",
      "face_region":    "mouth",
      "functional_tag": "smile / positive affect",
      "direction":      "decreased",
      "feature_type":   "frequency",
      "mean_sz":        null,
      "sd_sz":          null,
      "mean_hc":        null,
      "sd_hc":          null,
      "stat_reported":  "p<0.05",
      "notes":          ""
    }}
  }},
  "n_sz": 30,
  "n_hc": 28,
  "target_measure": "negative symptoms",
  "apathy_scale":   "PANSS",
  "method":         "OpenFace",
  "study_design":   "cross-sectional"
}}

Rules:
- direction: "decreased" | "increased" | "no difference" | null
- feature_type: "intensity" | "frequency" | "duration" | "presence" | null
  (intensity = magnitude of movement; frequency = how often it occurs;
   duration = how long it lasts; presence = binary detected/not)
- au_label / face_region / functional_tag: fill from the reference table;
  if AU not in table, use your FACS knowledge
- stat_reported: copy the exact test statistic string if reported, else null
- mean_sz/sd_sz/mean_hc/sd_hc: numerical values only, null if not reported
- target_measure: "apathy" | "negative symptoms" | "both" | "other"
- apathy_scale: scale name (PANSS, BNSS, SANS, CAINS, etc.) or null
- study_design: "cross-sectional" | "longitudinal" | "RCT" | "other"
- If no schizophrenia-control AU comparison, return aus_studied: []
""").strip()


def extract_paper(paper: dict) -> dict:
    # Use full text if available, otherwise abstract
    if paper.get("full_text"):
        body = f"[FULL TEXT — Results/Methods section]\n{paper['full_text']}"
    else:
        body = f"[ABSTRACT ONLY]\nTitle: {paper['title']}\n\nAbstract: {paper['abstract']}"

    prompt = EXTRACT_PROMPT.format(au_ref=AU_REF_JSON)

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user",
                        "content": f"{prompt}\n\n---\n{body}"}]
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Build tables
# ═══════════════════════════════════════════════════════════════════════════

def hedges_g(mean1, sd1, n1, mean2, sd2, n2):
    try:
        n1,n2 = float(n1), float(n2)
        m1,m2 = float(mean1), float(mean2)
        s1,s2 = float(sd1), float(sd2)
        if any(math.isnan(v) or v<=0 for v in [n1,n2,s1,s2]):
            return None, None
        sp = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        if sp == 0: return None, None
        d  = (m1-m2)/sp
        J  = 1 - (3/(4*(n1+n2-2)-1))
        g  = J*d
        se = math.sqrt((n1+n2)/(n1*n2) + g**2/(2*(n1+n2-2)))
        return g, se
    except Exception:
        return None, None


def build_long_table(papers, extractions) -> pd.DataFrame:
    rows = []
    for paper, ext in zip(papers, extractions):
        if "error" in ext or not ext.get("aus_studied"):
            continue
        fa   = paper.get("first_author","")
        yr   = paper.get("year","?")
        label = f"{fa} {yr}" if fa else f"Paper {paper['id'][-6:]}"
        for au, info in (ext.get("findings") or {}).items():
            if not isinstance(info, dict):
                info = {"direction": info}
            g, se_g = hedges_g(
                info.get("mean_sz"), info.get("sd_sz"), ext.get("n_sz"),
                info.get("mean_hc"), info.get("sd_hc"), ext.get("n_hc")
            )
            rows.append({
                "id":             paper["id"],
                "label":          label,
                "year":           paper["year"],
                "source":         paper.get("source",""),
                "journal":        paper.get("journal",""),
                "full_text_used": bool(paper.get("full_text")),
                "au":             au.upper().strip(),
                # semantic fields
                "au_label":       info.get("au_label",""),
                "face_region":    info.get("face_region",""),
                "functional_tag": info.get("functional_tag",""),
                "feature_type":   info.get("feature_type",""),
                # effect
                "direction":      info.get("direction"),
                "mean_sz":        info.get("mean_sz"),
                "sd_sz":          info.get("sd_sz"),
                "mean_hc":        info.get("mean_hc"),
                "sd_hc":          info.get("sd_hc"),
                "n_sz":           ext.get("n_sz"),
                "n_hc":           ext.get("n_hc"),
                "hedges_g":       g,
                "se_g":           se_g,
                "stat_reported":  info.get("stat_reported",""),
                "target_measure": ext.get("target_measure"),
                "apathy_scale":   ext.get("apathy_scale"),
                "method":         ext.get("method"),
                "study_design":   ext.get("study_design"),
                "notes":          info.get("notes",""),
            })
    return pd.DataFrame(rows)


def build_feature_vocab(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Master table: one row per AU.
    Aggregates direction consensus, feature type, and evidence count.
    This is the table YOU review to decide which features go into the model.
    """
    if long_df.empty:
        return pd.DataFrame()

    rows = []
    for au, grp in long_df.groupby("au"):
        # dominant direction
        dir_counts  = grp["direction"].value_counts()
        dom_dir     = dir_counts.index[0] if not dir_counts.empty else None
        dir_agree   = dir_counts.iloc[0] / dir_counts.sum() if not dir_counts.empty else 0

        # feature types reported
        feat_types  = grp["feature_type"].dropna().value_counts().to_dict()

        # semantic info (take most common non-empty value)
        def most_common(col):
            vals = grp[col].dropna()
            vals = vals[vals != ""]
            return vals.mode().iloc[0] if not vals.empty else ""

        rows.append({
            "au":               au,
            "au_label":         most_common("au_label"),
            "face_region":      most_common("face_region"),
            "functional_tag":   most_common("functional_tag"),
            "n_studies":        grp["id"].nunique(),
            "dominant_direction": dom_dir,
            "direction_agreement": f"{dir_agree:.0%}",
            "feature_types":    ", ".join(f"{k}({v})" for k,v in feat_types.items()),
            "has_effect_size":  grp["hedges_g"].notna().sum(),
            "methods":          ", ".join(grp["method"].dropna().unique()),
            "apathy_scales":    ", ".join(grp["apathy_scale"].dropna().unique()),
            "recommended_feature": "",   # YOU fill this after review
            "include_in_model":    "",   # YOU fill: yes / no / maybe
        })

    df = pd.DataFrame(rows).sort_values("n_studies", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Random-effects meta-analysis
# ═══════════════════════════════════════════════════════════════════════════

def dersimonian_laird(gs, ses):
    gs, ses = np.array(gs,dtype=float), np.array(ses,dtype=float)
    ws   = 1/ses**2
    g_fe = np.sum(ws*gs)/np.sum(ws)
    Q    = np.sum(ws*(gs-g_fe)**2)
    k    = len(gs); dfq = k-1
    pQ   = 1-stats.chi2.cdf(Q,dfq)
    C    = np.sum(ws)-np.sum(ws**2)/np.sum(ws)
    tau2 = max(0.0,(Q-dfq)/C)
    ws_re= 1/(ses**2+tau2)
    g_re = np.sum(ws_re*gs)/np.sum(ws_re)
    se_re= math.sqrt(1/np.sum(ws_re))
    I2   = max(0.0,(Q-dfq)/Q*100) if Q>0 else 0.0
    return dict(pooled_g=g_re, se=se_re,
                ci_lo=g_re-1.96*se_re, ci_hi=g_re+1.96*se_re,
                I2=I2, Q=Q, p_Q=pQ, tau2=tau2, k=k)


def run_meta(long_df):
    results = []
    for au, grp in long_df.dropna(subset=["hedges_g","se_g"]).groupby("au"):
        if len(grp) < 2: continue
        res = dersimonian_laird(grp["hedges_g"].values, grp["se_g"].values)
        results.append({"au":au, "n_studies":res["k"],
                         "total_n_sz":grp["n_sz"].dropna().sum(), **res})
    return pd.DataFrame(results).sort_values("n_studies",ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Forest plots  (now with semantic labels in title)
# ═══════════════════════════════════════════════════════════════════════════

DIR_COLORS = {
    "decreased":     "#2166AC",
    "increased":     "#D73027",
    "no difference": "#888888",
    None:            "#AAAAAA",
}
SOURCE_MARKERS = {
    "PubMed":"s", "SemanticScholar":"o",
    "GoogleScholar":"D", "CrossRef":"^",
}


def forest_plot(au, grp, meta, vocab_df):
    es_rows  = grp.dropna(subset=["hedges_g","se_g"]).copy()
    dir_rows = grp[grp["hedges_g"].isna()].copy()
    n_total  = len(es_rows) + (1 if len(dir_rows)>0 else 0) + (2 if meta else 0)
    fig_h    = max(5, n_total*0.55+2.5)

    # get semantic label for title
    au_label = ""
    func_tag = ""
    if not vocab_df.empty:
        row = vocab_df[vocab_df["au"]==au]
        if not row.empty:
            au_label = row.iloc[0].get("au_label","")
            func_tag = row.iloc[0].get("functional_tag","")

    fig, axes = plt.subplots(1,2, figsize=(14,fig_h),
                              gridspec_kw={"width_ratios":[3,2.5],"wspace":0.02})
    ax_f, ax_t = axes
    y = 0

    for _, row in es_rows.iterrows():
        ci_lo = row["hedges_g"] - 1.96*row["se_g"]
        ci_hi = row["hedges_g"] + 1.96*row["se_g"]
        col   = DIR_COLORS.get(row["direction"],"#555")
        mrk   = SOURCE_MARKERS.get(row["source"],"s")
        ax_f.plot([ci_lo,ci_hi],[y,y], color=col, lw=1.6, zorder=2)
        try:    sz = max(25, min(220, float(row["n_sz"])*3))
        except: sz = 50
        ax_f.scatter([row["hedges_g"]],[y], s=sz, color=col, marker=mrk,
                      zorder=3, edgecolors="white", linewidths=0.5)
        ci_str = f"{row['hedges_g']:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]"
        # feature type tag
        ft_tag = f" [{row['feature_type']}]" if row.get("feature_type") else ""
        ax_t.text(0.01, y, row["label"]+ft_tag, va="center", ha="left", fontsize=7.5)
        ax_t.text(0.50, y, row.get("source",""), va="center", ha="center",
                  fontsize=6.5, color="#666")
        ax_t.text(0.63, y, str(int(row["n_sz"])) if pd.notna(row["n_sz"]) else "?",
                  va="center", ha="center", fontsize=8)
        ax_t.text(0.75, y, str(int(row["n_hc"])) if pd.notna(row["n_hc"]) else "?",
                  va="center", ha="center", fontsize=8)
        ax_t.text(0.99, y, ci_str, va="center", ha="right",
                  fontsize=7.5, color=col, fontweight="bold")
        y -= 1

    if len(dir_rows) > 0:
        y -= 0.3
        ax_f.axhline(y+0.65, color="#CCC", lw=0.6, ls="--")
        ax_t.text(0.01, y, "Direction only:", va="center", ha="left",
                  fontsize=7, style="italic", color="#666")
        y -= 0.7
        for _, row in dir_rows.iterrows():
            col = DIR_COLORS.get(row["direction"],"#AAA")
            ax_f.scatter([0],[y], s=25, color=col, marker="^", alpha=0.5, zorder=2)
            ft_tag = f" [{row['feature_type']}]" if row.get("feature_type") else ""
            ax_t.text(0.01, y, row["label"]+ft_tag, va="center", ha="left", fontsize=7.5)
            ax_t.text(0.50, y, row.get("source",""), va="center", ha="center",
                      fontsize=6.5, color="#666")
            ax_t.text(0.63, y, str(int(row["n_sz"])) if pd.notna(row["n_sz"]) else "?",
                      va="center", ha="center", fontsize=8)
            ax_t.text(0.75, y, str(int(row["n_hc"])) if pd.notna(row["n_hc"]) else "?",
                      va="center", ha="center", fontsize=8)
            ax_t.text(0.99, y, row["direction"] or "?",
                      va="center", ha="right", fontsize=8, color=col)
            y -= 1

    if meta and len(es_rows) >= 2:
        y -= 0.5
        ax_f.axhline(y+0.75, color="#444", lw=0.8)
        g,lo,hi = meta["pooled_g"], meta["ci_lo"], meta["ci_hi"]
        ax_f.fill([lo,g,hi,g,lo],[y,y+0.38,y,y-0.38,y],
                   color="#1A1A1A", zorder=4, alpha=0.85)
        ax_t.text(0.01, y, f"Pooled (k={meta['k']})",
                  va="center", ha="left", fontsize=9, fontweight="bold")
        ax_t.text(0.99, y, f"{g:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                  va="center", ha="right", fontsize=9, fontweight="bold")
        ax_t.text(0.01, y-0.65,
                  f"I²={meta['I2']:.0f}%  Q={meta['Q']:.1f}  p={meta['p_Q']:.3f}  τ²={meta['tau2']:.3f}",
                  va="center", ha="left", fontsize=7.5, color="#555", style="italic")
        y -= 1.6

    ax_f.axvline(0, color="black", lw=1.0)
    ax_f.set_xlabel("Hedges' g  (SZ < HC = negative = hypo-expression)", fontsize=9)

    # rich title with semantic info
    title_str = f"Forest Plot — {au}"
    if au_label:
        title_str += f"\n{au_label}"
        if func_tag:
            title_str += f"  ·  {func_tag}"
    ax_f.set_title(title_str, fontsize=11, fontweight="bold", pad=10)

    ax_f.set_ylim(y-0.5, 1); ax_f.set_yticks([])
    ax_f.spines[["top","right","left"]].set_visible(False)
    ax_f.xaxis.grid(True, ls="--", alpha=0.4); ax_f.set_axisbelow(True)

    ax_t.set_xlim(0,1); ax_t.set_ylim(y-0.5,1); ax_t.axis("off")
    for x, lbl in [(0.01,"Study [feature type]"),(0.50,"Source"),
                   (0.63,"n SZ"),(0.75,"n HC"),(0.99,"g [95% CI]")]:
        ax_t.text(x, 0.6, lbl, va="bottom",
                  ha="left" if x<0.5 else ("center" if x<0.8 else "right"),
                  fontsize=8, fontweight="bold", color="#222")
    ax_t.axhline(0.55, color="#444", lw=0.8)

    legend_handles = [
        mpatches.Patch(color=DIR_COLORS["decreased"],     label="Decreased in SZ"),
        mpatches.Patch(color=DIR_COLORS["increased"],     label="Increased in SZ"),
        mpatches.Patch(color=DIR_COLORS["no difference"], label="No difference"),
        Line2D([0],[0],marker="s",color="w",markerfacecolor="#333",markersize=8,label="Pooled"),
        Line2D([0],[0],marker="s",color="w",markerfacecolor="#888",markersize=7,label="PubMed"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor="#888",markersize=7,label="Semantic Scholar"),
        Line2D([0],[0],marker="D",color="w",markerfacecolor="#888",markersize=6,label="Google Scholar"),
        Line2D([0],[0],marker="^",color="w",markerfacecolor="#888",markersize=7,label="CrossRef"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5,0.0))
    plt.tight_layout(rect=[0,0.08,1,1])
    out = os.path.join(RESULTS_DIR, f"forest_{au.replace(' ','_')}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    {out}")


def make_all_forest_plots(long_df, meta_df, vocab_df):
    for au in long_df["au"].value_counts()[lambda x: x>=2].index:
        grp  = long_df[long_df["au"]==au]
        mrow = meta_df[meta_df["au"]==au]
        meta = mrow.iloc[0].to_dict() if not mrow.empty else None
        forest_plot(au, grp, meta, vocab_df)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Feature vocab heatmap  (face region × AU × direction)
# ═══════════════════════════════════════════════════════════════════════════

def vocab_heatmap(vocab_df: pd.DataFrame):
    if vocab_df.empty: return
    dir_map = {"decreased":-1, "no difference":0, "increased":1}
    vocab_df = vocab_df.copy()
    vocab_df["dir_num"] = vocab_df["dominant_direction"].map(dir_map)
    vocab_df["au_full"] = vocab_df["au"] + "\n" + vocab_df["au_label"].str[:18]

    # group by face region
    regions = ["brow","eye","nose","mouth"]
    sub = vocab_df[vocab_df["face_region"].isin(regions)].copy()
    sub["face_region"] = pd.Categorical(sub["face_region"], categories=regions, ordered=True)
    sub = sub.sort_values(["face_region","au"])

    fig, ax = plt.subplots(figsize=(12, max(4, len(sub)*0.55+1)))
    colors = []
    for _, row in sub.iterrows():
        d = row["dir_num"]
        if pd.isna(d):
            colors.append("#DDDDDD")
        elif d < 0:
            # intensity of blue proportional to n_studies
            alpha = min(1.0, row["n_studies"]/5)
            colors.append(plt.cm.Blues(0.4 + alpha*0.5))
        elif d > 0:
            alpha = min(1.0, row["n_studies"]/5)
            colors.append(plt.cm.Reds(0.4 + alpha*0.5))
        else:
            colors.append("#AAAAAA")

    bars = ax.barh(sub["au_full"], sub["n_studies"], color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"k={v}" for v in sub["n_studies"]], padding=4, fontsize=8)

    # region separators
    seen_regions = []
    for i, (_, row) in enumerate(sub.iterrows()):
        if row["face_region"] not in seen_regions:
            seen_regions.append(row["face_region"])
            if i > 0:
                ax.axhline(i-0.5, color="#999", lw=0.8, ls="--")
            ax.text(sub["n_studies"].max()*1.08, i,
                    row["face_region"].upper(),
                    va="center", ha="left", fontsize=8,
                    color="#444", fontweight="bold")

    # add functional tags as text
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(0.1, i, f"  {row['functional_tag']}", va="center", ha="left",
                fontsize=7, color="white" if row["dir_num"] else "#555",
                style="italic")

    ax.set_xlabel("Number of studies", fontsize=10)
    ax.set_title("AU Feature Vocabulary — Evidence by Face Region\n"
                 "(Blue = hypo-expression in SZ; Red = hyper-expression)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "vocab_heatmap.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Other summary charts
# ═══════════════════════════════════════════════════════════════════════════

def bubble_summary(long_df, meta_df):
    if meta_df.empty: return
    fig, ax = plt.subplots(figsize=(11,6))
    sc = ax.scatter(meta_df["pooled_g"], meta_df["n_studies"],
                    s=np.sqrt(meta_df["total_n_sz"].fillna(50))*10,
                    c=meta_df["I2"], cmap="RdYlGn_r", vmin=0, vmax=100,
                    alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("I² (%)", fontsize=9)
    ax.errorbar(meta_df["pooled_g"], meta_df["n_studies"],
                xerr=[meta_df["pooled_g"]-meta_df["ci_lo"],
                      meta_df["ci_hi"]-meta_df["pooled_g"]],
                fmt="none", color="#444", lw=1.0, capsize=3, zorder=2)
    for _, row in meta_df.iterrows():
        ax.annotate(row["au"],(row["pooled_g"],row["n_studies"]),
                    xytext=(6,4), textcoords="offset points", fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Pooled Hedges' g", fontsize=10)
    ax.set_ylabel("Number of studies", fontsize=10)
    ax.set_title("AU Evidence Summary (size ∝ N; colour = I²)", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "bubble_summary.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close(); print(f"    {out}")


def direction_heatmap(long_df):
    dir_map = {"decreased":-1,"no difference":0,"increased":1}
    sub = long_df[["label","au","direction"]].copy()
    sub["dir_num"] = sub["direction"].map(dir_map)
    pivot = sub.pivot_table(index="label", columns="au", values="dir_num", aggfunc="first")
    if pivot.empty or pivot.shape[1] < 2: return
    fig, ax = plt.subplots(figsize=(max(8,pivot.shape[1]*0.7), max(4,pivot.shape[0]*0.42)))
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Direction of AU Effects Across Studies", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1,0,1], fraction=0.03)
    cbar.ax.set_yticklabels(["Decreased","No diff","Increased"], fontsize=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "direction_heatmap.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close(); print(f"    {out}")


def source_breakdown(long_df):
    counts = long_df.groupby("source")["id"].nunique().sort_values(ascending=True)
    if counts.empty: return
    fig, ax = plt.subplots(figsize=(7,3.5))
    bars = ax.barh(counts.index, counts.values,
                   color=["#4878D0","#EE854A","#6ACC65","#D65F5F"][:len(counts)])
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Papers with AU data", fontsize=9)
    ax.set_title("Papers Retrieved by Source", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "source_breakdown.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close(); print(f"    {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. LLM narrative
# ═══════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = textwrap.dedent("""
You are writing the Results section of a systematic review on facial AUs in schizophrenia.

Data provided:
  A) Random-effects meta-analysis results (pooled Hedges g per AU)
  B) AU feature vocabulary table (au, label, face region, functional tag, direction, feature type)
  C) Direction-only evidence summary
  D) Source breakdown

Write ~600 words covering:
1. Overall hypo-expression pattern and which face regions are affected
2. Specific AUs with strongest evidence — name them with their anatomical label
   and functional meaning (e.g. "AU12, the Lip Corner Puller, reflects smiling")
3. Whether effects are in intensity, frequency, or both
4. AUs specifically linked to apathy vs. broader negative symptoms
5. Heterogeneity (I²) and what it might mean
6. Recommended feature set for ML, specifying feature type per AU

Meta-analysis: {meta_json}
Feature vocabulary: {vocab_json}
Direction-only: {dir_json}
Source breakdown: {src_json}
""").strip()


def narrative_summary(meta_df, long_df, vocab_df) -> str:
    dir_summary = (long_df[long_df["hedges_g"].isna()]
                   .groupby("au")["direction"].value_counts()
                   .reset_index().to_dict(orient="records"))
    src_summary = long_df.groupby("source")["id"].nunique().to_dict()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role":"user",
                   "content": SUMMARY_PROMPT.format(
                       meta_json  = meta_df.to_json(orient="records",indent=2),
                       vocab_json = vocab_df.to_json(orient="records",indent=2),
                       dir_json   = json.dumps(dir_summary,indent=2),
                       src_json   = json.dumps(src_summary,indent=2),
                   )}]
    )
    return resp.content[0].text


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Step 1: Retrieving papers ===")
    papers = collect_all_papers()

    print("\n=== Step 2: Full-text retrieval ===")
    papers = enrich_with_full_text(papers)

    print("\n=== Step 3: Extracting AU data with Claude ===")
    extractions = []
    for p in tqdm(papers):
        extractions.append(extract_paper(p))
        time.sleep(0.3)

    with open(f"{RESULTS_DIR}/raw_extractions.json","w") as f:
        json.dump([{**p,"extraction":e} for p,e in zip(papers,extractions)], f, indent=2)

    print("\n=== Step 4: Building tables ===")
    long_df = build_long_table(papers, extractions)
    if long_df.empty:
        print("  No AU data extracted. Check raw_extractions.json"); exit(1)
    long_df.to_csv(f"{RESULTS_DIR}/au_long.csv", index=False)
    print(f"  {len(long_df)} AU observations, {long_df['hedges_g'].notna().sum()} with effect sizes")
    print(f"  Full text used in {long_df['full_text_used'].sum()} rows")

    vocab_df = build_feature_vocab(long_df)
    vocab_df.to_csv(f"{RESULTS_DIR}/au_feature_vocab.csv", index=False)
    print(f"\n  AU Feature Vocabulary ({len(vocab_df)} AUs):")
    print(vocab_df[["au","au_label","face_region","functional_tag",
                     "dominant_direction","feature_types","n_studies"]].to_string(index=False))

    print("\n=== Step 5: Meta-analyses ===")
    meta_df = run_meta(long_df)
    meta_df.to_csv(f"{RESULTS_DIR}/meta_results.csv", index=False)
    if not meta_df.empty:
        print(meta_df[["au","n_studies","pooled_g","ci_lo","ci_hi","I2","p_Q"]].to_string(index=False))

    print("\n=== Step 6: Forest plots ===")
    make_all_forest_plots(long_df, meta_df, vocab_df)

    print("\n=== Step 7: Summary charts ===")
    bubble_summary(long_df, meta_df)
    direction_heatmap(long_df)
    source_breakdown(long_df)
    vocab_heatmap(vocab_df)

    print("\n=== Step 8: LLM narrative ===")
    narrative = narrative_summary(meta_df, long_df, vocab_df)
    print(narrative[:500]+"...\n")
    with open(f"{RESULTS_DIR}/meta_analysis_summary.txt","w") as f:
        f.write(narrative)

    print(f"\nDone. Outputs in ./{RESULTS_DIR}/")
    print("""
NEXT STEPS:
  1. Open results/meta/au_feature_vocab.csv
  2. Review each AU row — check au_label, functional_tag, feature_type
  3. Fill in 'recommended_feature' and 'include_in_model' columns
  4. Update AU_INTENSITIES list in 02_feature_extraction.py accordingly
  5. For better forest plots: open au_long.csv and add mean_sz/sd_sz/mean_hc/sd_hc
     from full texts you read manually, then re-run from Step 5

EZPROXY SETUP (for institutional access):
  export EZPROXY_BASE="https://ezproxy.youruniversity.edu/login?url="
  export EZPROXY_COOKIE="your_session_cookie_from_browser"
  (get cookie: DevTools → Application → Cookies after logging into EZproxy)
""")
