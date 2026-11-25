"""
01_meta_analysis.py  (v4 — 8 free sources, deep retrieval)
-----------------------------------------------------------
Zero credentials needed. All sources are freely accessible.

Retrieval stack:
  1. PubMed / PMC       — NCBI Entrez + Europe PMC full-text XML
  2. Europe PMC         — full text for PMC-deposited papers
  3. Semantic Scholar   — abstracts + open-access PDF links
  4. OpenAlex           — 200M+ works, largest open scholarly graph
  5. CORE.ac.uk         — 200M+ open-access full texts (free key optional)
  6. Unpaywall          — best OA PDF finder, no key needed
  7. CrossRef           — DOI metadata + abstracts
  8. Google Scholar     — broadest coverage (via scholarly, rate-limited)
  9. bioRxiv/medRxiv    — preprints
 10. Direct PDF fetch   — publisher landing page citation_pdf_url meta tag
 11. Sci-Hub            — last resort (user assumes legal responsibility)

LLM: defaults to Anthropic claude-sonnet-4-6.
     Swap via LLM_BACKEND / LLM_BASE_URL / LLM_MODEL env vars.

Requirements:
  pip install anthropic openai biopython requests scholarly
              pandas numpy matplotlib scipy tqdm pymupdf
  Optional: CORE_API_KEY (free at core.ac.uk/api-keys)
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
# gemini-1.5-flash:  1500 req/day,  15 req/min,  1M tokens/min
# gemini-1.5-pro:     50 req/day,    2 req/min   (use only for narrative summary)
GEMINI_RPM        = int(os.environ.get("GEMINI_RPM", "14"))   # stay under 15
GEMINI_RPD        = int(os.environ.get("GEMINI_RPD", "1490"))  # stay under 1500
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
    prefer_pro=True → uses gemini-2.5-pro for final narrative (lower quota).
    """
    import time as _t

    if LLM_BACKEND == "gemini" or (not LLM_BASE_URL and GEMINI_API_KEY):
        from google import genai as _genai
        from google.genai import types as _gtypes
        _client = _genai.Client(api_key=GEMINI_API_KEY)
        model_name = "gemini-3.5-flash-lite" if prefer_pro else LLM_MODEL
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

AU_REFERENCE = {
    "AU1":  {"label":"Inner Brow Raise",       "muscle":"frontalis (medial)",    "face_region":"brow",  "functional_tag":"sadness/concern"},
    "AU2":  {"label":"Outer Brow Raise",        "muscle":"frontalis (lateral)",   "face_region":"brow",  "functional_tag":"surprise/attention"},
    "AU4":  {"label":"Brow Lowerer",            "muscle":"corrugator supercilii", "face_region":"brow",  "functional_tag":"anger/sadness/concentration"},
    "AU5":  {"label":"Upper Lid Raiser",        "muscle":"levator palpebrae",     "face_region":"eye",   "functional_tag":"surprise/fear"},
    "AU6":  {"label":"Cheek Raiser",            "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"enjoyment/Duchenne smile component"},
    "AU7":  {"label":"Lid Tightener",           "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"anger/disgust"},
    "AU9":  {"label":"Nose Wrinkler",           "muscle":"levator labii",         "face_region":"nose",  "functional_tag":"disgust"},
    "AU10": {"label":"Upper Lip Raiser",        "muscle":"levator labii",         "face_region":"mouth", "functional_tag":"disgust/contempt"},
    "AU12": {"label":"Lip Corner Puller",       "muscle":"zygomaticus major",     "face_region":"mouth", "functional_tag":"smile / positive affect"},
    "AU14": {"label":"Dimpler",                 "muscle":"buccinator",            "face_region":"mouth", "functional_tag":"amusement/subtle smile"},
    "AU15": {"label":"Lip Corner Depressor",    "muscle":"depressor anguli oris", "face_region":"mouth", "functional_tag":"sadness/disgust"},
    "AU17": {"label":"Chin Raiser",             "muscle":"mentalis",              "face_region":"mouth", "functional_tag":"doubt/sadness"},
    "AU20": {"label":"Lip Stretcher",           "muscle":"risorius",              "face_region":"mouth", "functional_tag":"fear"},
    "AU23": {"label":"Lip Tightener",           "muscle":"orbicularis oris",      "face_region":"mouth", "functional_tag":"anger"},
    "AU24": {"label":"Lip Pressor",             "muscle":"orbicularis oris",      "face_region":"mouth", "functional_tag":"anger/determination"},
    "AU25": {"label":"Lips Part",               "muscle":"depressor labii",       "face_region":"mouth", "functional_tag":"speech/surprise"},
    "AU26": {"label":"Jaw Drop",                "muscle":"masseter relaxation",   "face_region":"mouth", "functional_tag":"surprise/open mouth"},
    "AU43": {"label":"Eyes Closed",             "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"sleep/relaxation/avoidance"},
    "AU45": {"label":"Blink",                   "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"blink rate (attention/dryness)"},
    "AU46": {"label":"Wink",                    "muscle":"orbicularis oculi",     "face_region":"eye",   "functional_tag":"deliberate social signal"},
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

# ═══════════════════════════════════════════════════════════════════════════
# 1. RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def search_pubmed(query, n=20):
    # Always use the precise MeSH query for PubMed regardless of query param
    pubmed_term = PUBMED_QUERY
    try:
        h=Entrez.esearch(db="pubmed",term=pubmed_term,retmax=n,sort="relevance")
        ids=Entrez.read(h)["IdList"]; h.close()
        if not ids: return []
        h=Entrez.efetch(db="pubmed",id=ids,rettype="abstract",retmode="xml")
        recs=Entrez.read(h)["PubmedArticle"]; h.close()
    except Exception as e: print(f"    PubMed: {e}"); return []
    out=[]
    for rec in recs:
        try:
            art=rec["MedlineCitation"]["Article"]
            ttl=str(art.get("ArticleTitle",""))
            abst=" ".join(str(p) for p in art.get("Abstract",{}).get("AbstractText",[]))
            year=str(art.get("Journal",{}).get("JournalIssue",{}).get("PubDate",{}).get("Year","?"))
            aus=art.get("AuthorList",[]); fa=str(aus[0].get("LastName","")) if aus else ""
            pmid=str(rec["MedlineCitation"]["PMID"])
            jnl=str(art.get("Journal",{}).get("Title",""))
            ids2=rec.get("PubmedData",{}).get("ArticleIdList",[])
            doi=next((str(x) for x in ids2 if x.attributes.get("IdType")=="doi"),"")
            if abst.strip(): out.append(_paper(f"pmid_{pmid}","PubMed",fa,year,ttl,abst,jnl,doi))
        except Exception: continue
    return out


def search_europe_pmc(query, n=20):
    r=_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
           params={"query":query,"resultType":"core","pageSize":n,"format":"json"})
    if not r or r.status_code!=200: return []
    out=[]
    for item in r.json().get("resultList",{}).get("result",[]):
        abst=item.get("abstractText","") or ""; ttl=item.get("title","") or ""
        if not abst.strip(): continue
        pmid=item.get("pmid",""); pmcid=item.get("pmcid",""); doi=item.get("doi","") or ""
        aus=item.get("authorList",{}).get("author",[]); fa=aus[0].get("lastName","") if aus else ""
        p=_paper(f"epmc_{pmid or pmcid or _fp(ttl)[:8]}","EuropePMC",fa,
                  str(item.get("pubYear","?")),ttl,abst,item.get("journalTitle",""),doi)
        p["pmcid"]=pmcid; out.append(p)
    return out


def fetch_europe_pmc_fulltext(pmcid):
    if not pmcid: return None
    r=_get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",timeout=20)
    if not r or r.status_code!=200: return None
    text=re.sub(r"<[^>]+>"," ",r.text); text=re.sub(r"\s+"," ",text).strip()
    return extract_results_section(text)[:8000]


def search_semantic_scholar(query, n=20):
    r=_get("https://api.semanticscholar.org/graph/v1/paper/search",
           params={"query":query,"limit":n,
                   "fields":"title,abstract,year,authors,externalIds,venue,openAccessPdf"})
    if not r or r.status_code!=200: return []
    out=[]
    for p in r.json().get("data",[]):
        abst=p.get("abstract") or ""
        if not abst.strip(): continue
        aus=p.get("authors",[]); fa=aus[0].get("name","").split()[-1] if aus else ""
        doi=(p.get("externalIds") or {}).get("DOI","")
        oa=(p.get("openAccessPdf") or {}).get("url","")
        pp=_paper(f"ss_{p.get('paperId','')}","SemanticScholar",
                   fa,p.get("year","?"),p.get("title",""),abst,p.get("venue",""),doi)
        pp["oa_pdf_url"]=oa; out.append(pp)
    return out


def search_openalex(query, n=25):
    r=_get("https://api.openalex.org/works",
           params={"search":query,"per-page":n,"filter":"has_abstract:true",
                   "mailto":Entrez.email})
    if not r or r.status_code!=200: return []
    out=[]
    for w in r.json().get("results",[]):
        inv=w.get("abstract_inverted_index") or {}
        if not inv: continue
        words=[""]*(max(max(v) for v in inv.values())+1)
        for word,positions in inv.items():
            for pos in positions: words[pos]=word
        abst=" ".join(words).strip(); ttl=(w.get("title") or "").strip()
        if not abst or not ttl: continue
        aus=w.get("authorships",[]); fa=""
        if aus:
            name=(aus[0].get("author") or {}).get("display_name","")
            fa=name.split()[-1] if name else ""
        doi=(w.get("doi") or "").replace("https://doi.org/","")
        year=str(w.get("publication_year","?"))
        jnl=(w.get("primary_location") or {}).get("source",{})
        jnl=(jnl or {}).get("display_name","") if jnl else ""
        oa=((w.get("primary_location") or {}).get("pdf_url")) or            ((w.get("open_access") or {}).get("oa_url"))
        p=_paper(f"oa_{w.get('id','').split('/')[-1]}","OpenAlex",fa,year,ttl,abst,jnl,doi)
        p["oa_pdf_url"]=oa or ""; out.append(p)
    return out


def search_core(query, n=20):
    headers={"Authorization":f"Bearer {CORE_API_KEY}"} if CORE_API_KEY else {}
    r=_get("https://api.core.ac.uk/v3/search/works",
           params={"q":query,"limit":n},headers=headers,timeout=20)
    if not r or r.status_code not in (200,429): return []
    out=[]
    for w in r.json().get("results",[]):
        abst=w.get("abstract","") or ""; ttl=w.get("title","") or ""
        if not abst.strip(): continue
        doi=w.get("doi","") or ""; year=str(w.get("yearPublished","?"))
        aus=w.get("authors",[]); fa=(aus[0].get("name","")).split()[-1] if aus else ""
        pdf=w.get("downloadUrl","") or ""
        p=_paper(f"core_{w.get('id',_fp(ttl)[:8])}","CORE",fa,year,ttl,abst,w.get("publisher",""),doi)
        p["oa_pdf_url"]=pdf; out.append(p)
    return out


def unpaywall_pdf_url(doi):
    if not doi: return None
    r=_get(f"https://api.unpaywall.org/v2/{doi}",params={"email":Entrez.email})
    if not r or r.status_code!=200: return None
    loc=r.json().get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")


def search_crossref(query, n=20):
    r=_get("https://api.crossref.org/works",
           params={"query":query,"rows":n,
                   "select":"DOI,title,abstract,author,published,container-title",
                   "mailto":Entrez.email})
    if not r or r.status_code!=200: return []
    out=[]
    for item in r.json().get("message",{}).get("items",[]):
        abst=re.sub(r"<[^>]+>"," ",item.get("abstract","") or "").strip()
        ttls=item.get("title",[]); ttl=ttls[0] if ttls else ""
        if not abst or len(abst)<80: continue
        aus=item.get("author",[]); fa=aus[0].get("family","") if aus else ""
        pub=item.get("published",{}).get("date-parts",[[None]])[0]
        year=str(pub[0]) if pub and pub[0] else "?"
        doi=item.get("DOI",""); jnl=(item.get("container-title") or [""])[0]
        out.append(_paper(f"cr_{doi.replace('/','_')}","CrossRef",fa,year,ttl,abst,jnl,doi))
    return out


def search_google_scholar(query, n=10):
    try:
        from scholarly import scholarly as sch
        results=sch.search_pubs(query)
    except Exception as e: print(f"    GS: {e}"); return []
    out=[]
    for i,pub in enumerate(results):
        if i>=n: break
        try:
            bib=pub.get("bib",{}); abst=bib.get("abstract","") or ""; ttl=bib.get("title","") or ""
            if not abst.strip() or not ttl.strip(): continue
            fa=str(bib.get("author","")).split(" and ")[0].split(",")[0].strip()
            year=str(bib.get("pub_year","?"))
            p=_paper(f"gs_{_fp(ttl)[:10]}","GoogleScholar",fa,year,ttl,abst,bib.get("venue",""),"")
            p["oa_pdf_url"]=pub.get("eprint_url","") or ""; out.append(p); time.sleep(2.0)
        except Exception: continue
    return out


def search_rxiv(query, n=15):
    out=[]
    for server in ("biorxiv","medrxiv"):
        for term in ["schizophrenia action unit","schizophrenia facial expression apathy"]:
            r=_get(f"https://api.biorxiv.org/details/{server}/{term}/0/{n}/json",timeout=15)
            if not r or r.status_code!=200: continue
            for item in r.json().get("collection",[]):
                abst=item.get("abstract","") or ""; ttl=item.get("title","") or ""
                if not abst.strip(): continue
                doi=item.get("doi",""); fa=item.get("authors","").split(";")[0].split(",")[0].strip()
                year=str(item.get("date","?"))[:4]
                pdf=f"https://www.{server}.org/content/{doi}.full.pdf" if doi else ""
                p=_paper(f"rxiv_{_fp(ttl)[:10]}",server.capitalize(),fa,year,ttl,abst,server,doi)
                p["oa_pdf_url"]=pdf; out.append(p)
            time.sleep(0.5)
    return out


def collect_all_papers():
    seen,papers=[],[]
    seen_set=set()
    source_fns=[
        ("PubMed",           search_pubmed,          20),
        ("EuropePMC",        search_europe_pmc,       20),
        ("SemanticScholar",  search_semantic_scholar, 20),
        ("OpenAlex",         search_openalex,         25),
        ("CORE",             search_core,             20),
        ("CrossRef",         search_crossref,         20),
        ("bioRxiv/medRxiv",  search_rxiv,             15),
        ("GoogleScholar",    search_google_scholar,   10),
    ]
    for name,fn,n in source_fns:
        print(f"\n  [{name}]")
        # PubMed uses one precise MeSH query — no need to loop all queries
        query_list = ["pubmed_mesh"] if name == "PubMed" else QUERIES
        for q in query_list:
            print(f"    -> {q[:65]}")
            try: res=fn(q,n)
            except TypeError:
                try: res=fn(q)
                except Exception as e: print(f"      error: {e}"); res=[]
            except Exception as e: print(f"      error: {e}"); res=[]
            new=0
            for p in res:
                fp=_fp(p["title"])
                if fp not in seen_set: seen_set.add(fp); papers.append(p); new+=1
            print(f"      +{new}  (total {len(papers)})")
            time.sleep(0.5)
    print(f"\n  Total unique papers: {len(papers)}")
    return papers[:MAX_PAPERS]

# ═══════════════════════════════════════════════════════════════════════════
# 1b. RELEVANCE RANKING — score and keep top-K before extraction
# ═══════════════════════════════════════════════════════════════════════════

# AU codes and schizophrenia-specific terms
_AU_PAT   = re.compile(r"\bAU\s*\d{1,2}\b", re.I)
_TOOL_PAT = re.compile(
    r"\b(OpenFace|FACS|facial action coding|action unit|AU intensity|"
    r"AU frequency|blunted affect|flat affect|expressivity|facial movement)\b", re.I)
_SZ_PAT   = re.compile(
    r"\b(schizophrenia|schizophrenic|psychosis|psychotic|schizoaffective|"
    r"negative symptom|apathy|anhedonia)\b", re.I)
_QUANT_PAT = re.compile(
    r"\b(mean|SD|standard deviation|p\s*[=<]|t\(|F\(|effect size|"
    r"Cohen|Hedges|significant|regression|correlation)\b", re.I)


def rank_papers(papers: list[dict]) -> list[dict]:
    """
    Score each paper by relevance to AU + schizophrenia research.

    HARD REQUIREMENT: paper must have BOTH:
      - at least 1 AU signal (AU code OR tool name)
      - at least 1 schizophrenia term
    Papers missing either are scored 0 and excluded.

    Scoring (only applied when both conditions met):
      +5 per AU code mention  (AU4, AU12 etc.)
      +3 per tool/method term (OpenFace, FACS etc.)
      +2 per schizophrenia term
      +1 per quantitative term (mean, SD, p=)
      +10 bonus if AU code in title
      +10 bonus if tool name in title
      +5  bonus if schizophrenia term in title
    """
    for p in papers:
        text  = f"{p.get('title','')} {p.get('abstract','')}"
        title = p.get("title", "")

        au_hits   = len(_AU_PAT.findall(text))
        tool_hits = len(_TOOL_PAT.findall(text))
        sz_hits   = len(_SZ_PAT.findall(text))
        q_hits    = len(_QUANT_PAT.findall(text))

        # Hard gate: MUST have both AU signal AND schizophrenia term
        has_au = (au_hits > 0 or tool_hits > 0)
        has_sz = (sz_hits > 0)

        if not has_au or not has_sz:
            p["_relevance_score"] = 0
            p["_excluded"]        = True
            continue

        score = (au_hits * 5 + tool_hits * 3 + sz_hits * 2 + q_hits * 1)

        # Title bonuses — title mention = paper is specifically about this
        if _AU_PAT.search(title):   score += 10
        if _TOOL_PAT.search(title): score += 10
        if _SZ_PAT.search(title):   score += 5

        p["_relevance_score"] = score
        p["_excluded"]        = False

    # Only keep papers that passed the hard gate
    eligible = [p for p in papers if not p.get("_excluded", False)]
    excluded = len(papers) - len(eligible)
    ranked   = sorted(eligible, key=lambda p: p["_relevance_score"], reverse=True)

    print(f"\n=== Relevance ranking ===")
    print(f"  {excluded}/{len(papers)} papers excluded (missing AU or SZ signal)")
    print(f"  {len(eligible)} eligible → keeping top {min(TOP_K, len(eligible))}")
    print(f"\n  {'Score':>6}  {'AU':>3}  {'SZ':>3}  Title")
    print(f"  {'─'*6}  {'─'*3}  {'─'*3}  {'─'*65}")
    for p in ranked[:TOP_K]:
        au_n = len(_AU_PAT.findall(f"{p['title']} {p['abstract']}"))
        sz_n = len(_SZ_PAT.findall(f"{p['title']} {p['abstract']}"))
        print(f"  {p['_relevance_score']:>6.0f}  {au_n:>3}  {sz_n:>3}  {p['title'][:65]}")

    if len(ranked) > TOP_K:
        print(f"  ... {len(ranked)-TOP_K} eligible papers below cutoff")

    return ranked[:TOP_K]




# ═══════════════════════════════════════════════════════════════════════════
# 2. FULL-TEXT RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

_PDF_META_RE=[
    re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url',re.I),
    re.compile(r'"pdfUrl"\s*:\s*"([^"]+)"',re.I),
]

def extract_results_section(text):
    for pat in [r"(?i)\bresults?\b(.{200,}?)(?=\b(?:discussion|conclusion|references)\b)",
                r"(?i)\bmethods?\b(.{200,}?)(?=\b(?:results?|discussion)\b)"]:
        m=re.search(pat,text,re.DOTALL)
        if m: return m.group(0)[:8000]
    return text[:8000]

def pdf_bytes_to_text(pdf_bytes):
    try:
        import pymupdf as fitz
        doc=fitz.open(stream=pdf_bytes,filetype="pdf")
        text="\n".join(page.get_text() for page in doc); doc.close()
        return extract_results_section(text) or None
    except Exception: return None

def fetch_pdf(url):
    if not url: return None
    r=_get(url,timeout=30,stream=True)
    if not r or r.status_code!=200: return None
    ct=r.headers.get("Content-Type","")
    if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"): return None
    return pdf_bytes_to_text(r.content)

def fetch_publisher_pdf(doi):
    if not doi: return None
    r=_get(f"https://doi.org/{doi}",timeout=15,allow_redirects=True)
    if not r or r.status_code!=200: return None
    if "pdf" in r.headers.get("Content-Type","").lower(): return pdf_bytes_to_text(r.content)
    for pat in _PDF_META_RE:
        m=pat.search(r.text)
        if m:
            pdf_url=requests.compat.urljoin(r.url,m.group(1).replace("&amp;","&"))
            text=fetch_pdf(pdf_url)
            if text: return text
    return None

def fetch_scihub(doi):
    if not doi: return None
    for mirror in SCIHUB_MIRRORS:
        try:
            r=_get(f"{mirror}/{doi}",timeout=20,allow_redirects=True)
            if not r or r.status_code!=200: continue
            m=re.search(r'src=["\']([^"\']*\.pdf[^"\']*)["\']',r.text,re.I)
            if not m: m=re.search(r'src=["\']//([^"\']+)["\']',r.text,re.I)
            if m:
                pdf_url=m.group(1)
                if pdf_url.startswith("//"): pdf_url="https:"+pdf_url
                text=fetch_pdf(pdf_url)
                if text: return text
        except Exception: continue
    return None

def enrich_with_full_text(papers):
    print("\n  Attempting full-text retrieval...")
    for p in tqdm(papers):
        doi=p.get("doi","") or ""; pmcid=p.get("pmcid","") or ""; text=None
        if not text and pmcid:
            text=fetch_europe_pmc_fulltext(pmcid)
            if text: p["full_text_source"]="europe_pmc_xml"
        oa_url=p.get("oa_pdf_url","") or ""
        if not text and oa_url:
            text=fetch_pdf(oa_url)
            if text: p["full_text_source"]="oa_pdf"
        if not text and doi:
            url=unpaywall_pdf_url(doi)
            if url: text=fetch_pdf(url)
            if text: p["full_text_source"]="unpaywall"
        if not text and doi:
            text=fetch_publisher_pdf(doi)
            if text: p["full_text_source"]="publisher"
        if not text and doi:
            text=fetch_scihub(doi)
            if text: p["full_text_source"]="scihub"
        p["full_text"]=text; time.sleep(0.3)
    n_ft=sum(1 for p in papers if p.get("full_text"))
    print(f"    Full text: {n_ft}/{len(papers)}")
    src_counts={}
    for p in papers:
        s=p.get("full_text_source")
        if s: src_counts[s]=src_counts.get(s,0)+1
    for s,c in sorted(src_counts.items(),key=lambda x:-x[1]):
        print(f"      {s}: {c}")
    return papers


# ═══════════════════════════════════════════════════════════════════════════
# 3. LLM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = textwrap.dedent("""
You are extracting AU findings from a clinical paper for a systematic meta-analysis.

FACS AU reference:
{au_ref}

The text below uses paragraph tags like [p3:2] meaning page 3, paragraph 2.
For EVERY finding you extract, you MUST record:
  - exact_quote: the verbatim sentence(s) you used
  - source_page / source_para: from the [pPAGE:PARA] tag of that paragraph

Return EXACTLY this JSON — no preamble, no fences:
{{
  "aus_studied": ["AU4","AU12"],
  "findings": {{
    "AU4": {{
      "au_label":      "Brow Lowerer",
      "face_region":   "brow",
      "functional_tag":"anger/sadness/concentration",
      "direction":     "decreased",
      "feature_type":  "intensity",
      "mean_sz":       1.2,
      "sd_sz":         0.4,
      "mean_hc":       2.1,
      "sd_hc":         0.5,
      "stat_reported": "t(48)=3.2, p=0.002",
      "exact_quote":   "AU4 intensity was significantly lower in SZ (M=1.2, SD=0.4) vs HC (M=2.1, SD=0.5), t(48)=3.2, p=0.002.",
      "source_page":   3,
      "source_para":   2,
      "notes":         ""
    }},
    "AU12": {{
      "au_label":      "Lip Corner Puller",
      "face_region":   "mouth",
      "functional_tag":"smile / positive affect",
      "direction":     "decreased",
      "feature_type":  "frequency",
      "mean_sz":       null, "sd_sz": null, "mean_hc": null, "sd_hc": null,
      "stat_reported": "p<0.05",
      "exact_quote":   "The frequency of AU12 was significantly reduced in patients.",
      "source_page":   4,
      "source_para":   1,
      "notes":         ""
    }}
  }},
  "n_sz": 30, "n_hc": 28,
  "target_measure": "negative symptoms",
  "apathy_scale":   "PANSS",
  "method":         "OpenFace",
  "study_design":   "cross-sectional"
}}

Rules:
- exact_quote: copy verbatim from the [pPAGE:PARA] tagged text — never paraphrase
- source_page/source_para: integers from the tag of the paragraph containing the quote
- direction: "decreased" | "increased" | "no difference" | null
- feature_type: "intensity" | "frequency" | "duration" | "presence" | null
- mean_sz/sd_sz/mean_hc/sd_hc: numbers only, null if not reported
- target_measure: "apathy" | "negative symptoms" | "both" | "other"
- apathy_scale: PANSS / BNSS / SANS / CAINS / etc., or null
- study_design: "cross-sectional" | "longitudinal" | "RCT" | "other"
- If no SZ-control AU comparison: return aus_studied: []
""").strip()

def extract_paper(paper):
    body=(f"[FULL TEXT]\n{paper['full_text']}" if paper.get("full_text")
          else f"[ABSTRACT]\nTitle: {paper['title']}\n\n{paper['abstract']}")
    prompt=f"{EXTRACT_PROMPT.format(au_ref=AU_REF_JSON)}\n\n---\n{body}"
    try:
        raw=llm_chat(prompt,max_tokens=1200)
        raw=re.sub(r"^```(?:json)?\s*|```$","",raw,flags=re.MULTILINE).strip()
        m=re.search(r"\{.*\}",raw,re.DOTALL)
        return json.loads(m.group(0) if m else raw)
    except Exception as e: return {"error":str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Build tables
# ═══════════════════════════════════════════════════════════════════════════

def hedges_g(m1,s1,n1,m2,s2,n2):
    try:
        n1,n2=float(n1),float(n2); m1,m2=float(m1),float(m2); s1,s2=float(s1),float(s2)
        if any(math.isnan(v) or v<=0 for v in [n1,n2,s1,s2]): return None,None
        sp=math.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
        if sp==0: return None,None
        d=(m1-m2)/sp; J=1-(3/(4*(n1+n2-2)-1)); g=J*d
        se=math.sqrt((n1+n2)/(n1*n2)+g**2/(2*(n1+n2-2)))
        return g,se
    except Exception: return None,None

def build_long_table(papers,extractions):
    rows=[]
    for paper,ext in zip(papers,extractions):
        if "error" in ext or not ext.get("aus_studied"): continue
        fa=paper.get("first_author",""); yr=paper.get("year","?")
        label=f"{fa} {yr}" if fa else f"Paper {paper['id'][-6:]}"
        for au,info in (ext.get("findings") or {}).items():
            if not isinstance(info,dict): info={"direction":info}
            g,se=hedges_g(info.get("mean_sz"),info.get("sd_sz"),ext.get("n_sz"),
                          info.get("mean_hc"),info.get("sd_hc"),ext.get("n_hc"))
            doi = paper.get("doi","") or ""
            rows.append({
                "id":label,"paper_id":paper["id"],"year":paper["year"],
                "source":paper.get("source",""),"journal":paper.get("journal",""),
                "full_text_used":bool(paper.get("full_text")),
                "full_text_source":paper.get("full_text_source",""),
                "au":au.upper().strip(),
                "au_label":info.get("au_label",""),"face_region":info.get("face_region",""),
                "functional_tag":info.get("functional_tag",""),"feature_type":info.get("feature_type",""),
                "direction":info.get("direction"),
                "mean_sz":info.get("mean_sz"),"sd_sz":info.get("sd_sz"),
                "mean_hc":info.get("mean_hc"),"sd_hc":info.get("sd_hc"),
                "n_sz":ext.get("n_sz"),"n_hc":ext.get("n_hc"),
                "hedges_g":g,"se_g":se,
                "stat_reported":info.get("stat_reported",""),
                "exact_quote":info.get("exact_quote",""),
                "source_page":info.get("source_page"),
                "source_para":info.get("source_para"),
                "doi":doi,
                "doi_link":f"https://doi.org/{doi}" if doi else "",
                "pubmed_link":f"https://pubmed.ncbi.nlm.nih.gov/{paper['id'].replace('pmid_','')}" if paper["id"].startswith("pmid_") else "",
                "target_measure":ext.get("target_measure"),"apathy_scale":ext.get("apathy_scale"),
                "method":ext.get("method"),"study_design":ext.get("study_design"),
                "notes":info.get("notes",""),
            })
    return pd.DataFrame(rows)

def build_feature_vocab(long_df):
    if long_df.empty: return pd.DataFrame()
    rows=[]
    for au,grp in long_df.groupby("au"):
        dc=grp["direction"].value_counts()
        dom=dc.index[0] if not dc.empty else None
        agr=dc.iloc[0]/dc.sum() if not dc.empty else 0
        ft=grp["feature_type"].dropna().value_counts().to_dict()
        def mc(col):
            v=grp[col].dropna(); v=v[v!=""]; return v.mode().iloc[0] if not v.empty else ""
        rows.append({
            "au":au,"au_label":mc("au_label"),"face_region":mc("face_region"),
            "functional_tag":mc("functional_tag"),"n_studies":grp["paper_id"].nunique(),
            "dominant_direction":dom,"direction_agreement":f"{agr:.0%}",
            "feature_types":", ".join(f"{k}({v})" for k,v in ft.items()),
            "has_effect_size":grp["hedges_g"].notna().sum(),
            "methods":", ".join(grp["method"].dropna().unique()),
            "apathy_scales":", ".join(grp["apathy_scale"].dropna().unique()),
            "recommended_feature":"","include_in_model":"",
        })
    return pd.DataFrame(rows).sort_values("n_studies",ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Random-effects meta-analysis
# ═══════════════════════════════════════════════════════════════════════════

def dersimonian_laird(gs,ses):
    gs,ses=np.array(gs,dtype=float),np.array(ses,dtype=float)
    ws=1/ses**2; g_fe=np.sum(ws*gs)/np.sum(ws); Q=np.sum(ws*(gs-g_fe)**2)
    k=len(gs); dfq=k-1; pQ=1-stats.chi2.cdf(Q,dfq)
    C=np.sum(ws)-np.sum(ws**2)/np.sum(ws); tau2=max(0.0,(Q-dfq)/C)
    ws_re=1/(ses**2+tau2); g_re=np.sum(ws_re*gs)/np.sum(ws_re)
    se_re=math.sqrt(1/np.sum(ws_re))
    I2=max(0.0,(Q-dfq)/Q*100) if Q>0 else 0.0
    return dict(pooled_g=g_re,se=se_re,ci_lo=g_re-1.96*se_re,ci_hi=g_re+1.96*se_re,
                I2=I2,Q=Q,p_Q=pQ,tau2=tau2,k=k)

def run_meta(long_df):
    out=[]
    for au,grp in long_df.dropna(subset=["hedges_g","se_g"]).groupby("au"):
        if len(grp)<2: continue
        res=dersimonian_laird(grp["hedges_g"].values,grp["se_g"].values)
        out.append({"au":au,"n_studies":res["k"],"total_n_sz":grp["n_sz"].dropna().sum(),**res})
    return pd.DataFrame(out).sort_values("n_studies",ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 6-8. Plots (forest, bubble, heatmaps, source breakdown)
# ═══════════════════════════════════════════════════════════════════════════

DIR_COLORS={"decreased":"#2166AC","increased":"#D73027","no difference":"#888888",None:"#AAAAAA"}
SRC_MARKERS={"PubMed":"s","EuropePMC":"P","SemanticScholar":"o","OpenAlex":"D",
             "CORE":"h","CrossRef":"^","GoogleScholar":"v","Biorxiv":"*","Medrxiv":"*"}

def forest_plot(au,grp,meta,vocab_df):
    es=grp.dropna(subset=["hedges_g","se_g"]).copy()
    dirs=grp[grp["hedges_g"].isna()].copy()
    fig_h=max(5,(len(es)+len(dirs))*0.55+3)
    au_label=func_tag=""
    if not vocab_df.empty:
        vrow=vocab_df[vocab_df["au"]==au]
        if not vrow.empty:
            au_label=vrow.iloc[0].get("au_label",""); func_tag=vrow.iloc[0].get("functional_tag","")
    fig,axes=plt.subplots(1,2,figsize=(14,fig_h),gridspec_kw={"width_ratios":[3,2.5],"wspace":0.02})
    af,at=axes; y=0
    for _,row in es.iterrows():
        lo=row["hedges_g"]-1.96*row["se_g"]; hi=row["hedges_g"]+1.96*row["se_g"]
        col=DIR_COLORS.get(row["direction"],"#555"); mrk=SRC_MARKERS.get(row["source"],"s")
        af.plot([lo,hi],[y,y],color=col,lw=1.6,zorder=2)
        try: sz=max(25,min(220,float(row["n_sz"])*3))
        except: sz=50
        af.scatter([row["hedges_g"]],[y],s=sz,color=col,marker=mrk,zorder=3,edgecolors="white",linewidths=0.5)
        ft=f" [{row['feature_type']}]" if row.get("feature_type") else ""
        at.text(0.01,y,row["id"]+ft,va="center",ha="left",fontsize=7.5)
        at.text(0.50,y,row.get("source",""),va="center",ha="center",fontsize=6.5,color="#666")
        at.text(0.63,y,str(int(row["n_sz"])) if pd.notna(row["n_sz"]) else "?",va="center",ha="center",fontsize=8)
        at.text(0.75,y,str(int(row["n_hc"])) if pd.notna(row["n_hc"]) else "?",va="center",ha="center",fontsize=8)
        at.text(0.99,y,f"{row['hedges_g']:+.2f} [{lo:+.2f},{hi:+.2f}]",va="center",ha="right",fontsize=7.5,color=col,fontweight="bold")
        y-=1
    if len(dirs):
        y-=0.3; af.axhline(y+0.65,color="#CCC",lw=0.6,ls="--")
        at.text(0.01,y,"Direction only:",va="center",ha="left",fontsize=7,style="italic",color="#666"); y-=0.7
        for _,row in dirs.iterrows():
            col=DIR_COLORS.get(row["direction"],"#AAA"); ft=f" [{row['feature_type']}]" if row.get("feature_type") else ""
            af.scatter([0],[y],s=25,color=col,marker="^",alpha=0.5,zorder=2)
            at.text(0.01,y,row["id"]+ft,va="center",ha="left",fontsize=7.5)
            at.text(0.50,y,row.get("source",""),va="center",ha="center",fontsize=6.5,color="#666")
            at.text(0.63,y,str(int(row["n_sz"])) if pd.notna(row["n_sz"]) else "?",va="center",ha="center",fontsize=8)
            at.text(0.75,y,str(int(row["n_hc"])) if pd.notna(row["n_hc"]) else "?",va="center",ha="center",fontsize=8)
            at.text(0.99,y,row["direction"] or "?",va="center",ha="right",fontsize=8,color=col); y-=1
    if meta and len(es)>=2:
        y-=0.5; af.axhline(y+0.75,color="#444",lw=0.8)
        g,lo,hi=meta["pooled_g"],meta["ci_lo"],meta["ci_hi"]
        af.fill([lo,g,hi,g,lo],[y,y+0.38,y,y-0.38,y],color="#1A1A1A",zorder=4,alpha=0.85)
        at.text(0.01,y,f"Pooled (k={meta['k']})",va="center",ha="left",fontsize=9,fontweight="bold")
        at.text(0.99,y,f"{g:+.2f} [{lo:+.2f},{hi:+.2f}]",va="center",ha="right",fontsize=9,fontweight="bold")
        at.text(0.01,y-0.65,f"I\u00b2={meta['I2']:.0f}%  Q={meta['Q']:.1f}  p={meta['p_Q']:.3f}  \u03c4\u00b2={meta['tau2']:.3f}",
                va="center",ha="left",fontsize=7.5,color="#555",style="italic"); y-=1.6
    af.axvline(0,color="black",lw=1.0)
    af.set_xlabel("Hedges g  (SZ < HC = negative = hypo-expression)",fontsize=9)
    title=f"Forest Plot - {au}"
    if au_label: title+=f"\n{au_label}"+(" - "+func_tag if func_tag else "")
    af.set_title(title,fontsize=11,fontweight="bold",pad=10)
    af.set_ylim(y-0.5,1); af.set_yticks([]); af.spines[["top","right","left"]].set_visible(False)
    af.xaxis.grid(True,ls="--",alpha=0.4); af.set_axisbelow(True)
    at.set_xlim(0,1); at.set_ylim(y-0.5,1); at.axis("off")
    for x,lbl in [(0.01,"Study [feature type]"),(0.50,"Source"),(0.63,"n SZ"),(0.75,"n HC"),(0.99,"g [95% CI]")]:
        at.text(x,0.6,lbl,va="bottom",ha="left" if x<0.5 else ("center" if x<0.8 else "right"),fontsize=8,fontweight="bold",color="#222")
    at.axhline(0.55,color="#444",lw=0.8)
    legend_handles=[mpatches.Patch(color=DIR_COLORS["decreased"],label="Decreased in SZ"),
                    mpatches.Patch(color=DIR_COLORS["increased"],label="Increased in SZ"),
                    mpatches.Patch(color=DIR_COLORS["no difference"],label="No difference"),
                    Line2D([0],[0],marker="s",color="w",markerfacecolor="#333",markersize=8,label="Pooled"),
                   ]+[Line2D([0],[0],marker=m,color="w",markerfacecolor="#888",markersize=7,label=s)
                      for s,m in SRC_MARKERS.items()]
    fig.legend(handles=legend_handles,loc="lower center",ncol=5,fontsize=7,frameon=True,bbox_to_anchor=(0.5,0.0))
    plt.tight_layout(rect=[0,0.1,1,1])
    out_path=os.path.join(RESULTS_DIR,f"forest_{au.replace(' ','_')}.png")
    plt.savefig(out_path,dpi=160,bbox_inches="tight"); plt.close(); print(f"    {out_path}")

def make_all_forest_plots(long_df,meta_df,vocab_df):
    for au in long_df["au"].value_counts()[lambda x:x>=2].index:
        grp=long_df[long_df["au"]==au]; mrow=meta_df[meta_df["au"]==au]
        forest_plot(au,grp,mrow.iloc[0].to_dict() if not mrow.empty else None,vocab_df)

def bubble_summary(long_df,meta_df):
    if meta_df.empty: return
    fig,ax=plt.subplots(figsize=(11,6))
    sc=ax.scatter(meta_df["pooled_g"],meta_df["n_studies"],s=np.sqrt(meta_df["total_n_sz"].fillna(50))*10,
                  c=meta_df["I2"],cmap="RdYlGn_r",vmin=0,vmax=100,alpha=0.75,edgecolors="white",linewidths=0.8,zorder=3)
    plt.colorbar(sc,ax=ax,pad=0.02).set_label("I2 (%)",fontsize=9)
    ax.errorbar(meta_df["pooled_g"],meta_df["n_studies"],xerr=[meta_df["pooled_g"]-meta_df["ci_lo"],meta_df["ci_hi"]-meta_df["pooled_g"]],fmt="none",color="#444",lw=1,capsize=3,zorder=2)
    for _,r in meta_df.iterrows(): ax.annotate(r["au"],(r["pooled_g"],r["n_studies"]),xytext=(6,4),textcoords="offset points",fontsize=9,fontweight="bold")
    ax.axvline(0,color="black",lw=1); ax.set_xlabel("Pooled Hedges g",fontsize=10); ax.set_ylabel("Number of studies",fontsize=10)
    ax.set_title("AU Evidence Summary",fontsize=12,fontweight="bold"); ax.xaxis.grid(True,ls="--",alpha=0.4); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); out=os.path.join(RESULTS_DIR,"bubble_summary.png"); plt.savefig(out,dpi=160,bbox_inches="tight"); plt.close(); print(f"    {out}")

def direction_heatmap(long_df):
    dm={"decreased":-1,"no difference":0,"increased":1}
    sub=long_df[["id","au","direction"]].copy(); sub["d"]=sub["direction"].map(dm)
    pivot=sub.pivot_table(index="id",columns="au",values="d",aggfunc="first")
    if pivot.empty or pivot.shape[1]<2: return
    fig,ax=plt.subplots(figsize=(max(8,pivot.shape[1]*0.7),max(4,pivot.shape[0]*0.42)))
    im=ax.imshow(pivot.values,cmap="RdBu",vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns,rotation=45,ha="right",fontsize=9)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index,fontsize=8)
    ax.set_title("Direction of AU Effects Across Studies",fontsize=11,fontweight="bold")
    cb=fig.colorbar(im,ax=ax,ticks=[-1,0,1],fraction=0.03); cb.ax.set_yticklabels(["Decreased","No diff","Increased"],fontsize=8)
    plt.tight_layout(); out=os.path.join(RESULTS_DIR,"direction_heatmap.png"); plt.savefig(out,dpi=160,bbox_inches="tight"); plt.close(); print(f"    {out}")

def vocab_heatmap(vocab_df):
    if vocab_df.empty: return
    dm={"decreased":-1,"no difference":0,"increased":1}; v=vocab_df.copy(); v["d"]=v["dominant_direction"].map(dm)
    v["au_full"]=v["au"]+"\n"+v["au_label"].str[:18]
    regions=["brow","eye","nose","mouth"]; sub=v[v["face_region"].isin(regions)].copy()
    sub["face_region"]=pd.Categorical(sub["face_region"],categories=regions,ordered=True); sub=sub.sort_values(["face_region","au"])
    fig,ax=plt.subplots(figsize=(12,max(4,len(sub)*0.55+1))); colors=[]
    for _,row in sub.iterrows():
        d=row["d"]
        if pd.isna(d): colors.append("#DDD")
        elif d<0: a=min(1.0,row["n_studies"]/5); colors.append(plt.cm.Blues(0.4+a*0.5))
        elif d>0: a=min(1.0,row["n_studies"]/5); colors.append(plt.cm.Reds(0.4+a*0.5))
        else: colors.append("#AAA")
    bars=ax.barh(sub["au_full"],sub["n_studies"],color=colors,edgecolor="white")
    ax.bar_label(bars,labels=[f"k={v}" for v in sub["n_studies"]],padding=4,fontsize=8)
    seen=[]
    for i,(_,row) in enumerate(sub.iterrows()):
        if row["face_region"] not in seen:
            seen.append(row["face_region"])
            if i>0: ax.axhline(i-0.5,color="#999",lw=0.8,ls="--")
            ax.text(sub["n_studies"].max()*1.08,i,row["face_region"].upper(),va="center",ha="left",fontsize=8,color="#444",fontweight="bold")
    for i,(_,row) in enumerate(sub.iterrows()):
        ax.text(0.1,i,f"  {row['functional_tag']}",va="center",ha="left",fontsize=7,color="white" if row["d"] else "#555",style="italic")
    ax.set_xlabel("Number of studies",fontsize=10); ax.set_title("AU Feature Vocabulary by Face Region\n(Blue=hypo, Red=hyper in SZ)",fontsize=11,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); out=os.path.join(RESULTS_DIR,"vocab_heatmap.png"); plt.savefig(out,dpi=160,bbox_inches="tight"); plt.close(); print(f"    {out}")

def source_breakdown(long_df):
    counts=long_df.groupby("source")["paper_id"].nunique().sort_values(ascending=True)
    if counts.empty: return
    fig,ax=plt.subplots(figsize=(8,4))
    bars=ax.barh(counts.index,counts.values,color=["#4878D0","#EE854A","#6ACC65","#D65F5F","#956CB4","#8C613C","#DC7EC0","#797979"][:len(counts)])
    ax.bar_label(bars,padding=4,fontsize=9); ax.set_xlabel("Papers with AU data",fontsize=9)
    ax.set_title("Papers Retrieved by Source",fontsize=11,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); out=os.path.join(RESULTS_DIR,"source_breakdown.png"); plt.savefig(out,dpi=160,bbox_inches="tight"); plt.close(); print(f"    {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. LLM narrative
# ═══════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT=textwrap.dedent("""
You are writing the Results section of a systematic review on facial AUs in schizophrenia.
Write ~600 words covering:
1. Overall hypo-expression pattern and affected face regions
2. Specific AUs with strongest evidence with anatomical label and functional meaning
3. Whether effects are intensity, frequency, or both
4. AUs specifically linked to apathy vs broader negative symptoms
5. Heterogeneity (I2) interpretation
6. Recommended ML feature set specifying feature type per AU

Meta-analysis: {meta_json}
Feature vocabulary: {vocab_json}
Direction-only: {dir_json}
Sources: {src_json}
""").strip()

def narrative_summary(meta_df,long_df,vocab_df):
    dir_s=(long_df[long_df["hedges_g"].isna()].groupby("au")["direction"].value_counts().reset_index().to_dict(orient="records"))
    src_s=long_df.groupby("source")["paper_id"].nunique().to_dict()
    return llm_chat(SUMMARY_PROMPT.format(meta_json=meta_df.to_json(orient="records",indent=2),
                                           vocab_json=vocab_df.to_json(orient="records",indent=2),
                                           dir_json=json.dumps(dir_s,indent=2),
                                           src_json=json.dumps(src_s,indent=2)),max_tokens=1500)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    print("=== Step 1: Retrieving papers (8 sources) ===")
    papers=collect_all_papers()
    print("\n=== Step 1b: Relevance ranking ===")
    papers = rank_papers(papers)

    print("\n=== Step 2: Full-text retrieval ===")
    papers=enrich_with_full_text(papers)
    print(f"\n=== Step 3: LLM extraction ({LLM_MODEL}) ===")
    extractions=[]
    for p in tqdm(papers): extractions.append(extract_paper(p)); time.sleep(0.3)
    with open(f"{RESULTS_DIR}/raw_extractions.json","w") as f:
        json.dump([{**p,"extraction":e} for p,e in zip(papers,extractions)],f,indent=2)
    print("\n=== Step 4: Building tables ===")
    long_df=build_long_table(papers,extractions)
    if long_df.empty: print("  No AU data. Check raw_extractions.json"); exit(1)
    long_df.to_csv(f"{RESULTS_DIR}/au_long.csv",index=False)
    print(f"  {len(long_df)} AU observations, {long_df['hedges_g'].notna().sum()} with effect sizes")
    vocab_df=build_feature_vocab(long_df)
    vocab_df.to_csv(f"{RESULTS_DIR}/au_feature_vocab.csv",index=False)
    print(vocab_df[["au","au_label","face_region","functional_tag","dominant_direction","feature_types","n_studies"]].to_string(index=False))
    print("\n=== Step 5: Meta-analyses ===")
    meta_df=run_meta(long_df); meta_df.to_csv(f"{RESULTS_DIR}/meta_results.csv",index=False)
    if not meta_df.empty: print(meta_df[["au","n_studies","pooled_g","ci_lo","ci_hi","I2","p_Q"]].to_string(index=False))
    print("\n=== Step 6: Forest plots ==="); make_all_forest_plots(long_df,meta_df,vocab_df)
    print("\n=== Step 7: Charts ==="); bubble_summary(long_df,meta_df); direction_heatmap(long_df); vocab_heatmap(vocab_df); source_breakdown(long_df)
    print("\n=== Step 8: Narrative ==="); narrative=narrative_summary(meta_df,long_df,vocab_df)
    print(narrative[:500]+"...\n")
    with open(f"{RESULTS_DIR}/meta_analysis_summary.txt","w") as f: f.write(narrative)
    print(f"\nDone. Outputs in ./{RESULTS_DIR}/")
    print("""
NEXT STEPS:
  1. Open results/meta/au_feature_vocab.csv
  2. Fill recommended_feature and include_in_model columns
  3. Update AU_INTENSITIES in 02_feature_extraction.py
  Optional: export CORE_API_KEY=... (free at core.ac.uk/api-keys)
""")
