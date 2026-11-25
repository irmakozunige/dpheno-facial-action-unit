"""
00_retrieval_eval.py
--------------------
RAGAS-style evaluation of the retrieval + extraction pipeline.

TWO MODES:

  Mode 1 — SAMPLE (run BEFORE meta-analysis to do a quick sanity check):
    python src/00_retrieval_eval.py --mode sample
    Retrieves 20 papers, scores retrieval quality, helps tune queries.

  Mode 2 — FULL (run AFTER meta-analysis to evaluate actual results):
    python src/00_retrieval_eval.py --mode full
    Reads results/meta/raw_extractions.json + au_long.csv produced by
    01_meta_analysis.py and evaluates the real pipeline output.

Metrics:
  Context Relevance   — are retrieved chunks about AUs/schizophrenia?
  Faithfulness        — does the extracted AU claim follow from the chunk?
  Answer Relevance    — does the extraction address the research question?
  Extraction Yield    — % papers that produced >= 1 AU finding
  Full-text Hit Rate  — papers with full text vs abstract-only
  AU Keyword Hit Rate — chunks that mention actual AU codes
  Chunk Dedup Rate    — redundancy across sources

Optional gold set (Mode 2):
  Fill GOLD_SET below after reading papers manually to unlock
  Context Recall and Precision@AU metrics.

Outputs in results/eval/:
  retrieval_eval.csv
  retrieval_eval.png
  eval_report.txt
"""

import os, sys, json, time, hashlib, re, textwrap, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from Bio import Entrez

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "eval")
META_DIR     = os.path.join(PROJECT_ROOT, "results", "meta")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EVAL_MODEL     = os.environ.get("EVAL_MODEL", "gemini-3.5-flash-lite")
SAMPLE_N       = int(os.environ.get("EVAL_SAMPLE", "20"))

Entrez.email = "irmak.ozarslan@unige.ch"
BROWSER_UA   = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

RESEARCH_QUESTION = (
    "Which facial Action Units (AUs) are significantly different in intensity "
    "or frequency between schizophrenia patients and healthy controls, "
    "and what is the direction and magnitude of those differences?"
)

GOLD_SET = [
    # Fill after manually reading papers:
    # {
    #     "doi":          "10.1016/j.schres.2020.01.001",
    #     "aus_in_paper": ["AU6", "AU12", "AU25"],
    #     "key_sentence": "AU12 intensity was significantly lower in SZ...",
    # },
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Gemini SDK client
# ═══════════════════════════════════════════════════════════════════════════

_gemini_client = None
_rpm_times: list[float] = []
_day_count: list[int]   = [0]


def _init_gemini():
    global _gemini_client
    if _gemini_client is not None:
        return
    from google import genai
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _rate_gate():
    now = time.time()
    _rpm_times[:] = [t for t in _rpm_times if now - t < 60]
    if len(_rpm_times) >= 14:
        wait = 60 - (now - _rpm_times[0]) + 1
        print(f"  [rate limit] waiting {wait:.0f}s")
        time.sleep(wait)
    if _day_count[0] >= 1490:
        raise RuntimeError("Gemini daily quota reached (1490 req).")
    _rpm_times.append(time.time())
    _day_count[0] += 1


def gemini_judge(prompt: str, debug: bool = False) -> str:
    _init_gemini()
    from google.genai import types as _types
    for attempt in range(5):
        _rate_gate()
        try:
            response = _gemini_client.models.generate_content(
                model=EVAL_MODEL,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.0,
                ),
            )
            if debug:
                finish = (response.candidates[0].finish_reason
                          if response.candidates else "NO_CANDIDATES")
                print(f"    finish_reason={finish}  day_requests={_day_count[0]}")
            if not response.candidates:
                return "BLOCKED:NO_CANDIDATES"
            text = (response.text or "").strip()
            if not text:
                finish = response.candidates[0].finish_reason
                if str(finish) == "SAFETY":
                    time.sleep(2); continue
                return ""
            return text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 2 ** attempt * 10
                print(f"  [429] backing off {wait}s")
                time.sleep(wait)
            elif "503" in err:
                time.sleep(30)
            else:
                if debug: print(f"  SDK error attempt {attempt+1}: {e}")
                time.sleep(5)
    return ""


def parse_score(text: str) -> float:
    m = re.search(r"[Ss]core\s*[:\-]\s*([\d.]+)", text)
    if m:
        v = float(m.group(1))
        return v if v <= 1.0 else v / 5.0
    m = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"\b(0\.\d+|1\.0)\b", text)
    if m:
        return float(m.group(1))
    return 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 2. RAGAS-style prompts
# ═══════════════════════════════════════════════════════════════════════════

CR_PROMPT = textwrap.dedent("""
You are evaluating a retrieval system for a systematic meta-analysis on
facial Action Units (AUs) in schizophrenia.

Research question: {question}

Retrieved chunk: \"\"\"{chunk}\"\"\"

Score how relevant this chunk is (0=irrelevant, 1=perfectly relevant).
A relevant chunk reports AU numbers, schizophrenia patients, and
quantitative comparisons (means, p-values, effect sizes).

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

FAITH_PROMPT = textwrap.dedent("""
Source chunk: \"\"\"{chunk}\"\"\"
Extracted claim: \"\"\"{claim}\"\"\"

Does the claim follow from the source without adding missing information?
Score 1.0=fully supported, 0.5=partial, 0.0=hallucinated.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

AR_PROMPT = textwrap.dedent("""
Research question: {question}
Extracted finding: \"\"\"{finding}\"\"\"

Does this finding directly address the research question?
Score 1.0=directly answers it, 0.5=tangential, 0.0=off-topic.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

RECALL_PROMPT = textwrap.dedent("""
Gold sentence: \"\"\"{gold_sentence}\"\"\"
Retrieved context: \"\"\"{context}\"\"\"

Is the gold sentence (or its information) present in the context?
Score 1.0=present, 0.5=partial, 0.0=missing.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

REPORT_PROMPT = textwrap.dedent("""
You are evaluating a retrieval + extraction pipeline for a meta-analysis on
facial Action Units in schizophrenia.

Evaluation results (JSON): {results_json}

Write a concise report (~300 words):
1. Overall retrieval quality
2. Full-text coverage and impact on extraction
3. Source-level quality differences
4. Faithfulness / hallucination risk
5. Specific recommendations to improve before the full run

Flag any metric below 0.5 as requiring attention. Be direct and practical.
""").strip()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scoring functions
# ═══════════════════════════════════════════════════════════════════════════

def score_context_relevance(chunks: list[dict]) -> float:
    if not chunks: return 0.0
    scores = []
    for chunk in chunks[:5]:
        resp = gemini_judge(CR_PROMPT.format(
            question=RESEARCH_QUESTION, chunk=chunk["text"][:600]))
        scores.append(parse_score(resp))
    return float(np.mean(scores)) if scores else 0.0


def score_faithfulness(claim: str, chunks: list[dict]) -> float:
    if not claim or not chunks: return 0.5
    best = max(chunks, key=lambda c: c.get("_score", 0))
    resp = gemini_judge(FAITH_PROMPT.format(
        chunk=best["text"][:600], claim=claim[:300]))
    return parse_score(resp)


def score_answer_relevance(finding: str) -> float:
    if not finding: return 0.0
    resp = gemini_judge(AR_PROMPT.format(
        question=RESEARCH_QUESTION, finding=finding[:400]))
    return parse_score(resp)


def score_recall(gold_sentence: str, chunks: list[dict]) -> float:
    combined = "\n".join(c["text"][:300] for c in chunks[:5])
    resp = gemini_judge(RECALL_PROMPT.format(
        gold_sentence=gold_sentence, context=combined))
    return parse_score(resp)


def generate_report(df: pd.DataFrame) -> str:
    summary      = df.describe().round(3).to_dict()
    source_means = {}
    if "source" in df.columns:
        source_means = (df.groupby("source")
                          [["context_relevance", "au_hit_rate"]]
                          .mean().round(3).to_dict())
    results_json = json.dumps(
        {"summary": summary, "by_source": source_means}, indent=2)
    result = gemini_judge(
        REPORT_PROMPT.format(results_json=results_json[:3000]))
    return result if result and not result.startswith("BLOCKED") else \
           "Report generation failed."


# ═══════════════════════════════════════════════════════════════════════════
# 4. Non-LLM metrics
# ═══════════════════════════════════════════════════════════════════════════

def chunk_dedup_rate(all_chunks: list[dict]) -> float:
    if len(all_chunks) < 2: return 0.0
    def shingles(text, k=5):
        words = text.lower().split()
        return set(tuple(words[i:i+k]) for i in range(len(words)-k+1))
    total = dups = 0
    sets = [shingles(c["text"]) for c in all_chunks[:100]]
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            total += 1
            if sets[i] and sets[j]:
                j_sim = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                if j_sim > 0.8: dups += 1
    return dups / total if total > 0 else 0.0


def au_pattern_hit_rate(chunks: list[dict]) -> float:
    au_re = re.compile(r"\bAU\s*\d{1,2}\b", re.I)
    hits  = sum(1 for c in chunks[:10] if au_re.search(c["text"]))
    return hits / min(10, len(chunks)) if chunks else 0.0


def make_chunks(text: str, doi: str = "") -> list[dict]:
    AU_PAT  = re.compile(r"\bAU\d{1,2}\b", re.I)
    KEY_PAT = re.compile(
        r"\b(action unit|schizophrenia|flat affect|blunted|apathy|negative symptom|"
        r"FACS|OpenFace|intensity|frequency|mean|SD|significant|decreased|increased)\b",
        re.I)
    NUM_PAT = re.compile(r"[0-9]+\.[0-9]+")
    chunks  = []
    for i, para in enumerate(text.split("\n\n")):
        para = para.strip()
        if len(para) < 40: continue
        score = (len(AU_PAT.findall(para)) * 5 +
                 len(KEY_PAT.findall(para)) +
                 min(len(NUM_PAT.findall(para)), 5))
        chunks.append({"doi": doi, "page": 0, "para_idx": i,
                        "text": para[:800], "_score": score})
    ranked = sorted(chunks, key=lambda c: c["_score"], reverse=True)
    return [c for c in ranked if c["_score"] > 0][:15]


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODE 1 — Sample retrieval (pre meta-analysis sanity check)
# ═══════════════════════════════════════════════════════════════════════════

def _get(url, params=None, timeout=12):
    try:
        return requests.get(url, params=params,
                            headers={"User-Agent": BROWSER_UA}, timeout=timeout)
    except Exception:
        return None


def fetch_sample_papers(n: int) -> list[dict]:
    seen, papers = set(), []

    # PubMed with MeSH-anchored query
    pubmed_query = (
        '(("facial expression"[MeSH] OR "action unit"[tiab] OR "AU1"[tiab] OR '
        '"AU4"[tiab] OR "AU12"[tiab] OR "OpenFace"[tiab] OR "FACS"[tiab] OR '
        '"facial action coding"[tiab]) AND '
        '("schizophrenia"[MeSH] OR "psychosis"[MeSH] OR "negative symptoms"[tiab]))'
    )
    try:
        h   = Entrez.esearch(db="pubmed", term=pubmed_query,
                              retmax=n, sort="relevance")
        ids = Entrez.read(h)["IdList"]; h.close()
        h   = Entrez.efetch(db="pubmed", id=ids,
                            rettype="abstract", retmode="xml")
        recs = Entrez.read(h)["PubmedArticle"]; h.close()
        for rec in recs:
            art  = rec["MedlineCitation"]["Article"]
            ttl  = str(art.get("ArticleTitle", ""))
            abst = " ".join(str(p) for p in
                            art.get("Abstract", {}).get("AbstractText", []))
            pmid = str(rec["MedlineCitation"]["PMID"])
            ids2 = rec.get("PubmedData", {}).get("ArticleIdList", [])
            doi  = next((str(x) for x in ids2
                         if x.attributes.get("IdType") == "doi"), "")
            fp   = hashlib.md5(ttl.lower().encode()).hexdigest()
            if abst.strip() and fp not in seen:
                seen.add(fp)
                papers.append({"id": f"pmid_{pmid}", "source": "PubMed",
                                "title": ttl, "abstract": abst, "doi": doi,
                                "full_text": None, "chunks": []})
    except Exception as e:
        print(f"  PubMed error: {e}")

    # Semantic Scholar
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search",
             params={"query": "facial action units AU schizophrenia blunted affect "
                              "FACS OpenFace intensity frequency",
                     "limit": n,
                     "fields": "title,abstract,externalIds,openAccessPdf"})
    if r and r.status_code == 200:
        for p in r.json().get("data", []):
            abst = p.get("abstract") or ""
            ttl  = p.get("title", "") or ""
            fp   = hashlib.md5(ttl.lower().encode()).hexdigest()
            if abst.strip() and fp not in seen:
                seen.add(fp)
                doi = (p.get("externalIds") or {}).get("DOI", "")
                oa  = (p.get("openAccessPdf") or {}).get("url", "")
                papers.append({"id": f"ss_{p.get('paperId', '')}",
                                "source": "SemanticScholar",
                                "title": ttl, "abstract": abst,
                                "doi": doi, "full_text": None,
                                "chunks": [], "oa_pdf": oa})

    return papers[:n]


def fetch_fulltext_for_paper(paper: dict) -> dict:
    doi = paper.get("doi", "")
    text = None

    if doi:
        r = _get(f"https://api.unpaywall.org/v2/{doi}",
                 params={"email": Entrez.email})
        if r and r.status_code == 200:
            loc = r.json().get("best_oa_location") or {}
            url = loc.get("url_for_pdf") or loc.get("url")
            if url:
                pr = _get(url, timeout=25)
                if pr and pr.status_code == 200 and \
                        "pdf" in pr.headers.get("Content-Type", ""):
                    try:
                        import pymupdf as fitz
                        doc  = fitz.open(stream=pr.content, filetype="pdf")
                        text = "\n".join(p.get_text() for p in doc)
                        doc.close()
                    except Exception:
                        pass

    if not text:
        oa = paper.get("oa_pdf", "")
        if oa:
            pr = _get(oa, timeout=25)
            if pr and pr.status_code == 200:
                try:
                    import pymupdf as fitz
                    doc  = fitz.open(stream=pr.content, filetype="pdf")
                    text = "\n".join(p.get_text() for p in doc)
                    doc.close()
                except Exception:
                    pass

    if text:
        paper["full_text"] = text
        paper["chunks"]    = make_chunks(text, doi)
    else:
        paper["full_text"] = None
        paper["chunks"]    = [{"doi": doi, "page": 0, "para_idx": 0,
                                "text": paper["abstract"], "_score": 3}]
    return paper


def run_sample_mode() -> pd.DataFrame:
    print(f"=== Step 1: Retrieving {SAMPLE_N} sample papers ===")
    papers = fetch_sample_papers(SAMPLE_N)
    print(f"  Got {len(papers)} papers")

    print("\n=== Step 2: Full-text retrieval ===")
    for p in tqdm(papers):
        fetch_fulltext_for_paper(p)
        time.sleep(0.3)
    n_ft = sum(1 for p in papers if p["full_text"])
    print(f"  Full text: {n_ft}/{len(papers)}")

    print("\n=== Step 3: Dedup rate ===")
    all_chunks = [c for p in papers for c in p["chunks"]]
    dedup = chunk_dedup_rate(all_chunks)
    print(f"  Dedup rate: {dedup:.1%}")

    print(f"\n=== Step 4: LLM evaluation ===")
    rows = []
    for paper in tqdm(papers):
        gold = next((g for g in GOLD_SET
                     if g["doi"] == paper.get("doi", "")), None)
        row  = evaluate_paper_row(paper, gold, dedup)
        rows.append(row)
        time.sleep(0.2)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. MODE 2 — Full evaluation on meta-analysis outputs
# ═══════════════════════════════════════════════════════════════════════════

def run_full_mode() -> pd.DataFrame:
    extractions_path = os.path.join(META_DIR, "raw_extractions.json")
    au_long_path     = os.path.join(META_DIR, "au_long.csv")

    if not os.path.exists(extractions_path):
        print(f"ERROR: {extractions_path} not found.")
        print("Run 01_meta_analysis.py first, then re-run this script.")
        sys.exit(1)

    print(f"=== Loading meta-analysis outputs ===")
    with open(extractions_path) as f:
        raw = json.load(f)
    print(f"  {len(raw)} papers in raw_extractions.json")

    if os.path.exists(au_long_path):
        au_df = pd.read_csv(au_long_path)
        print(f"  {len(au_df)} AU observations in au_long.csv")
        print(f"  Extraction yield: "
              f"{au_df['paper_id'].nunique()}/{len(raw)} papers "
              f"({au_df['paper_id'].nunique()/len(raw):.0%}) produced AU findings")
    else:
        au_df = pd.DataFrame()

    # Rebuild paper dicts from raw_extractions.json
    papers = []
    for entry in raw:
        ext  = entry.get("extraction", {})
        text = entry.get("full_text") or entry.get("abstract", "")
        doi  = entry.get("doi", "")
        papers.append({
            "id":           entry.get("id", ""),
            "source":       entry.get("source", ""),
            "title":        entry.get("title", ""),
            "abstract":     entry.get("abstract", ""),
            "doi":          doi,
            "full_text":    entry.get("full_text"),
            "chunks":       make_chunks(text, doi) if text else [],
            "extraction":   ext,
        })

    print(f"\n=== Dedup rate ===")
    all_chunks = [c for p in papers for c in p["chunks"]]
    dedup = chunk_dedup_rate(all_chunks)
    print(f"  Dedup rate: {dedup:.1%}")

    print(f"\n=== LLM evaluation on {len(papers)} papers ===")
    rows = []
    for paper in tqdm(papers):
        gold = next((g for g in GOLD_SET
                     if g["doi"] == paper.get("doi", "")), None)
        row  = evaluate_paper_row(paper, gold, dedup)

        # Extra metrics only available in full mode
        ext = paper.get("extraction", {})
        aus = ext.get("aus_studied", []) or []
        row["n_aus_extracted"]  = len(aus)
        row["aus_extracted"]    = ", ".join(aus)
        row["extraction_error"] = "error" in ext

        rows.append(row)
        time.sleep(0.2)

    df = pd.DataFrame(rows)

    # Extraction yield
    if not au_df.empty:
        print(f"\n=== AU extraction summary ===")
        print(f"  Total AU observations: {len(au_df)}")
        print(f"  Unique AUs found: {au_df['au'].nunique()}")
        print(f"  AUs: {', '.join(sorted(au_df['au'].unique()))}")
        top_aus = au_df["au"].value_counts().head(10)
        print(f"  Top AUs by paper count:\n{top_aus.to_string()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7. Shared per-paper evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_paper_row(paper: dict, gold: dict | None,
                        dedup: float) -> dict:
    row = {
        "id":                paper["id"],
        "source":            paper.get("source", ""),
        "title":             paper.get("title", "")[:80],
        "has_full_text":     paper.get("full_text") is not None,
        "n_chunks":          len(paper.get("chunks", [])),
        "au_hit_rate":       au_pattern_hit_rate(paper.get("chunks", [])),
        "dedup_rate":        dedup,
        "context_relevance": None,
        "faithfulness":      None,
        "answer_relevance":  None,
        "context_recall":    None,
        "precision_at_au":   None,
    }

    if not GEMINI_API_KEY:
        return row

    chunks = paper.get("chunks", [])
    row["context_relevance"] = score_context_relevance(chunks)

    # Use extraction claim if available (full mode), else best chunk
    ext   = paper.get("extraction", {})
    claim = ""
    if ext and ext.get("findings"):
        first_au = next(iter(ext["findings"].values()), {})
        claim    = first_au.get("exact_quote", "") or ""
    if not claim and chunks:
        best  = max(chunks, key=lambda c: c.get("_score", 0))
        if re.search(r"AU\s*\d{1,2}", best["text"], re.I):
            claim = best["text"][:300]

    if claim:
        row["faithfulness"]     = score_faithfulness(claim, chunks)
        row["answer_relevance"] = score_answer_relevance(claim)

    if gold:
        row["context_recall"] = score_recall(
            gold["key_sentence"], chunks)
        au_re     = re.compile(r"AU\s*(\d{1,2})", re.I)
        found_aus = list({f"AU{m.group(1)}" for c in chunks
                          for m in au_re.finditer(c["text"])})
        if found_aus and gold.get("aus_in_paper"):
            ext_set = {a.upper().replace(" ", "") for a in found_aus}
            gld_set = {a.upper().replace(" ", "")
                       for a in gold["aus_in_paper"]}
            row["precision_at_au"] = len(ext_set & gld_set) / len(ext_set)

    return row


# ═══════════════════════════════════════════════════════════════════════════
# 8. Shared output — plots + report
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print("\n=== Evaluation Summary ===")
    for col in ["context_relevance", "faithfulness", "answer_relevance"]:
        if col in df.columns and df[col].notna().any():
            print(f"  {col:25s}: {df[col].mean():.3f}  "
                  f"(min {df[col].min():.2f}, max {df[col].max():.2f})")
    print(f"  {'full_text_hit_rate':25s}: {df['has_full_text'].mean():.1%}")
    print(f"  {'au_keyword_hit_rate':25s}: {df['au_hit_rate'].mean():.1%}")
    print(f"  {'chunk_dedup_rate':25s}: {df['dedup_rate'].iloc[0]:.1%}")

    print("\n=== Paper breakdown (sorted by context relevance) ===")
    sort_col = "context_relevance" if df["context_relevance"].notna().any() \
               else "au_hit_rate"
    for _, row in df.sort_values(sort_col, ascending=False).iterrows():
        cr  = f"{row['context_relevance']:.2f}" \
              if pd.notna(row["context_relevance"]) else " n/a"
        hit = "AU✓" if row["au_hit_rate"] > 0 else "   "
        ft  = "FT✓" if row["has_full_text"] else "   "
        print(f"  {hit} {ft} CR={cr}  {row['title'][:65]}")


def plot_results(df: pd.DataFrame, mode: str):
    llm_cols = [c for c in ["context_relevance", "faithfulness",
                             "answer_relevance"]
                if c in df.columns and df[c].notna().any()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Retrieval Evaluation — {mode.upper()} mode",
                 fontsize=13, fontweight="bold")

    # Radar
    ax = axes[0]
    if llm_cols:
        means  = df[llm_cols].mean()
        angles = np.linspace(0, 2*np.pi, len(means), endpoint=False).tolist()
        vals   = means.tolist()
        angles += angles[:1]; vals += vals[:1]
        ax2 = fig.add_axes([0.03, 0.12, 0.28, 0.76], polar=True)
        ax2.plot(angles, vals, "o-", lw=2, color="#4878D0")
        ax2.fill(angles, vals, alpha=0.2, color="#4878D0")
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([c.replace("_", "\n") for c in llm_cols],
                             fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.set_title("RAGAS-style metrics", fontsize=10, pad=14)
        ax.remove()
    else:
        ax.text(0.5, 0.5, "No LLM scores", ha="center", va="center",
                fontsize=11, color="gray")
        ax.axis("off")

    # Per-source box
    ax = axes[1]
    if "context_relevance" in df.columns and \
            df["context_relevance"].notna().any() and "source" in df.columns:
        groups = {src: grp["context_relevance"].dropna().tolist()
                  for src, grp in df.groupby("source")
                  if grp["context_relevance"].notna().any()}
        if groups:
            ax.boxplot(list(groups.values()), labels=list(groups.keys()),
                       patch_artist=True,
                       boxprops={"facecolor": "#B5D4F4", "linewidth": 0.8},
                       medianprops={"color": "#185FA5", "linewidth": 2})
            ax.set_ylabel("Context Relevance", fontsize=10)
            ax.set_title("Relevance by source", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                     fontsize=8)
    else:
        ax.text(0.5, 0.5, "No per-source data", ha="center", va="center",
                fontsize=11, color="gray")
        ax.axis("off")

    # System metrics
    ax = axes[2]
    sys_means = {
        "Full-text\nhit rate":  df["has_full_text"].mean(),
        "AU keyword\nhit rate": df["au_hit_rate"].mean(),
        "Mean chunks\n÷10":     df["n_chunks"].mean() / 10,
    }
    if mode == "full" and "n_aus_extracted" in df.columns:
        sys_means["Mean AUs\nextracted"] = df["n_aus_extracted"].mean() / 5
    bars = ax.bar(list(sys_means.keys()), list(sys_means.values()),
                   color=["#6ACC65", "#4878D0", "#EE854A", "#956CB4"],
                   edgecolor="white")
    ax.bar_label(bars,
                 labels=[f"{v:.0%}" if i < 2 else f"{v*10:.1f}"
                         for i, v in enumerate(sys_means.values())],
                 padding=3, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("System metrics", fontsize=10)
    ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"retrieval_eval_{mode}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    except Exception as e:
        print(f"  Plot save failed: {e}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"],
                        default="sample",
                        help="sample=pre-analysis check, full=evaluate meta-analysis output")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not set — LLM metrics skipped.\n")
    else:
        print(f"=== Gemini smoke test ({EVAL_MODEL}) ===")
        try:
            resp = gemini_judge("Reply with exactly the word: OK", debug=True)
            print(f"  Response: {resp} {'✓' if 'OK' in resp.upper() else '✗'}")
            if "OK" not in resp.upper():
                print("  WARNING: unexpected response — LLM metrics may be unreliable")
        except Exception as e:
            print(f"  FAILED: {e}")
            GEMINI_API_KEY = ""

    # Run selected mode
    if args.mode == "sample":
        df = run_sample_mode()
    else:
        df = run_full_mode()

    # Save CSV
    out_csv = os.path.join(RESULTS_DIR, f"retrieval_eval_{args.mode}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nCSV saved: {out_csv}")

    print_summary(df)

    print(f"\n=== Plots ===")
    plot_results(df, args.mode)

    if GEMINI_API_KEY:
        print(f"\n=== Narrative report ===")
        report = generate_report(df)
        print(report)
        out_rep = os.path.join(RESULTS_DIR, f"eval_report_{args.mode}.txt")
        with open(out_rep, "w") as f:
            f.write(report)
        print(f"Report saved: {out_rep}")

    print(f"\nDone. Outputs in {RESULTS_DIR}/")
    print("""
THRESHOLDS:
  context_relevance  > 0.6  → on-topic chunks          ✓
  faithfulness       > 0.7  → low hallucination risk    ✓
  answer_relevance   > 0.6  → extraction addresses RQ   ✓
  full_text_hit_rate > 0.4  → good OA coverage          ✓
  au_hit_rate        > 0.5  → AU codes in chunks        ✓
  dedup_rate         < 0.3  → manageable redundancy     ✓
""")
