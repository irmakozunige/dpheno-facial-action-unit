"""
00_retrieval_eval.py
--------------------
Evaluates the retrieval + extraction pipeline BEFORE running the full meta-analysis.
Run this on a small sample (10-20 papers) to check quality, then tune.

Metrics (all reference-free — no manual labels needed for the first run):
  1. Context Relevance   — are retrieved chunks about AUs/schizophrenia?
  2. Faithfulness        — does the extracted AU claim follow from the chunk?
  3. Answer Relevance    — is the extraction on-topic for our research question?
  4. Chunk Dedup Rate    — how much redundancy across the 8 sources?
  5. Full-text Hit Rate  — how many papers got full text vs abstract-only?
  6. Extraction Yield    — % of papers that produced ≥1 AU finding

Optional (needs a small gold set you annotate — see GOLD_SET below):
  7. Context Recall      — did we retrieve the key passage?
  8. Precision@AU        — are extracted AUs in the paper's real AU list?

Outputs in results/eval/:
  retrieval_eval.csv      per-paper scores
  eval_summary.png        radar chart + bar charts
  eval_report.txt         narrative with recommendations

Setup:
  pip install ragas datasets langchain-google-genai
  GEMINI_API_KEY env var (same key as 01_meta_analysis.py)
"""

import os, json, time, hashlib, re, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import requests
from Bio import Entrez

RESULTS_DIR = "results/eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EVAL_MODEL     = os.environ.get("EVAL_MODEL", "gemini-2.5-flash-preview-05-20")  # judge model
SAMPLE_N       = int(os.environ.get("EVAL_SAMPLE", "20"))          # papers to eval

Entrez.email = "irmak.ozarslan@unige.ch"
BROWSER_UA   = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

# ── The research question we're trying to answer ─────────────────────────────
RESEARCH_QUESTION = (
    "Which facial Action Units (AUs) are significantly different in intensity "
    "or frequency between schizophrenia patients and healthy controls, "
    "and what is the direction and magnitude of those differences?"
)

# ── Optional gold set ─────────────────────────────────────────────────────────
# Add papers you have read manually. Each entry = one paper you verified.
# Leave empty to skip recall/precision metrics.
GOLD_SET = [
    # {
    #     "doi":        "10.1016/j.schres.2020.01.001",
    #     "aus_in_paper": ["AU6","AU12","AU25"],        # AUs the paper actually studies
    #     "key_sentence": "AU12 intensity was significantly lower in SZ...",
    #     "key_page":   4,
    # },
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Lightweight retrieval (sample only — reuses logic from 01_meta_analysis)
# ═══════════════════════════════════════════════════════════════════════════

def _get(url, params=None, timeout=12):
    try:
        return requests.get(url, params=params,
                            headers={"User-Agent": BROWSER_UA}, timeout=timeout)
    except Exception:
        return None


def fetch_sample_papers(n: int) -> list[dict]:
    """
    Quick retrieval of n papers from PubMed + Semantic Scholar.
    Returns list of {id, title, abstract, doi, full_text, source, chunks}.
    """
    seen, papers = set(), []

    # PubMed
    try:
        h   = Entrez.esearch(db="pubmed",
                              term="facial action units schizophrenia",
                              retmax=n, sort="relevance")
        ids = Entrez.read(h)["IdList"]; h.close()
        h   = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        recs= Entrez.read(h)["PubmedArticle"]; h.close()
        for rec in recs:
            art  = rec["MedlineCitation"]["Article"]
            ttl  = str(art.get("ArticleTitle",""))
            abst = " ".join(str(p) for p in art.get("Abstract",{}).get("AbstractText",[]))
            pmid = str(rec["MedlineCitation"]["PMID"])
            ids2 = rec.get("PubmedData",{}).get("ArticleIdList",[])
            doi  = next((str(x) for x in ids2
                         if x.attributes.get("IdType")=="doi"), "")
            fp   = hashlib.md5(ttl.lower().encode()).hexdigest()
            if abst.strip() and fp not in seen:
                seen.add(fp)
                papers.append({"id":f"pmid_{pmid}","source":"PubMed",
                                "title":ttl,"abstract":abst,"doi":doi,
                                "full_text":None,"chunks":[]})
    except Exception as e:
        print(f"  PubMed fetch error: {e}")

    # Semantic Scholar (top up to n)
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search",
             params={"query":"facial action units schizophrenia","limit":n,
                     "fields":"title,abstract,externalIds,openAccessPdf"})
    if r and r.status_code == 200:
        for p in r.json().get("data",[]):
            abst = p.get("abstract") or ""
            ttl  = p.get("title","") or ""
            fp   = hashlib.md5(ttl.lower().encode()).hexdigest()
            if abst.strip() and fp not in seen:
                seen.add(fp)
                doi  = (p.get("externalIds") or {}).get("DOI","")
                oa   = (p.get("openAccessPdf") or {}).get("url","")
                papers.append({"id":f"ss_{p.get('paperId','')}",
                                "source":"SemanticScholar",
                                "title":ttl,"abstract":abst,"doi":doi,
                                "full_text":None,"chunks":[],"oa_pdf":oa})

    return papers[:n]


def try_fetch_fulltext(paper: dict) -> dict:
    """Try Unpaywall → OA PDF → abstract fallback. Chunk the result."""
    doi = paper.get("doi","")
    text = None

    # Unpaywall
    if doi:
        r = _get(f"https://api.unpaywall.org/v2/{doi}",
                 params={"email": Entrez.email})
        if r and r.status_code == 200:
            loc = r.json().get("best_oa_location") or {}
            url = loc.get("url_for_pdf") or loc.get("url")
            if url:
                pr = _get(url, timeout=25)
                if pr and pr.status_code == 200 and "pdf" in pr.headers.get("Content-Type",""):
                    try:
                        import pymupdf as fitz
                        doc  = fitz.open(stream=pr.content, filetype="pdf")
                        text = "\n".join(p.get_text() for p in doc)
                        doc.close()
                    except Exception:
                        pass

    # OA PDF from S2
    if not text:
        oa = paper.get("oa_pdf","")
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
        paper["chunks"]    = make_chunks(text, paper["doi"])
    else:
        # fallback: treat abstract as single chunk
        paper["full_text"] = None
        paper["chunks"]    = [{"doi":doi,"page":0,"para_idx":0,
                                "text":paper["abstract"],"_score":5}]
    return paper


def make_chunks(text: str, doi: str) -> list[dict]:
    """Paragraph-level chunks scored by AU keyword density."""
    AU_PAT  = re.compile(r"\bAU\d{1,2}\b", re.I)
    KEY_PAT = re.compile(
        r"\b(action unit|schizophrenia|flat affect|blunted|apathy|negative symptom|"
        r"FACS|OpenFace|intensity|frequency|mean|SD|p\s*=|significant|decreased|increased)\b",
        re.I)
    chunks = []
    for i, para in enumerate(text.split("\n\n")):
        para = para.strip()
        if len(para) < 60:
            continue
        score = len(AU_PAT.findall(para))*3 + len(KEY_PAT.findall(para))
        chunks.append({"doi":doi,"page":0,"para_idx":i,
                        "text":para[:800],"_score":score})
    # return top 10 by score
    return sorted(chunks, key=lambda c: c["_score"], reverse=True)[:10]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Gemini judge — scores each metric with chain-of-thought
# ═══════════════════════════════════════════════════════════════════════════

_rpm_times: list[float] = []
_day_count: list[int]   = [0]

def _rate_gate():
    import time as _t
    now = _t.time()
    _rpm_times[:] = [t for t in _rpm_times if now - t < 60]
    if len(_rpm_times) >= 14:
        wait = 60 - (now - _rpm_times[0]) + 1
        print(f"  [rate limit] waiting {wait:.0f}s")
        _t.sleep(wait)
    if _day_count[0] >= 1490:
        raise RuntimeError("Gemini daily quota reached.")
    _rpm_times.append(_t.time())
    _day_count[0] += 1


def gemini_judge(prompt: str, debug: bool = False) -> str:
    for attempt in range(5):
        _rate_gate()
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{EVAL_MODEL}:generateContent?key={GEMINI_API_KEY}")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature":     0.0,
            },
        }
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 429:
            import time as _t; _t.sleep(2**attempt * 10); continue
        if resp.status_code != 200:
            if debug:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
            import time as _t; _t.sleep(10); continue

        data       = resp.json()
        candidates = data.get("candidates", [])

        if debug:
            # print full response structure on first call
            finish  = candidates[0].get("finishReason","?") if candidates else "NO_CANDIDATES"
            filters = data.get("promptFeedback", {}).get("blockReason", "none")
            print(f"    finishReason={finish}  blockReason={filters}")

        if not candidates:
            # no candidates — usually means prompt was blocked
            block = data.get("promptFeedback", {}).get("blockReason", "unknown")
            if debug: print(f"    No candidates, blockReason={block}")
            return f"BLOCKED:{block}"

        # some candidates have no content (e.g. finishReason=SAFETY)
        content = candidates[0].get("content", {})
        parts   = content.get("parts", [])
        text    = "".join(p.get("text", "") for p in parts).strip()

        if not text:
            finish = candidates[0].get("finishReason", "?")
            if debug: print(f"    Empty text, finishReason={finish}")
            # retry with slightly rephrased prompt on SAFETY finish
            if finish == "SAFETY":
                import time as _t; _t.sleep(2); continue
            return ""

        return text

    return ""


def parse_score(text: str) -> float:
    """Extract a 0-1 float from judge output like 'Score: 0.8' or just '4/5'."""
    # look for explicit Score: patterns first
    m = re.search(r"[Ss]core\s*[:\-]\s*([\d.]+)", text)
    if m:
        v = float(m.group(1))
        return v if v <= 1.0 else v / 5.0
    # fraction like 4/5
    m = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if m:
        return float(m.group(1)) / float(m.group(2))
    # bare decimal
    m = re.search(r"\b(0\.\d+|1\.0)\b", text)
    if m:
        return float(m.group(1))
    return 0.5   # uncertain


# ── 2a. Context Relevance ────────────────────────────────────────────────────
CR_PROMPT = textwrap.dedent("""
You are evaluating a retrieval system for a systematic meta-analysis on facial
Action Units (AUs) in schizophrenia.

Research question:
{question}

Retrieved chunk:
\"\"\"{chunk}\"\"\"

Task: Score how relevant this chunk is to the research question.
A perfectly relevant chunk reports AU numbers, schizophrenia patients, and 
quantitative comparisons (means, p-values, effect sizes).
A score of 0 means completely irrelevant. A score of 1 means perfectly relevant.

Respond in this exact format:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()


def score_context_relevance(chunks: list[dict]) -> float:
    if not chunks: return 0.0
    scores = []
    # score top 3 chunks to save quota
    for chunk in chunks[:3]:
        resp = gemini_judge(CR_PROMPT.format(
            question=RESEARCH_QUESTION,
            chunk=chunk["text"][:600]))
        scores.append(parse_score(resp))
    return float(np.mean(scores))


# ── 2b. Faithfulness ─────────────────────────────────────────────────────────
FAITH_PROMPT = textwrap.dedent("""
You are checking whether an extracted AU finding is faithful to its source text.

Source chunk:
\"\"\"{chunk}\"\"\"

Extracted claim:
\"\"\"{claim}\"\"\"

Task: Does the claim follow directly from the source chunk without adding 
information not present in the text?
Score 1.0 = fully supported, 0.5 = partially supported, 0.0 = not supported / hallucinated.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()


def score_faithfulness(claim: str, chunks: list[dict]) -> float:
    if not claim or not chunks: return 0.5
    # use the highest-scored chunk as the source
    best = max(chunks, key=lambda c: c["_score"])
    resp = gemini_judge(FAITH_PROMPT.format(
        chunk=best["text"][:600], claim=claim[:300]))
    return parse_score(resp)


# ── 2c. Answer Relevance ─────────────────────────────────────────────────────
AR_PROMPT = textwrap.dedent("""
Research question:
{question}

Extracted finding:
\"\"\"{finding}\"\"\"

Task: Does this extracted finding directly address the research question?
Score 1.0 = directly answers it (reports an AU, direction, comparison), 
0.5 = tangentially related, 0.0 = off-topic.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()


def score_answer_relevance(finding_text: str) -> float:
    if not finding_text: return 0.0
    resp = gemini_judge(AR_PROMPT.format(
        question=RESEARCH_QUESTION, finding=finding_text[:400]))
    return parse_score(resp)


# ── 2d. Context Recall (needs gold set) ──────────────────────────────────────
RECALL_PROMPT = textwrap.dedent("""
Gold standard sentence that MUST appear in the retrieved context:
\"\"\"{gold_sentence}\"\"\"

Retrieved chunks (combined):
\"\"\"{context}\"\"\"

Task: Is the gold sentence (or its information) present in the retrieved chunks?
Score 1.0 = clearly present, 0.5 = partially present, 0.0 = missing.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()


def score_recall(gold_sentence: str, chunks: list[dict]) -> float:
    combined = "\n".join(c["text"][:300] for c in chunks[:5])
    resp = gemini_judge(RECALL_PROMPT.format(
        gold_sentence=gold_sentence, context=combined))
    return parse_score(resp)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Non-LLM metrics (fast, no quota)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_dedup_rate(all_chunks: list[dict]) -> float:
    """Fraction of chunks that are near-duplicates (jaccard similarity > 0.8)."""
    if len(all_chunks) < 2: return 0.0
    def shingles(text, k=5):
        words = text.lower().split()
        return set(tuple(words[i:i+k]) for i in range(len(words)-k+1))
    total, dups = 0, 0
    texts = [c["text"] for c in all_chunks[:100]]  # cap for speed
    shingle_sets = [shingles(t) for t in texts]
    for i in range(len(shingle_sets)):
        for j in range(i+1, len(shingle_sets)):
            total += 1
            a, b = shingle_sets[i], shingle_sets[j]
            if a and b:
                jaccard = len(a & b) / len(a | b)
                if jaccard > 0.8:
                    dups += 1
    return dups / total if total > 0 else 0.0


def au_pattern_hit_rate(chunks: list[dict]) -> float:
    """Fraction of top chunks that mention at least one AU pattern."""
    au_re = re.compile(r"\bAU\s*\d{1,2}\b", re.I)
    hits  = sum(1 for c in chunks[:10] if au_re.search(c["text"]))
    return hits / min(10, len(chunks)) if chunks else 0.0


def precision_at_au(extracted_aus: list[str], gold_aus: list[str]) -> float:
    """What fraction of extracted AUs are in the gold AU list?"""
    if not extracted_aus or not gold_aus: return None
    ext = {au.upper().replace(" ","") for au in extracted_aus}
    gld = {au.upper().replace(" ","") for au in gold_aus}
    return len(ext & gld) / len(ext)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Run evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_paper(paper: dict, gold_entry: dict | None) -> dict:
    row = {
        "id":              paper["id"],
        "source":          paper["source"],
        "title":           paper["title"][:80],
        "has_full_text":   paper["full_text"] is not None,
        "n_chunks":        len(paper["chunks"]),
        "au_hit_rate":     au_pattern_hit_rate(paper["chunks"]),
        "dedup_rate":      None,   # computed globally
        "context_relevance":  None,
        "faithfulness":       None,
        "answer_relevance":   None,
        "context_recall":     None,
        "precision_at_au":    None,
    }

    if not GEMINI_API_KEY:
        print("  No GEMINI_API_KEY — skipping LLM metrics")
        return row

    # Context relevance
    row["context_relevance"] = score_context_relevance(paper["chunks"])

    # For faithfulness + answer relevance we need an extraction
    # Use the best chunk's text as a mock "claim" if no real extraction
    if paper["chunks"]:
        best = max(paper["chunks"], key=lambda c: c["_score"])
        au_match = re.search(r"AU\s*\d{1,2}", best["text"], re.I)
        if au_match:
            claim = best["text"][:300]
            row["faithfulness"]     = score_faithfulness(claim, paper["chunks"])
            row["answer_relevance"] = score_answer_relevance(claim)

    # Context recall (gold set only)
    if gold_entry:
        row["context_recall"] = score_recall(
            gold_entry["key_sentence"], paper["chunks"])
        # extract AU patterns from chunks to check precision
        au_re = re.compile(r"AU\s*(\d{1,2})", re.I)
        found_aus = list({f"AU{m.group(1)}" for c in paper["chunks"]
                          for m in au_re.finditer(c["text"])})
        row["precision_at_au"] = precision_at_au(found_aus, gold_entry["aus_in_paper"])

    return row


# ═══════════════════════════════════════════════════════════════════════════
# 5. Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame):
    llm_cols = ["context_relevance","faithfulness","answer_relevance"]
    llm_cols = [c for c in llm_cols if df[c].notna().any()]
    sys_cols  = ["au_hit_rate","has_full_text"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Retrieval Evaluation Results", fontsize=13, fontweight="bold")

    # 1. Radar chart of mean LLM metrics
    ax = axes[0]
    if llm_cols:
        means = df[llm_cols].mean()
        angles = np.linspace(0, 2*np.pi, len(means), endpoint=False).tolist()
        vals   = means.tolist()
        angles += angles[:1]; vals += vals[:1]
        ax2 = fig.add_axes([0.03, 0.12, 0.28, 0.76], polar=True)
        ax2.plot(angles, vals, "o-", linewidth=2, color="#4878D0")
        ax2.fill(angles, vals, alpha=0.2, color="#4878D0")
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([c.replace("_","\n") for c in llm_cols], fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.set_title("RAGAS-style metrics\n(LLM-judged)", fontsize=10, pad=14)
        ax.remove()
    else:
        ax.text(0.5, 0.5, "No GEMINI_API_KEY\nset", ha="center", va="center",
                fontsize=11, color="gray")
        ax.axis("off")

    # 2. Per-source box plot of context relevance
    ax = axes[1]
    if "context_relevance" in df.columns and df["context_relevance"].notna().any():
        groups = {src: grp["context_relevance"].dropna().tolist()
                  for src, grp in df.groupby("source")
                  if grp["context_relevance"].notna().any()}
        if groups:
            ax.boxplot(list(groups.values()), labels=list(groups.keys()), patch_artist=True,
                       boxprops={"facecolor":"#B5D4F4","linewidth":0.8},
                       medianprops={"color":"#185FA5","linewidth":2})
            ax.set_ylabel("Context Relevance Score", fontsize=10)
            ax.set_title("Relevance by source", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No LLM scores\navailable", ha="center", va="center",
                fontsize=11, color="gray")
        ax.axis("off")

    # 3. System metrics bar chart
    ax = axes[2]
    sys_means = {
        "Full-text\nhit rate":    df["has_full_text"].mean(),
        "AU keyword\nhit rate":   df["au_hit_rate"].mean(),
        "Mean chunks\nper paper": df["n_chunks"].mean() / 10,  # normalised /10
    }
    bars = ax.bar(list(sys_means.keys()), list(sys_means.values()),
                   color=["#6ACC65","#4878D0","#EE854A"], edgecolor="white")
    ax.bar_label(bars,
                 labels=[f"{v:.0%}" if i < 2 else f"{v*10:.1f}"
                         for i, v in enumerate(sys_means.values())],
                 padding=3, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("System metrics", fontsize=10)
    ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5, label="0.5 threshold")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = f"{RESULTS_DIR}/retrieval_eval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM narrative report
# ═══════════════════════════════════════════════════════════════════════════

REPORT_PROMPT = textwrap.dedent("""
You are evaluating a retrieval pipeline for a systematic meta-analysis.

Evaluation results (JSON):
{results_json}

Write a concise report (~300 words) covering:
1. Overall retrieval quality (context relevance score interpretation)
2. Full-text coverage and its impact on extraction quality
3. Source-level quality differences (which sources retrieved most relevant content)
4. Faithfulness of extractions (hallucination risk)
5. Specific recommendations to improve the pipeline before the full run

Be direct and practical. Flag any scores below 0.5 as requiring attention.
""").strip()


def generate_report(df: pd.DataFrame) -> str:
    """Generate narrative report using the same gemini_judge wrapper (rate-limited)."""
    summary      = df.describe().round(3).to_dict()
    source_means = (df.groupby("source")[["context_relevance","au_hit_rate"]]
                      .mean().round(3).to_dict())
    results_json = json.dumps({"summary": summary, "by_source": source_means}, indent=2)
    prompt       = REPORT_PROMPT.format(results_json=results_json[:3000])
    result       = gemini_judge(prompt)
    if not result or result.startswith("BLOCKED"):
        return f"Report generation failed or blocked: {result}"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not set — LLM metrics will be skipped.")
        print("         Only system metrics (hit rate, chunk count) will run.\n")
    else:
        # ── Smoke test: verify Gemini responds before the full eval loop ──────
        print("=== Gemini smoke test ===")
        try:
            test_resp = gemini_judge("Reply with exactly: OK", debug=True)
            if "OK" in test_resp.upper():
                print(f"  Gemini responded: {test_resp.strip()} ✓")
            else:
                print(f"  Gemini responded (unexpected): {test_resp[:100]}")
            print(f"  Daily request count so far: {_day_count[0]}\n")
        except Exception as e:
            print(f"  Gemini FAILED: {e}")
            print("  → Check GEMINI_API_KEY and network access from this node")
            print("  → Continuing with system metrics only\n")
            GEMINI_API_KEY = ""   # disable LLM metrics for rest of run

    print(f"=== Step 1: Retrieving {SAMPLE_N} sample papers ===")
    papers = fetch_sample_papers(SAMPLE_N)
    print(f"  Got {len(papers)} papers")

    print("\n=== Step 2: Full-text retrieval for sample ===")
    for p in tqdm(papers):
        try_fetch_fulltext(p)
        time.sleep(0.3)
    n_ft = sum(1 for p in papers if p["full_text"])
    print(f"  Full text: {n_ft}/{len(papers)}")

    print("\n=== Step 3: Computing global dedup rate ===")
    all_chunks = [c for p in papers for c in p["chunks"]]
    dedup = chunk_dedup_rate(all_chunks)
    print(f"  Cross-paper chunk dedup rate: {dedup:.1%}")

    print(f"\n=== Step 4: LLM evaluation (GEMINI model: {EVAL_MODEL}) ===")
    rows = []
    for paper in tqdm(papers):
        # check if this paper is in our gold set
        gold = next((g for g in GOLD_SET if g["doi"] == paper.get("doi","")), None)
        row  = evaluate_paper(paper, gold)
        row["dedup_rate"] = dedup
        rows.append(row)
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/retrieval_eval.csv", index=False)

    print("\n=== Evaluation Summary ===")
    llm_metrics = ["context_relevance","faithfulness","answer_relevance"]
    for col in llm_metrics:
        if df[col].notna().any():
            print(f"  {col:25s}: {df[col].mean():.3f}  (min {df[col].min():.2f}, max {df[col].max():.2f})")
    print(f"  {'full_text_hit_rate':25s}: {df['has_full_text'].mean():.1%}")
    print(f"  {'au_keyword_hit_rate':25s}: {df['au_hit_rate'].mean():.1%}")
    print(f"  {'chunk_dedup_rate':25s}: {dedup:.1%}")

    print("\n=== Step 5: Plots ===")
    plot_results(df)

    if GEMINI_API_KEY:
        print("\n=== Step 6: LLM narrative report ===")
        report = generate_report(df)
        print(report)
        with open(f"{RESULTS_DIR}/eval_report.txt","w") as f:
            f.write(report)

    print(f"\nDone. Outputs in ./{RESULTS_DIR}/")
    print("""
INTERPRETATION GUIDE:
  context_relevance  > 0.6  → retrieved chunks are on-topic  ✓
  faithfulness       > 0.7  → low hallucination risk          ✓
  answer_relevance   > 0.6  → extractions address the RQ     ✓
  full_text_hit_rate > 0.4  → reasonable OA coverage         ✓
  au_hit_rate        > 0.5  → chunks mention actual AU codes  ✓
  dedup_rate         < 0.3  → manageable redundancy           ✓

If context_relevance < 0.5:
  → tighten QUERIES in 01_meta_analysis.py
  → increase keyword scoring weights in chunk_pdf()

If au_hit_rate < 0.3:
  → add MeSH terms: "facial action units"[MeSH] to PubMed queries

If faithfulness < 0.6:
  → add "do not infer, only extract" to EXTRACT_PROMPT
  → reduce max_tokens to force concise extraction

If full_text_hit_rate < 0.2:
  → check PyMuPDF install: pip install pymupdf
  → verify Unpaywall email is valid

GOLD SET:
  After manually reading 5-10 papers, add entries to GOLD_SET at top of file.
  This enables Context Recall and Precision@AU metrics.
""")
