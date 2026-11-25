"""
00_retrieval_eval.py
--------------------
Evaluates the retrieval + extraction pipeline BEFORE running the full meta-analysis.

Metrics (reference-free — no manual labels needed for first run):
  1. Context Relevance   — are retrieved chunks about AUs/schizophrenia?
  2. Faithfulness        — does the extracted AU claim follow from the chunk?
  3. Answer Relevance    — is the extraction on-topic for our research question?
  4. Chunk Dedup Rate    — redundancy across sources
  5. Full-text Hit Rate  — how many papers got full text vs abstract-only
  6. AU Keyword Hit Rate — chunks that mention actual AU codes

Optional (needs gold set — see GOLD_SET below):
  7. Context Recall      — did we retrieve the key passage?
  8. Precision@AU        — are extracted AUs in the paper's real AU list?

LLM judge: Gemini 2.5 Flash via official google-genai SDK (>= 2.3.0)

Outputs in results/eval/:
  retrieval_eval.csv
  retrieval_eval.png
  eval_report.txt

Setup:
  pip install google-genai>=2.3.0
  export GEMINI_API_KEY=...
"""

import os, json, time, hashlib, re, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from Bio import Entrez

RESULTS_DIR = "results/eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    # {
    #     "doi":          "10.1016/j.schres.2020.01.001",
    #     "aus_in_paper": ["AU6","AU12","AU25"],
    #     "key_sentence": "AU12 intensity was significantly lower in SZ...",
    #     "key_page":     4,
    # },
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Gemini SDK client — single shared instance, rate-limited
# ═══════════════════════════════════════════════════════════════════════════

_gemini_model  = None   # initialised lazily after key check
_rpm_times: list[float] = []
_day_count: list[int]   = [0]


def _init_gemini():
    """Initialise the google-genai client once."""
    global _gemini_model
    if _gemini_model is not None:
        return
    from google import genai
    from google.genai import types
    _client = genai.Client(api_key=GEMINI_API_KEY)
    # store client and model name, generate_content called per-request
    _gemini_model = (_client, EVAL_MODEL)


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
    """Call Gemini via official SDK with rate limiting and retry."""
    _init_gemini()
    _init_gemini()
    client, model_name = _gemini_model
    from google.genai import types as _types
    for attempt in range(5):
        _rate_gate()
        try:
            response = client.models.generate_content(
                model=model_name,
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
                if debug:
                    print("    No candidates returned")
                return "BLOCKED:NO_CANDIDATES"

            text = (response.text or "").strip()

            if not text:
                finish = response.candidates[0].finish_reason
                if debug:
                    print(f"    Empty response, finish_reason={finish}")
                if str(finish) == "SAFETY":
                    time.sleep(2)
                    continue
                return ""

            return text

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 2 ** attempt * 10
                print(f"    [429] backing off {wait}s")
                time.sleep(wait)
            elif "503" in err:
                time.sleep(30)
            else:
                if debug:
                    print(f"    SDK error attempt {attempt+1}: {e}")
                time.sleep(5)

    return ""


def parse_score(text: str) -> float:
    """Extract a 0-1 float from judge output."""
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
# 2. Lightweight retrieval (sample only)
# ═══════════════════════════════════════════════════════════════════════════

def _get(url, params=None, timeout=12):
    try:
        return requests.get(url, params=params,
                            headers={"User-Agent": BROWSER_UA}, timeout=timeout)
    except Exception:
        return None


def fetch_sample_papers(n: int) -> list[dict]:
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
            ttl  = str(art.get("ArticleTitle", ""))
            abst = " ".join(str(p) for p in art.get("Abstract", {}).get("AbstractText", []))
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
             params={"query": "facial action units schizophrenia", "limit": n,
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
                                "title": ttl, "abstract": abst, "doi": doi,
                                "full_text": None, "chunks": [], "oa_pdf": oa})

    return papers[:n]


def make_chunks(text: str, doi: str) -> list[dict]:
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
        score = len(AU_PAT.findall(para)) * 3 + len(KEY_PAT.findall(para))
        chunks.append({"doi": doi, "page": 0, "para_idx": i,
                        "text": para[:800], "_score": score})
    return sorted(chunks, key=lambda c: c["_score"], reverse=True)[:10]


def try_fetch_fulltext(paper: dict) -> dict:
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
                if pr and pr.status_code == 200 and "pdf" in pr.headers.get("Content-Type", ""):
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
                                "text": paper["abstract"], "_score": 5}]
    return paper


# ═══════════════════════════════════════════════════════════════════════════
# 3. RAGAS-style scoring prompts
# ═══════════════════════════════════════════════════════════════════════════

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
Score 0 = completely irrelevant, Score 1 = perfectly relevant.

Respond in this exact format:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

FAITH_PROMPT = textwrap.dedent("""
You are checking whether an extracted AU finding is faithful to its source text.

Source chunk:
\"\"\"{chunk}\"\"\"

Extracted claim:
\"\"\"{claim}\"\"\"

Task: Does the claim follow directly from the source chunk without adding
information not present in the text?
Score 1.0 = fully supported, 0.5 = partially, 0.0 = not supported / hallucinated.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

AR_PROMPT = textwrap.dedent("""
Research question:
{question}

Extracted finding:
\"\"\"{finding}\"\"\"

Task: Does this extracted finding directly address the research question?
Score 1.0 = directly answers it, 0.5 = tangentially related, 0.0 = off-topic.

Respond:
Reasoning: <one sentence>
Score: <0.0 to 1.0>
""").strip()

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

REPORT_PROMPT = textwrap.dedent("""
You are evaluating a retrieval pipeline for a systematic meta-analysis on
facial Action Units in schizophrenia.

Evaluation results (JSON):
{results_json}

Write a concise report (~300 words) covering:
1. Overall retrieval quality (context relevance score interpretation)
2. Full-text coverage and its impact on extraction quality
3. Source-level quality differences
4. Faithfulness of extractions (hallucination risk)
5. Specific recommendations to improve the pipeline before the full run

Be direct and practical. Flag any scores below 0.5 as requiring attention.
""").strip()


def score_context_relevance(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    scores = []
    for chunk in chunks[:3]:
        resp = gemini_judge(CR_PROMPT.format(
            question=RESEARCH_QUESTION, chunk=chunk["text"][:600]))
        scores.append(parse_score(resp))
    return float(np.mean(scores))


def score_faithfulness(claim: str, chunks: list[dict]) -> float:
    if not claim or not chunks:
        return 0.5
    best = max(chunks, key=lambda c: c["_score"])
    resp = gemini_judge(FAITH_PROMPT.format(
        chunk=best["text"][:600], claim=claim[:300]))
    return parse_score(resp)


def score_answer_relevance(finding_text: str) -> float:
    if not finding_text:
        return 0.0
    resp = gemini_judge(AR_PROMPT.format(
        question=RESEARCH_QUESTION, finding=finding_text[:400]))
    return parse_score(resp)


def score_recall(gold_sentence: str, chunks: list[dict]) -> float:
    combined = "\n".join(c["text"][:300] for c in chunks[:5])
    resp = gemini_judge(RECALL_PROMPT.format(
        gold_sentence=gold_sentence, context=combined))
    return parse_score(resp)


def generate_report(df: pd.DataFrame) -> str:
    """Generate narrative report — uses same gemini_judge, same model, rate-limited."""
    summary      = df.describe().round(3).to_dict()
    source_means = (df.groupby("source")[["context_relevance", "au_hit_rate"]]
                      .mean().round(3).to_dict())
    results_json = json.dumps({"summary": summary, "by_source": source_means}, indent=2)
    result = gemini_judge(REPORT_PROMPT.format(results_json=results_json[:3000]))
    if not result or result.startswith("BLOCKED"):
        return f"Report generation failed: {result}"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Non-LLM metrics
# ═══════════════════════════════════════════════════════════════════════════

def chunk_dedup_rate(all_chunks: list[dict]) -> float:
    if len(all_chunks) < 2:
        return 0.0
    def shingles(text, k=5):
        words = text.lower().split()
        return set(tuple(words[i:i+k]) for i in range(len(words)-k+1))
    total, dups = 0, 0
    texts        = [c["text"] for c in all_chunks[:100]]
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
    au_re = re.compile(r"\bAU\s*\d{1,2}\b", re.I)
    hits  = sum(1 for c in chunks[:10] if au_re.search(c["text"]))
    return hits / min(10, len(chunks)) if chunks else 0.0


def precision_at_au(extracted_aus: list[str], gold_aus: list[str]):
    if not extracted_aus or not gold_aus:
        return None
    ext = {au.upper().replace(" ", "") for au in extracted_aus}
    gld = {au.upper().replace(" ", "") for au in gold_aus}
    return len(ext & gld) / len(ext)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_paper(paper: dict, gold_entry: dict | None) -> dict:
    row = {
        "id":                 paper["id"],
        "source":             paper["source"],
        "title":              paper["title"][:80],
        "has_full_text":      paper["full_text"] is not None,
        "n_chunks":           len(paper["chunks"]),
        "au_hit_rate":        au_pattern_hit_rate(paper["chunks"]),
        "dedup_rate":         None,
        "context_relevance":  None,
        "faithfulness":       None,
        "answer_relevance":   None,
        "context_recall":     None,
        "precision_at_au":    None,
    }

    if not GEMINI_API_KEY:
        return row

    row["context_relevance"] = score_context_relevance(paper["chunks"])

    if paper["chunks"]:
        best = max(paper["chunks"], key=lambda c: c["_score"])
        if re.search(r"AU\s*\d{1,2}", best["text"], re.I):
            claim                   = best["text"][:300]
            row["faithfulness"]     = score_faithfulness(claim, paper["chunks"])
            row["answer_relevance"] = score_answer_relevance(claim)

    if gold_entry:
        row["context_recall"] = score_recall(
            gold_entry["key_sentence"], paper["chunks"])
        au_re     = re.compile(r"AU\s*(\d{1,2})", re.I)
        found_aus = list({f"AU{m.group(1)}" for c in paper["chunks"]
                          for m in au_re.finditer(c["text"])})
        row["precision_at_au"] = precision_at_au(found_aus, gold_entry["aus_in_paper"])

    return row


# ═══════════════════════════════════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(df: pd.DataFrame):
    llm_cols = [c for c in ["context_relevance", "faithfulness", "answer_relevance"]
                if df[c].notna().any()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Retrieval Evaluation Results", fontsize=13, fontweight="bold")

    # Radar
    ax = axes[0]
    if llm_cols:
        means  = df[llm_cols].mean()
        angles = np.linspace(0, 2*np.pi, len(means), endpoint=False).tolist()
        vals   = means.tolist()
        angles += angles[:1]; vals += vals[:1]
        ax2 = fig.add_axes([0.03, 0.12, 0.28, 0.76], polar=True)
        ax2.plot(angles, vals, "o-", linewidth=2, color="#4878D0")
        ax2.fill(angles, vals, alpha=0.2, color="#4878D0")
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([c.replace("_", "\n") for c in llm_cols], fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.set_title("RAGAS-style metrics\n(LLM-judged)", fontsize=10, pad=14)
        ax.remove()
    else:
        ax.text(0.5, 0.5, "No GEMINI_API_KEY\nset", ha="center", va="center",
                fontsize=11, color="gray")
        ax.axis("off")

    # Per-source box
    ax = axes[1]
    if "context_relevance" in df.columns and df["context_relevance"].notna().any():
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
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No LLM scores\navailable", ha="center", va="center",
                fontsize=11, color="gray")
        ax.axis("off")

    # System metrics
    ax = axes[2]
    sys_means = {
        "Full-text\nhit rate":    df["has_full_text"].mean(),
        "AU keyword\nhit rate":   df["au_hit_rate"].mean(),
        "Mean chunks\nper paper": df["n_chunks"].mean() / 10,
    }
    bars = ax.bar(list(sys_means.keys()), list(sys_means.values()),
                   color=["#6ACC65", "#4878D0", "#EE854A"], edgecolor="white")
    ax.bar_label(bars,
                 labels=[f"{v:.0%}" if i < 2 else f"{v*10:.1f}"
                         for i, v in enumerate(sys_means.values())],
                 padding=3, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("System metrics", fontsize=10)
    ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5, label="0.5 threshold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = f"{RESULTS_DIR}/retrieval_eval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not set — LLM metrics will be skipped.")
        print("         Only system metrics (hit rate, chunk count) will run.\n")
    else:
        print(f"=== Gemini smoke test (model: {EVAL_MODEL}) ===")
        try:
            test_resp = gemini_judge("Reply with exactly the word: OK", debug=True)
            if "OK" in test_resp.upper():
                print(f"  Response: {test_resp.strip()} ✓")
            else:
                print(f"  Unexpected response: {test_resp[:100]}")
            print(f"  Requests used so far: {_day_count[0]}\n")
        except Exception as e:
            print(f"  Gemini FAILED: {e}")
            print("  → Check GEMINI_API_KEY and outbound network from this node\n")
            GEMINI_API_KEY = ""

    print(f"=== Step 1: Retrieving {SAMPLE_N} sample papers ===")
    papers = fetch_sample_papers(SAMPLE_N)
    print(f"  Got {len(papers)} papers")

    print("\n=== Step 2: Full-text retrieval ===")
    for p in tqdm(papers):
        try_fetch_fulltext(p)
        time.sleep(0.3)
    n_ft = sum(1 for p in papers if p["full_text"])
    print(f"  Full text: {n_ft}/{len(papers)}")

    print("\n=== Step 3: Dedup rate ===")
    all_chunks = [c for p in papers for c in p["chunks"]]
    dedup = chunk_dedup_rate(all_chunks)
    print(f"  Cross-paper chunk dedup rate: {dedup:.1%}")

    print(f"\n=== Step 4: LLM evaluation (model: {EVAL_MODEL}) ===")
    rows = []
    for paper in tqdm(papers):
        gold = next((g for g in GOLD_SET if g["doi"] == paper.get("doi", "")), None)
        row  = evaluate_paper(paper, gold)
        row["dedup_rate"] = dedup
        rows.append(row)
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/retrieval_eval.csv", index=False)

    print("\n=== Evaluation Summary ===")
    for col in ["context_relevance", "faithfulness", "answer_relevance"]:
        if df[col].notna().any():
            print(f"  {col:25s}: {df[col].mean():.3f}  "
                  f"(min {df[col].min():.2f}, max {df[col].max():.2f})")
    print(f"  {'full_text_hit_rate':25s}: {df['has_full_text'].mean():.1%}")
    print(f"  {'au_keyword_hit_rate':25s}: {df['au_hit_rate'].mean():.1%}")
    print(f"  {'chunk_dedup_rate':25s}: {dedup:.1%}")

    print("\n=== Step 5: Plots ===")
    plot_results(df)

    if GEMINI_API_KEY:
        print("\n=== Step 6: Narrative report ===")
        report = generate_report(df)
        print(report)
        with open(f"{RESULTS_DIR}/eval_report.txt", "w") as f:
            f.write(report)

    print(f"\nDone. Outputs in ./{RESULTS_DIR}/")
    print("""
INTERPRETATION GUIDE:
  context_relevance  > 0.6  → retrieved chunks are on-topic         ✓
  faithfulness       > 0.7  → low hallucination risk                ✓
  answer_relevance   > 0.6  → extractions address the RQ            ✓
  full_text_hit_rate > 0.4  → reasonable OA coverage                ✓
  au_hit_rate        > 0.5  → chunks mention actual AU codes         ✓
  dedup_rate         < 0.3  → manageable redundancy                 ✓

If context_relevance < 0.5 → tighten QUERIES in 01_meta_analysis.py
If au_hit_rate < 0.3       → add MeSH: "facial action units"[MeSH] to PubMed queries
If faithfulness < 0.6      → add "do not infer, only extract" to EXTRACT_PROMPT
If full_text_hit_rate < 0.2 → pip install pymupdf, check Unpaywall email
""")
