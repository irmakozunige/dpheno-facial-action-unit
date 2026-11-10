"""
01_meta_analysis.py
--------------------
Multi-source literature retrieval + LLM extraction + forest plots.

Sources:
  - PubMed       (via Biopython Entrez)
  - Semantic Scholar (official API — covers Nature, Frontiers, Wiley, etc.)
  - Google Scholar   (via scholarly — broad coverage incl. ScienceDirect)
  - CrossRef         (DOI metadata — Nature, Wiley, Frontiers, Elsevier)

Outputs in results/meta/:
  raw_extractions.json
  au_long.csv               one row per paper x AU
  meta_results.csv          pooled Hedges g per AU
  forest_AU**.png           one forest plot per AU with >=2 studies
  bubble_summary.png        AU evidence overview
  direction_heatmap.png     study x AU direction matrix
  meta_analysis_summary.txt LLM narrative

Requirements:
  pip install anthropic biopython requests scholarly semanticscholar
              pandas numpy matplotlib scipy tqdm
  ANTHROPIC_API_KEY env var
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
import anthropic
from Bio import Entrez

# ── Config ─────────────────────────────────────────────────────────────────
Entrez.email = "your@email.com"   # required by NCBI
MAX_PAPERS   = 100
RESULTS_DIR  = "results/meta"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOKEN = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=TOKEN)

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

# ═══════════════════════════════════════════════════════════════════════════
# 1. RETRIEVAL — four sources
# ═══════════════════════════════════════════════════════════════════════════

# ── 1a. PubMed ──────────────────────────────────────────────────────────────
def search_pubmed(query: str, max_results: int = 20) -> list[dict]:
    try:
        handle  = Entrez.esearch(db="pubmed", term=query,
                                 retmax=max_results, sort="relevance")
        ids     = Entrez.read(handle)["IdList"]
        handle.close()
        if not ids:
            return []
        handle  = Entrez.efetch(db="pubmed", id=ids,
                                rettype="abstract", retmode="xml")
        records = Entrez.read(handle)["PubmedArticle"]
        handle.close()
    except Exception as e:
        print(f"    PubMed error: {e}")
        return []

    papers = []
    for rec in records:
        try:
            art      = rec["MedlineCitation"]["Article"]
            title    = str(art.get("ArticleTitle", ""))
            abstract = " ".join(str(p) for p in
                                art.get("Abstract", {}).get("AbstractText", []))
            year     = str(art.get("Journal", {})
                             .get("JournalIssue", {})
                             .get("PubDate", {})
                             .get("Year", "?"))
            authors  = art.get("AuthorList", [])
            first_au = str(authors[0].get("LastName", "")) if authors else ""
            pmid     = str(rec["MedlineCitation"]["PMID"])
            journal  = str(art.get("Journal", {}).get("Title", ""))
            if abstract.strip():
                papers.append({"id": f"pmid_{pmid}", "source": "PubMed",
                                "first_author": first_au, "year": year,
                                "title": title, "abstract": abstract,
                                "journal": journal})
        except Exception:
            continue
    return papers


# ── 1b. Semantic Scholar ────────────────────────────────────────────────────
def search_semantic_scholar(query: str, max_results: int = 20) -> list[dict]:
    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,year,authors,externalIds,venue",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"    Semantic Scholar error: {e}")
        return []

    papers = []
    for p in data:
        abstract = p.get("abstract") or ""
        if not abstract.strip():
            continue
        authors  = p.get("authors", [])
        first_au = authors[0].get("name", "").split()[-1] if authors else ""
        ss_id    = p.get("paperId", "")
        papers.append({
            "id":           f"ss_{ss_id}",
            "source":       "SemanticScholar",
            "first_author": first_au,
            "year":         str(p.get("year", "?")),
            "title":        p.get("title", ""),
            "abstract":     abstract,
            "journal":      p.get("venue", ""),
        })
    return papers


# ── 1c. Google Scholar (via scholarly) ─────────────────────────────────────
def search_google_scholar(query: str, max_results: int = 10) -> list[dict]:
    try:
        from scholarly import scholarly as sch
        results = sch.search_pubs(query)
    except ImportError:
        print("    scholarly not installed — skipping Google Scholar")
        return []
    except Exception as e:
        print(f"    Google Scholar error: {e}")
        return []

    papers = []
    for i, pub in enumerate(results):
        if i >= max_results:
            break
        try:
            bib      = pub.get("bib", {})
            abstract = bib.get("abstract", "") or ""
            title    = bib.get("title", "") or ""
            if not abstract.strip() or not title.strip():
                continue
            authors  = bib.get("author", "")
            first_au = str(authors).split(" and ")[0].split(",")[0].strip() \
                       if authors else ""
            year     = str(bib.get("pub_year", "?"))
            uid      = hashlib.md5(title.encode()).hexdigest()[:10]
            papers.append({
                "id":           f"gs_{uid}",
                "source":       "GoogleScholar",
                "first_author": first_au,
                "year":         year,
                "title":        title,
                "abstract":     abstract,
                "journal":      bib.get("venue", ""),
            })
            time.sleep(1.5)   # be polite — scholar rate-limits aggressively
        except Exception:
            continue
    return papers


# ── 1d. CrossRef (Nature, Wiley, Frontiers, Elsevier DOI metadata) ──────────
def search_crossref(query: str, max_results: int = 20) -> list[dict]:
    url    = "https://api.crossref.org/works"
    params = {
        "query":            query,
        "rows":             max_results,
        "select":           "DOI,title,abstract,author,published,container-title",
        "mailto":           "your@email.com",
    }
    # filter to known high-relevance publishers
    publisher_filters = [
        "Nature Publishing Group",
        "Frontiers Media SA",
        "Wiley",
        "Elsevier",
        "SAGE Publications",
        "Springer",
    ]
    try:
        resp  = requests.get(url, params=params, timeout=15)
        items = resp.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"    CrossRef error: {e}")
        return []

    papers = []
    for item in items:
        abstract = item.get("abstract", "") or ""
        # CrossRef abstracts often have JATS XML tags — strip them
        abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""
        if not abstract or len(abstract) < 80:
            continue
        authors  = item.get("author", [])
        first_au = authors[0].get("family", "") if authors else ""
        pub_date = item.get("published", {}).get("date-parts", [[None]])[0]
        year     = str(pub_date[0]) if pub_date and pub_date[0] else "?"
        doi      = item.get("DOI", "")
        journal  = (item.get("container-title") or [""])[0]
        papers.append({
            "id":           f"cr_{doi.replace('/','_')}",
            "source":       "CrossRef",
            "first_author": first_au,
            "year":         year,
            "title":        title,
            "abstract":     abstract,
            "journal":      journal,
        })
    return papers


# ── Collect from all sources ────────────────────────────────────────────────
def collect_all_papers() -> list[dict]:
    seen, papers = set(), []

    source_fns = [
        ("PubMed",           search_pubmed,          20),
        ("SemanticScholar",  search_semantic_scholar, 20),
        ("GoogleScholar",    search_google_scholar,   10),
        ("CrossRef",         search_crossref,         20),
    ]

    for source_name, fn, per_query_max in source_fns:
        print(f"\n  [{source_name}]")
        for q in QUERIES:
            print(f"    → {q}")
            try:
                results = fn(q, max_results=per_query_max)
            except Exception as e:
                print(f"      ⚠ {e}")
                results = []
            new = 0
            for p in results:
                # de-duplicate by title fingerprint
                fp = hashlib.md5(p["title"].lower().encode()).hexdigest()
                if fp not in seen:
                    seen.add(fp)
                    papers.append(p)
                    new += 1
            print(f"      +{new} new  (total {len(papers)})")
            time.sleep(0.4)

    print(f"\n  Total unique papers: {len(papers)}")
    return papers[:MAX_PAPERS]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-paper extraction via Claude
# ═══════════════════════════════════════════════════════════════════════════
EXTRACT_PROMPT = textwrap.dedent("""
You are a clinical research assistant doing a systematic meta-analysis.
Given a paper abstract, extract EXACTLY this JSON — no preamble, no fences:

{
  "aus_studied": ["AU4", "AU12"],
  "findings": {
    "AU4":  {"direction": "decreased", "mean_sz": 1.2, "sd_sz": 0.4, "mean_hc": 2.1, "sd_hc": 0.5},
    "AU12": {"direction": "decreased", "mean_sz": null, "sd_sz": null, "mean_hc": null, "sd_hc": null}
  },
  "n_sz": 30,
  "n_hc": 28,
  "target_measure": "negative symptoms",
  "method": "OpenFace",
  "notes": ""
}

Rules:
- direction: "decreased" | "increased" | "no difference" | null
- mean_sz/sd_sz: AU intensity in schizophrenia group (null if not reported)
- mean_hc/sd_hc: AU intensity in healthy controls (null if not reported)
- n_sz / n_hc: group sizes (null if not reported)
- target_measure: "apathy" | "negative symptoms" | "both" | "other"
- method: "OpenFace" | "FACS" | "other" | null
- If the paper has no schizophrenia-control comparison, return aus_studied: []
""").strip()


def extract_paper(paper: dict) -> dict:
    text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user",
                        "content": f"{EXTRACT_PROMPT}\n\n---\n{text}"}]
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Effect sizes (Hedges' g)
# ═══════════════════════════════════════════════════════════════════════════
def hedges_g(mean1, sd1, n1, mean2, sd2, n2):
    try:
        n1, n2 = float(n1), float(n2)
        mean1, mean2 = float(mean1), float(mean2)
        sd1, sd2 = float(sd1), float(sd2)
        if any(math.isnan(v) or v <= 0 for v in [n1, n2, sd1, sd2]):
            return None, None
        sp  = math.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
        if sp == 0:
            return None, None
        d   = (mean1 - mean2) / sp
        df  = n1 + n2 - 2
        J   = 1 - (3 / (4*df - 1))
        g   = J * d
        se  = math.sqrt((n1+n2)/(n1*n2) + g**2/(2*(n1+n2-2)))
        return g, se
    except Exception:
        return None, None


def build_long_table(papers, extractions) -> pd.DataFrame:
    rows = []
    for paper, ext in zip(papers, extractions):
        if "error" in ext or not ext.get("aus_studied"):
            continue
        label = f"{paper['first_author']} {paper['year']}" \
                if paper.get("first_author") else f"Paper {paper['id'][-6:]}"
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
                "source":         paper.get("source", ""),
                "journal":        paper.get("journal", ""),
                "au":             au.upper().strip(),
                "direction":      info.get("direction"),
                "mean_sz":        info.get("mean_sz"),
                "sd_sz":          info.get("sd_sz"),
                "mean_hc":        info.get("mean_hc"),
                "sd_hc":          info.get("sd_hc"),
                "n_sz":           ext.get("n_sz"),
                "n_hc":           ext.get("n_hc"),
                "hedges_g":       g,
                "se_g":           se_g,
                "target_measure": ext.get("target_measure"),
                "method":         ext.get("method"),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DerSimonian–Laird random-effects meta-analysis
# ═══════════════════════════════════════════════════════════════════════════
def dersimonian_laird(gs, ses):
    gs, ses = np.array(gs, dtype=float), np.array(ses, dtype=float)
    ws      = 1 / ses**2
    g_fe    = np.sum(ws * gs) / np.sum(ws)
    Q       = np.sum(ws * (gs - g_fe)**2)
    k       = len(gs)
    df_q    = k - 1
    p_Q     = 1 - stats.chi2.cdf(Q, df_q)
    C       = np.sum(ws) - np.sum(ws**2) / np.sum(ws)
    tau2    = max(0.0, (Q - df_q) / C)
    ws_re   = 1 / (ses**2 + tau2)
    g_re    = np.sum(ws_re * gs) / np.sum(ws_re)
    se_re   = math.sqrt(1 / np.sum(ws_re))
    ci_lo   = g_re - 1.96 * se_re
    ci_hi   = g_re + 1.96 * se_re
    I2      = max(0.0, (Q - df_q) / Q * 100) if Q > 0 else 0.0
    return dict(pooled_g=g_re, se=se_re, ci_lo=ci_lo, ci_hi=ci_hi,
                I2=I2, Q=Q, p_Q=p_Q, tau2=tau2, k=k)


def run_meta(long_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for au, grp in long_df.dropna(subset=["hedges_g","se_g"]).groupby("au"):
        if len(grp) < 2:
            continue
        res = dersimonian_laird(grp["hedges_g"].values, grp["se_g"].values)
        results.append({"au": au, "n_studies": res["k"],
                         "total_n_sz": grp["n_sz"].dropna().sum(), **res})
    return pd.DataFrame(results).sort_values("n_studies", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Forest plots
# ═══════════════════════════════════════════════════════════════════════════
DIR_COLORS = {
    "decreased":     "#2166AC",
    "increased":     "#D73027",
    "no difference": "#888888",
    None:            "#AAAAAA",
}
SOURCE_MARKERS = {
    "PubMed":          "s",
    "SemanticScholar": "o",
    "GoogleScholar":   "D",
    "CrossRef":        "^",
}


def forest_plot(au: str, grp: pd.DataFrame, meta: dict):
    es_rows  = grp.dropna(subset=["hedges_g","se_g"]).copy()
    dir_rows = grp[grp["hedges_g"].isna()].copy()

    n_total  = len(es_rows) + (1 if len(dir_rows) > 0 else 0) + (2 if meta else 0)
    fig_h    = max(5, n_total * 0.55 + 2.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h),
                              gridspec_kw={"width_ratios": [3, 2.5], "wspace": 0.02})
    ax_f, ax_t = axes

    y = 0

    # ── studies with effect sizes ──────────────────────────────────────────
    for _, row in es_rows.iterrows():
        ci_lo = row["hedges_g"] - 1.96 * row["se_g"]
        ci_hi = row["hedges_g"] + 1.96 * row["se_g"]
        col   = DIR_COLORS.get(row["direction"], "#555")
        mrk   = SOURCE_MARKERS.get(row["source"], "s")
        ax_f.plot([ci_lo, ci_hi], [y, y], color=col, lw=1.6, zorder=2)
        try:
            sz = max(25, min(220, float(row["n_sz"]) * 3))
        except Exception:
            sz = 50
        ax_f.scatter([row["hedges_g"]], [y], s=sz, color=col,
                      marker=mrk, zorder=3, edgecolors="white", linewidths=0.5)
        ci_str = f"{row['hedges_g']:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]"
        ax_t.text(0.01, y, row["label"],       va="center", ha="left",  fontsize=8)
        ax_t.text(0.50, y, row.get("source",""), va="center", ha="center", fontsize=7,
                  color="#666666")
        ax_t.text(0.63, y, str(int(row["n_sz"])) if pd.notna(row["n_sz"]) else "?",
                  va="center", ha="center", fontsize=8)
        ax_t.text(0.75, y, str(int(row["n_hc"])) if pd.notna(row["n_hc"]) else "?",
                  va="center", ha="center", fontsize=8)
        ax_t.text(0.99, y, ci_str, va="center", ha="right", fontsize=7.5,
                  color=col, fontweight="bold")
        y -= 1

    # ── direction-only ─────────────────────────────────────────────────────
    if len(dir_rows) > 0:
        y -= 0.3
        ax_f.axhline(y + 0.65, color="#CCCCCC", lw=0.6, ls="--")
        ax_t.text(0.01, y, "Direction only (no quantitative data):",
                  va="center", ha="left", fontsize=7.5, style="italic", color="#666")
        y -= 0.7
        for _, row in dir_rows.iterrows():
            col = DIR_COLORS.get(row["direction"], "#AAA")
            ax_f.scatter([0], [y], s=30, color=col, marker="^", alpha=0.5, zorder=2)
            ax_t.text(0.01, y, row["label"],        va="center", ha="left",  fontsize=8)
            ax_t.text(0.50, y, row.get("source",""), va="center", ha="center", fontsize=7, color="#666")
            ax_t.text(0.63, y, str(int(row["n_sz"])) if pd.notna(row["n_sz"]) else "?",
                      va="center", ha="center", fontsize=8)
            ax_t.text(0.75, y, str(int(row["n_hc"])) if pd.notna(row["n_hc"]) else "?",
                      va="center", ha="center", fontsize=8)
            ax_t.text(0.99, y, row["direction"] or "?",
                      va="center", ha="right", fontsize=8, color=col)
            y -= 1

    # ── pooled diamond ─────────────────────────────────────────────────────
    if meta and len(es_rows) >= 2:
        y -= 0.5
        ax_f.axhline(y + 0.75, color="#444", lw=0.8)
        g, lo, hi = meta["pooled_g"], meta["ci_lo"], meta["ci_hi"]
        ax_f.fill([lo, g, hi, g, lo], [y, y+0.38, y, y-0.38, y],
                   color="#1A1A1A", zorder=4, alpha=0.85)
        ci_str = f"{g:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        het    = f"I²={meta['I2']:.0f}%   Q={meta['Q']:.1f}   p={meta['p_Q']:.3f}   τ²={meta['tau2']:.3f}"
        ax_t.text(0.01, y, f"Pooled (k={meta['k']})", va="center", ha="left",
                  fontsize=9, fontweight="bold")
        ax_t.text(0.99, y, ci_str, va="center", ha="right",
                  fontsize=9, fontweight="bold")
        ax_t.text(0.01, y - 0.65, het, va="center", ha="left",
                  fontsize=7.5, color="#555", style="italic")
        y -= 1.6

    # ── forest formatting ──────────────────────────────────────────────────
    ax_f.axvline(0, color="black", lw=1.0)
    ax_f.set_xlabel("Hedges' g  (SZ < HC = negative = hypo-expression)", fontsize=9)
    ax_f.set_title(f"Forest Plot — {au}", fontsize=13, fontweight="bold", pad=10)
    ax_f.set_ylim(y - 0.5, 1)
    ax_f.set_yticks([])
    ax_f.spines[["top","right","left"]].set_visible(False)
    ax_f.xaxis.grid(True, ls="--", alpha=0.4)
    ax_f.set_axisbelow(True)

    # ── table formatting ───────────────────────────────────────────────────
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(y - 0.5, 1)
    ax_t.axis("off")
    for x, lbl in [(0.01,"Study"), (0.50,"Source"), (0.63,"n SZ"),
                   (0.75,"n HC"), (0.99,"g [95% CI]")]:
        ax_t.text(x, 0.6, lbl, va="bottom",
                  ha="left" if x < 0.5 else ("center" if x < 0.8 else "right"),
                  fontsize=8.5, fontweight="bold", color="#222")
    ax_t.axhline(0.55, color="#444", lw=0.8)

    # ── legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=DIR_COLORS["decreased"],     label="Decreased in SZ"),
        mpatches.Patch(color=DIR_COLORS["increased"],     label="Increased in SZ"),
        mpatches.Patch(color=DIR_COLORS["no difference"], label="No difference"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#333", markersize=8,
               label="Pooled (diamond)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#888", markersize=7,
               label="PubMed"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markersize=7,
               label="Semantic Scholar"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#888", markersize=6,
               label="Google Scholar"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#888", markersize=7,
               label="CrossRef"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(RESULTS_DIR, f"forest_{au.replace(' ','_')}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out}")


def make_all_forest_plots(long_df, meta_df):
    au_counts    = long_df["au"].value_counts()
    aus_to_plot  = au_counts[au_counts >= 2].index.tolist()
    for au in aus_to_plot:
        grp      = long_df[long_df["au"] == au]
        meta_row = meta_df[meta_df["au"] == au]
        meta     = meta_row.iloc[0].to_dict() if not meta_row.empty else None
        forest_plot(au, grp, meta)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Bubble summary
# ═══════════════════════════════════════════════════════════════════════════
def bubble_summary(long_df, meta_df):
    if meta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        meta_df["pooled_g"], meta_df["n_studies"],
        s      = np.sqrt(meta_df["total_n_sz"].fillna(50)) * 10,
        c      = meta_df["I2"],
        cmap   = "RdYlGn_r", vmin=0, vmax=100,
        alpha  = 0.75, edgecolors="white", linewidths=0.8, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("I² (heterogeneity %)", fontsize=9)
    ax.errorbar(meta_df["pooled_g"], meta_df["n_studies"],
                xerr=[meta_df["pooled_g"]-meta_df["ci_lo"],
                      meta_df["ci_hi"]-meta_df["pooled_g"]],
                fmt="none", color="#444", lw=1.0, capsize=3, zorder=2)
    for _, row in meta_df.iterrows():
        ax.annotate(row["au"], (row["pooled_g"], row["n_studies"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Pooled Hedges' g  (negative = hypo-expression in SZ)", fontsize=10)
    ax.set_ylabel("Number of studies", fontsize=10)
    ax.set_title("AU Evidence Summary\n(size ∝ total N; colour = I²)",
                 fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, ls="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "bubble_summary.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Direction heatmap
# ═══════════════════════════════════════════════════════════════════════════
def direction_heatmap(long_df):
    dir_map = {"decreased": -1, "no difference": 0, "increased": 1}
    sub     = long_df[["label","au","direction"]].copy()
    sub["dir_num"] = sub["direction"].map(dir_map)
    pivot   = sub.pivot_table(index="label", columns="au",
                               values="dir_num", aggfunc="first")
    if pivot.empty or pivot.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1]*0.7),
                                    max(4, pivot.shape[0]*0.42)))
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Direction of AU Effects Across Studies\n"
                 "(Blue=decreased in SZ, Red=increased, White=no diff)",
                 fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1,0,1], fraction=0.03)
    cbar.ax.set_yticklabels(["Decreased","No diff","Increased"], fontsize=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "direction_heatmap.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Source breakdown chart
# ═══════════════════════════════════════════════════════════════════════════
def source_breakdown(long_df):
    """Bar chart showing how many papers came from each source."""
    counts = long_df.groupby("source")["id"].nunique().sort_values(ascending=True)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(counts.index, counts.values,
                   color=["#4878D0","#EE854A","#6ACC65","#D65F5F"][:len(counts)])
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of papers with AU data", fontsize=9)
    ax.set_title("Papers Retrieved by Source", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "source_breakdown.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. LLM narrative
# ═══════════════════════════════════════════════════════════════════════════
SUMMARY_PROMPT = textwrap.dedent("""
You are a clinical neuroscience researcher writing the Results section of a
systematic review on facial Action Units (AUs) in schizophrenia.

Below are:
  A) Meta-analysis results (pooled Hedges g per AU, random-effects DerSimonian-Laird)
  B) Direction-only table (AUs without quantitative data)
  C) Source breakdown (how many studies per database)

Write a structured narrative (~500 words):
1. Overall hypo-expression pattern in schizophrenia
2. AUs with strongest, most consistent evidence (by g magnitude + low I2)
3. AUs with heterogeneous results and possible reasons
4. AUs specifically linked to apathy vs. broader negative symptoms
5. Recommended AU feature set for an ML pipeline targeting apathy

Meta-analysis (JSON): {meta_json}
Direction-only (JSON): {dir_json}
Source breakdown (JSON): {src_json}
""").strip()


def narrative_summary(meta_df, long_df) -> str:
    dir_summary = (long_df[long_df["hedges_g"].isna()]
                   .groupby("au")["direction"].value_counts()
                   .reset_index()
                   .to_dict(orient="records"))
    src_summary = long_df.groupby("source")["id"].nunique().to_dict()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{"role":"user",
                   "content": SUMMARY_PROMPT.format(
                       meta_json = meta_df.to_json(orient="records", indent=2),
                       dir_json  = json.dumps(dir_summary, indent=2),
                       src_json  = json.dumps(src_summary, indent=2),
                   )}]
    )
    return resp.content[0].text


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Step 1: Retrieving papers ===")
    papers = collect_all_papers()

    print("\n=== Step 2: Extracting AU data with Claude ===")
    extractions = []
    for p in tqdm(papers):
        extractions.append(extract_paper(p))
        time.sleep(0.3)

    with open(f"{RESULTS_DIR}/raw_extractions.json", "w") as f:
        json.dump([{**p, "extraction": e}
                   for p, e in zip(papers, extractions)], f, indent=2)

    print("\n=== Step 3: Building long table + effect sizes ===")
    long_df = build_long_table(papers, extractions)
    if long_df.empty:
        print("  No AU data extracted. Check raw_extractions.json")
        exit(1)
    long_df.to_csv(f"{RESULTS_DIR}/au_long.csv", index=False)
    print(f"  {len(long_df)} AU observations, "
          f"{long_df['hedges_g'].notna().sum()} with effect sizes")

    print("\n=== Step 4: Random-effects meta-analyses ===")
    meta_df = run_meta(long_df)
    meta_df.to_csv(f"{RESULTS_DIR}/meta_results.csv", index=False)
    if not meta_df.empty:
        print(meta_df[["au","n_studies","pooled_g","ci_lo","ci_hi","I2","p_Q"]]
              .to_string(index=False))

    print("\n=== Step 5: Forest plots ===")
    make_all_forest_plots(long_df, meta_df)

    print("\n=== Step 6: Summary charts ===")
    bubble_summary(long_df, meta_df)
    direction_heatmap(long_df)
    source_breakdown(long_df)

    print("\n=== Step 7: LLM narrative ===")
    narrative = narrative_summary(meta_df, long_df)
    print(narrative[:600] + "...\n")
    with open(f"{RESULTS_DIR}/meta_analysis_summary.txt", "w") as f:
        f.write(narrative)

    print(f"\nDone. All outputs in ./{RESULTS_DIR}/")
    print("""
NOTE: Abstracts rarely report raw means/SDs for AUs.
After reading full texts, fill in au_long.csv columns:
  mean_sz, sd_sz, mean_hc, sd_hc
then re-run from Step 4 for richer forest plots.
""")
