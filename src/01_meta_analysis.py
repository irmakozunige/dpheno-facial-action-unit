"""
01_meta_analysis.py
--------------------
Main runner: ranking, extraction, meta-analysis, plots, narrative.

Imports from:
  meta_config.py    — LLM client, AU reference, prompts, config
  meta_retrieval.py — paper retrieval + full-text functions

Run via: sbatch scripts/run_meta.sbatch
"""

from meta_config import (
    llm_chat, AU_REFERENCE, AU_REF_JSON, EXTRACT_PROMPT,
    RESULTS_DIR, TOP_K, LLM_MODEL,
    _AU_PAT, _TOOL_PAT, _SZ_PAT, _QUANT_PAT
)
from meta_retrieval import collect_all_papers, enrich_with_full_text

# ═══════════════════════════════════════════════════════════════════════════
# 1b. RELEVANCE RANKING — score and keep top-K before extraction
# ═══════════════════════════════════════════════════════════════════════════

# AU codes and schizophrenia-specific terms
_AU_PAT   = re.compile(r"\bAU\s*\d{1,2}\b", re.I)
_TOOL_PAT = re.compile(
    r"\b(OpenFace|FACS|facial action coding system|facial action coding|"
    r"action unit|AU intensity|AU frequency|AU occurrence|AU activation)\b", re.I)
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

        # Hard gate: MUST have BOTH:
        #   AU signal = actual AU code (AU4, AU12) OR specific tool (OpenFace, FACS)
        #   SZ signal = schizophrenia/psychosis term
        has_au = (au_hits > 0 or tool_hits > 0)
        has_sz = (sz_hits > 0)
        # extra check: AU code must appear as AU+number, not just "action unit" alone
        # unless OpenFace/FACS is mentioned (they always measure AUs)
        au_code_hits = len(re.findall(r"\bAU\s*\d{1,2}\b", text, re.I))
        openface_facs = len(re.findall(r"\b(OpenFace|FACS|facial action coding system|facial action coding|AffectNet|iMotions|Noldus FaceReader|FaceReader)\b", text, re.I))
        has_au = (au_code_hits > 0 or openface_facs > 0)

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
You are extracting facial Action Unit (AU) findings from a clinical paper for a
systematic meta-analysis on schizophrenia.

FACS AU reference (use this to map descriptions to AU codes):
{au_ref}

IMPORTANT MAPPING RULES:
- "smile" / "lip corner" / "zygomaticus" → AU12
- "brow raise" / "eyebrow raise" / "inner brow" → AU1 or AU2
- "brow lower" / "furrowing" / "corrugator" → AU4
- "cheek raise" / "Duchenne smile" / "orbicularis oculi" → AU6
- "upper lid raise" / "wide eyes" → AU5
- "lip corner depress" / "sadness expression" → AU15
- "jaw drop" / "mouth open" → AU26
- "nose wrinkle" → AU9
- "overall expressivity" / "facial expressivity" / "mean AU intensity" → use "EXPR"
- "head movement" / "head motion" / "head pose variability" → use "HEAD"
- If paper uses FACS/OpenFace/automated coding and reports ANY facial expression
  difference between SZ and HC, extract it even if AU codes are not explicit.

Return EXACTLY this JSON — no preamble, no fences:
{{
  "aus_studied": ["AU12", "AU6", "EXPR"],
  "findings": {{
    "AU12": {{
      "au_label":      "Lip Corner Puller",
      "face_region":   "mouth",
      "functional_tag":"smile / positive affect",
      "direction":     "decreased",
      "feature_type":  "frequency",
      "mean_sz":       null,
      "sd_sz":         null,
      "mean_hc":       null,
      "sd_hc":         null,
      "stat_reported": "p<0.05",
      "exact_quote":   "Smiling behavior was significantly reduced in SZ patients.",
      "source_page":   null,
      "source_para":   null,
      "notes":         ""
    }},
    "EXPR": {{
      "au_label":      "Overall Facial Expressivity",
      "face_region":   "composite",
      "functional_tag":"mean AU intensity across all AUs",
      "direction":     "decreased",
      "feature_type":  "intensity",
      "mean_sz":       0.42,
      "sd_sz":         0.11,
      "mean_hc":       0.68,
      "sd_hc":         0.14,
      "stat_reported": "t(88)=4.1, p<0.001",
      "exact_quote":   "Overall facial expressivity was significantly lower in SZ.",
      "source_page":   null,
      "source_para":   null,
      "notes":         ""
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
- aus_studied MUST be non-empty if the paper reports ANY facial expression
  difference between schizophrenia patients and healthy controls/comparison group
- Map facial descriptions to AU codes using the reference table above
- Use "EXPR" for overall expressivity measures, "HEAD" for head movement
- direction: "decreased" | "increased" | "no difference" | null
- feature_type: "intensity" | "frequency" | "duration" | "presence" | null
- exact_quote: most relevant sentence from the text, verbatim
- mean_sz/sd_sz/mean_hc/sd_hc: numbers only, null if not reported
- target_measure: "apathy" | "negative symptoms" | "both" | "other"
- apathy_scale: PANSS / BNSS / SANS / CAINS / etc., or null
- study_design: "cross-sectional" | "longitudinal" | "RCT" | "other"
- Only return aus_studied: [] if the paper has NO comparison between SZ and HC/controls
""").strip()

def extract_paper(paper):
    body = (f"[FULL TEXT]\n{paper['full_text']}" if paper.get("full_text")
            else f"[ABSTRACT]\nTitle: {paper['title']}\n\n{paper['abstract']}")

    # Cap at 15000 chars — safe with 250k TPM limit
    body = body[:15000]

    prompt = f"{EXTRACT_PROMPT.format(au_ref=AU_REF_JSON)}\n\n---\n{body}"

    try:
        raw = llm_chat(prompt, max_tokens=1200)

        if not raw:
            print(f"  ⚠ Empty response for: {paper['title'][:60]}")
            return {"error": "empty_response"}

        raw = re.sub(r"^```(?:json)?\s*|```$", "", raw, flags=re.MULTILINE).strip()
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            print(f"  ⚠ No JSON in response for: {paper['title'][:60]}")
            print(f"    Raw response: {raw[:200]}")
            return {"error": "no_json", "raw": raw[:300]}

        result = json.loads(m.group(0))
        aus    = result.get("aus_studied", [])
        print(f"  ✓ {paper['title'][:55]} → {len(aus)} AUs: {aus}")
        return result

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error for {paper['title'][:50]}: {e}")
        return {"error": f"json_parse: {e}", "raw": raw[:300] if 'raw' in locals() else ""}
    except Exception as e:
        print(f"  ⚠ Error for {paper['title'][:50]}: {e}")
        return {"error": str(e)}


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
    for i, p in enumerate(tqdm(papers)):
        ext = extract_paper(p)
        # retry once on empty response — may have hit rate limit
        if ext.get("error") in ("empty_response", "no_json"):
            print(f"  Retrying after 15s...")
            time.sleep(15)
            ext = extract_paper(p)
        extractions.append(ext)
        time.sleep(5)   # 5s between calls — well within 14 rpm limit
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
