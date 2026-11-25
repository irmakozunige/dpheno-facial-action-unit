"""
meta_retrieval.py
-----------------
All paper retrieval functions (8 sources) + full-text fetching.

Sources: PubMed, Europe PMC, Semantic Scholar, OpenAlex, CORE,
         CrossRef, Google Scholar, bioRxiv/medRxiv, Unpaywall,
         Publisher direct, Sci-Hub

Imported by: 01_meta_analysis.py
"""

from meta_config import (
    Entrez, BROWSER_UA, QUERIES, PUBMED_QUERY, MAX_PAPERS,
    CORE_API_KEY, SCIHUB_MIRRORS, _fp, _get,
    extract_results_section, pdf_bytes_to_text,
    unpaywall_pdf_url
)
import os, re, time, hashlib
import requests

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
    if not CORE_API_KEY:
        print("    [CORE] No API key — skipping (get free key at core.ac.uk/api-keys)")
        return []
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    # Request extra fields for better full-text coverage
    r = _get("https://api.core.ac.uk/v3/search/works",
             params={"q": query, "limit": n,
                     "fields": "id,title,abstract,authors,yearPublished,"
                               "doi,publisher,downloadUrl,fullText,links"},
             headers=headers, timeout=30)
    if not r:
        print("    [CORE] No response"); return []
    if r.status_code == 401:
        print("    [CORE] Invalid API key"); return []
    if r.status_code == 429:
        print("    [CORE] Rate limited"); return []
    if r.status_code != 200:
        print(f"    [CORE] HTTP {r.status_code}"); return []
    out = []
    for w in r.json().get("results", []):
        abst = w.get("abstract", "") or ""
        ttl  = w.get("title", "") or ""
        if not abst.strip(): continue
        doi  = w.get("doi", "") or ""
        year = str(w.get("yearPublished", "?"))
        aus  = w.get("authors", [])
        fa   = (aus[0].get("name", "")).split()[-1] if aus else ""
        # CORE provides multiple download URLs — try all
        pdf_url = w.get("downloadUrl", "") or ""
        # also check links array for PDF
        if not pdf_url:
            for link in (w.get("links") or []):
                if link.get("type") == "download":
                    pdf_url = link.get("url", "")
                    break
        p = _paper(f"core_{w.get('id', _fp(ttl)[:8])}", "CORE",
                   fa, year, ttl, abst, w.get("publisher", ""), doi)
        p["oa_pdf_url"] = pdf_url
        # CORE sometimes returns full text directly — use it if available
        if w.get("fullText"):
            p["full_text"] = w["fullText"][:8000]
            p["full_text_source"] = "core_fulltext"
        out.append(p)
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
                fp = _fp(p["title"])
                # also deduplicate by DOI if available
                doi_key = f"doi_{p.get('doi','')}" if p.get("doi") else None
                if fp not in seen_set and (not doi_key or doi_key not in seen_set):
                    seen_set.add(fp)
                    if doi_key: seen_set.add(doi_key)
                    papers.append(p); new+=1
            print(f"      +{new}  (total {len(papers)})")
            time.sleep(0.5)
    print(f"\n  Total unique papers: {len(papers)}")
    return papers[:MAX_PAPERS]