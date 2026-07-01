import os
import re
import json
import asyncio
import httpx
from typing import Any, Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


load_dotenv()

app = FastAPI(title="DrugIQ Dossier Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"


class DossierRequest(BaseModel):
    target: str
    compound: str
    indication: str
    context: str = ""


@app.get("/")
def root():
    return {
        "status": "ok",
        "tool": "DrugIQ Dossier Intelligence Engine",
        "health": "/health",
        "dossier_endpoint": "/dossier-v2",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tool": "DrugIQ Dossier Intelligence Engine",
        "model": GROQ_MODEL,
        "groq_key_configured": bool(GROQ_API_KEY),
    }


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def clamp(value: float, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(round(value))))


def clean_xml(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


async def groq_json(prompt: str, system: str) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GROQ_BASE,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.15,
                    "max_tokens": 2500,
                    "response_format": {"type": "json_object"},
                },
            )

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                return {"error": data["error"].get("message", "Groq error")}

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"^```\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed

            return {"error": f"Could not parse model response: {text[:300]}"}

    except httpx.HTTPStatusError as e:
        return {
            "error": f"Groq API HTTP error: {e.response.status_code}",
            "details": e.response.text[:500],
        }
    except Exception as e:
        return {"error": f"Groq request failed: {str(e)}"}


async def opentargets_disease_and_targets(indication: str) -> Dict[str, Any]:
    result = {
        "disease_name": indication,
        "efo_id": "",
        "targets": [],
        "source": "OpenTargets",
    }

    search_query = """
    query SearchDisease($query: String!) {
      search(queryString: $query, entityNames: ["disease"]) {
        hits {
          id
          name
        }
      }
    }
    """

    targets_query = """
    query DiseaseTargets($efoId: String!) {
      disease(efoId: $efoId) {
        name
        associatedTargets(page: {index: 0, size: 25}) {
          rows {
            score
            target {
              id
              approvedSymbol
              approvedName
              biotype
              functionDescriptions
            }
          }
        }
      }
    }
    """

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            search = await client.post(
                OPENTARGETS_GQL,
                json={
                    "query": search_query,
                    "variables": {"query": indication},
                },
                headers={"Content-Type": "application/json"},
            )

            if search.status_code != 200:
                result["error"] = f"OpenTargets search HTTP {search.status_code}"
                return result

            hits = (
                search.json()
                .get("data", {})
                .get("search", {})
                .get("hits", [])
            )

            if not hits:
                return result

            efo_id = hits[0].get("id", "")
            result["efo_id"] = efo_id
            result["disease_name"] = hits[0].get("name", indication)

            gql = await client.post(
                OPENTARGETS_GQL,
                json={
                    "query": targets_query,
                    "variables": {"efoId": efo_id},
                },
                headers={"Content-Type": "application/json"},
            )

            if gql.status_code != 200:
                result["error"] = f"OpenTargets target HTTP {gql.status_code}"
                return result

            rows = (
                gql.json()
                .get("data", {})
                .get("disease", {})
                .get("associatedTargets", {})
                .get("rows", [])
                or []
            )

            for row in rows:
                target = row.get("target", {}) or {}

                result["targets"].append(
                    {
                        "id": target.get("id", ""),
                        "symbol": target.get("approvedSymbol", ""),
                        "name": target.get("approvedName", ""),
                        "biotype": target.get("biotype", ""),
                        "score": row.get("score", 0),
                        "functions": target.get("functionDescriptions", []) or [],
                    }
                )

    except Exception as e:
        result["error"] = str(e)

    return result


async def pubchem_properties(compound: str) -> Dict[str, Any]:
    result = {
        "compound": compound,
        "cid": None,
        "properties": {},
        "source": "PubChem",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            cid_resp = await client.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/cids/JSON"
            )

            if cid_resp.status_code != 200:
                result["error"] = f"PubChem CID HTTP {cid_resp.status_code}"
                return result

            cid = (
                cid_resp.json()
                .get("IdentifierList", {})
                .get("CID", [None])[0]
            )

            if not cid:
                return result

            result["cid"] = cid

            props_resp = await client.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/"
                "MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,"
                "HBondAcceptorCount,RotatableBondCount,TPSA,IUPACName/JSON"
            )

            if props_resp.status_code == 200:
                props = (
                    props_resp.json()
                    .get("PropertyTable", {})
                    .get("Properties", [{}])[0]
                )
                result["properties"] = props
            else:
                result["error"] = f"PubChem properties HTTP {props_resp.status_code}"

    except Exception as e:
        result["error"] = str(e)

    return result


async def chembl_lookup(compound: str) -> Dict[str, Any]:
    result = {
        "compound": compound,
        "chembl_id": "",
        "max_phase": "",
        "indication": "",
        "molecule_type": "",
        "properties": {},
        "source": "ChEMBL",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule/search",
                params={
                    "q": compound,
                    "format": "json",
                    "limit": 1,
                },
            )

            if resp.status_code != 200:
                result["error"] = f"ChEMBL HTTP {resp.status_code}"
                return result

            molecules = resp.json().get("molecules", []) or []

            if not molecules:
                return result

            mol = molecules[0]
            result["chembl_id"] = mol.get("molecule_chembl_id", "")
            result["max_phase"] = mol.get("max_phase", "")
            result["indication"] = mol.get("indication_class", "") or ""
            result["molecule_type"] = mol.get("molecule_type", "") or ""
            result["properties"] = mol.get("molecule_properties", {}) or {}

    except Exception as e:
        result["error"] = str(e)

    return result


async def pubmed_evidence(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    papers = []

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                },
            )

            if search.status_code != 200:
                return papers

            ids = (
                search.json()
                .get("esearchresult", {})
                .get("idlist", [])
            )

            if not ids:
                return papers

            fetch = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )

            if fetch.status_code != 200:
                return papers

            xml = fetch.text

            titles = re.findall(
                r"<ArticleTitle>(.*?)</ArticleTitle>",
                xml,
                re.DOTALL,
            )
            abstracts = re.findall(
                r"<AbstractText[^>]*>(.*?)</AbstractText>",
                xml,
                re.DOTALL,
            )
            years = re.findall(
                r"<PubDate>.*?<Year>(\d{4})</Year>",
                xml,
                re.DOTALL,
            )
            pmids = re.findall(
                r"<PMID.*?>(\d+)</PMID>",
                xml,
                re.DOTALL,
            )

            for i, title in enumerate(titles[:max_results]):
                papers.append(
                    {
                        "pmid": pmids[i] if i < len(pmids) else "",
                        "title": clean_xml(title),
                        "abstract": clean_xml(
                            abstracts[i] if i < len(abstracts) else ""
                        )[:700],
                        "year": years[i] if i < len(years) else "",
                        "source": "PubMed",
                    }
                )

    except Exception:
        pass

    return papers


async def clinical_trials(indication: str, max_results: int = 12) -> List[Dict[str, Any]]:
    trials = []

    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            resp = await client.get(
                CLINICALTRIALS_BASE,
                params={
                    "query.cond": indication,
                    "filter.overallStatus": "RECRUITING",
                    "pageSize": max_results,
                    "format": "json",
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "DrugIQ/1.0",
                },
            )

            if resp.status_code != 200:
                return trials

            studies = resp.json().get("studies", []) or []

            for study in studies[:max_results]:
                protocol = study.get("protocolSection", {}) or {}
                identification = protocol.get("identificationModule", {}) or {}
                design = protocol.get("designModule", {}) or {}
                sponsor = protocol.get("sponsorCollaboratorsModule", {}) or {}
                description = protocol.get("descriptionModule", {}) or {}

                trials.append(
                    {
                        "nct_id": identification.get("nctId", ""),
                        "title": identification.get("briefTitle", "No title"),
                        "phase": ", ".join(design.get("phases", []) or []) or "N/A",
                        "sponsor": (
                            sponsor.get("leadSponsor", {}) or {}
                        ).get("name", ""),
                        "summary": description.get("briefSummary", "")[:500],
                        "source": "ClinicalTrials.gov",
                    }
                )

    except Exception:
        pass

    return trials


def score_target_validation(target: str, ot_data: Dict[str, Any]) -> Dict[str, Any]:
    targets = ot_data.get("targets", []) or []
    target_upper = target.upper()

    exact = None

    for item in targets:
        if safe_str(item.get("symbol")).upper() == target_upper:
            exact = item
            break

    if exact:
        raw_score = safe_float(exact.get("score"), 0)
        score = clamp(50 + raw_score * 50)
        return {
            "score": score,
            "status": "supported",
            "reason": f"{target} appears in OpenTargets disease associations.",
            "matched_target": exact,
        }

    if targets:
        return {
            "score": 42,
            "status": "gap",
            "reason": f"{target} was not found among the top OpenTargets associations.",
            "top_targets": targets[:5],
        }

    return {
        "score": 30,
        "status": "no_data",
        "reason": "No target association data returned.",
    }


def score_chemistry(pubchem: Dict[str, Any], chembl: Dict[str, Any]) -> Dict[str, Any]:
    props = pubchem.get("properties", {}) or {}

    mw = safe_float(props.get("MolecularWeight"), 0)
    logp = safe_float(props.get("XLogP"), 0)
    hbd = safe_int(props.get("HBondDonorCount"), 0)
    hba = safe_int(props.get("HBondAcceptorCount"), 0)
    rotb = safe_int(props.get("RotatableBondCount"), 0)
    tpsa = safe_float(props.get("TPSA"), 0)

    violations = []

    if mw and mw > 500:
        violations.append(f"MW {mw:.0f} > 500")
    if logp and logp > 5:
        violations.append(f"LogP {logp:.2f} > 5")
    if hbd > 5:
        violations.append(f"HBD {hbd} > 5")
    if hba > 10:
        violations.append(f"HBA {hba} > 10")
    if rotb > 10:
        violations.append(f"Rotatable bonds {rotb} > 10")
    if tpsa > 140:
        violations.append(f"TPSA {tpsa:.0f} > 140")

    if not pubchem.get("cid") and not chembl.get("chembl_id"):
        return {
            "score": 35,
            "status": "compound_not_resolved",
            "violations": [],
            "properties": {
                "mw": mw,
                "logp": logp,
                "hbd": hbd,
                "hba": hba,
                "rotb": rotb,
                "tpsa": tpsa,
            },
            "chembl_phase": "",
            "reason": "Compound could not be resolved in PubChem or ChEMBL.",
        }

    score = 85 - (len(violations) * 15)

    max_phase = chembl.get("max_phase", "")

    phase_num = safe_int(max_phase, -1)

    if phase_num >= 4:
        score += 10
    elif phase_num >= 2:
        score += 5

    score = clamp(score)

    if len(violations) >= 3:
        status = "high_liability"
    elif len(violations) >= 1:
        status = "moderate_liability"
    else:
        status = "drug_like"

    return {
        "score": score,
        "status": status,
        "violations": violations,
        "properties": {
            "mw": mw,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
            "rotb": rotb,
            "tpsa": tpsa,
        },
        "chembl_phase": max_phase,
        "reason": (
            "Compound appears drug-like."
            if not violations
            else "Compound has physicochemical liabilities: "
            + "; ".join(violations)
        ),
    }


def score_precedent(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(papers)

    if count >= 6:
        score = 75
        status = "rich_literature"
    elif count >= 3:
        score = 60
        status = "moderate_literature"
    elif count >= 1:
        score = 45
        status = "sparse_literature"
    else:
        score = 30
        status = "no_literature"

    return {
        "score": score,
        "status": status,
        "paper_count": count,
        "reason": f"{count} relevant PubMed paper(s) found.",
    }


def score_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(trials)

    if count >= 10:
        score = 78
        status = "competitive_validated"
    elif count >= 4:
        score = 65
        status = "active_landscape"
    elif count >= 1:
        score = 52
        status = "sparse_landscape"
    else:
        score = 40
        status = "white_space_or_low_activity"

    return {
        "score": score,
        "status": status,
        "trial_count": count,
        "reason": f"{count} recruiting clinical trial(s) found for the indication.",
    }


def build_kill_criteria(
    target_score: Dict[str, Any],
    chem_score: Dict[str, Any],
    precedent_score: Dict[str, Any],
    trial_score: Dict[str, Any],
) -> List[Dict[str, Any]]:
    criteria = []

    if target_score["score"] < 45:
        criteria.append(
            {
                "severity": "high",
                "category": "Target validation",
                "criterion": "Weak target-disease association",
                "why_it_matters": "A weak disease link increases risk that modulating the target will not translate clinically.",
                "suggested_resolution": "Confirm target biology with genetic, transcriptomic, pathway, or disease-model evidence.",
            }
        )

    if chem_score["score"] < 55:
        criteria.append(
            {
                "severity": "high",
                "category": "Chemistry / ADME",
                "criterion": "Material drug-likeness liabilities",
                "why_it_matters": "Poor physicochemical profile can cause exposure, solubility, permeability, or formulation failure.",
                "suggested_resolution": "Run ADME profiling and consider analog redesign or scaffold optimization.",
            }
        )

    if len(chem_score.get("violations", [])) >= 3:
        criteria.append(
            {
                "severity": "high",
                "category": "Chemistry / ADME",
                "criterion": "Multiple Lipinski/Veber-style violations",
                "why_it_matters": "Multiple property violations reduce probability of oral drug viability.",
                "suggested_resolution": "Prioritize analogs with lower MW, lower LogP, lower TPSA, or fewer rotatable bonds.",
            }
        )

    if precedent_score["score"] < 40:
        criteria.append(
            {
                "severity": "medium",
                "category": "Prior art",
                "criterion": "Sparse public precedent",
                "why_it_matters": "Thin literature may indicate novelty, but also increases biological uncertainty.",
                "suggested_resolution": "Run focused literature review and confirm pathway relevance experimentally.",
            }
        )

    if trial_score["score"] < 45:
        criteria.append(
            {
                "severity": "medium",
                "category": "Clinical landscape",
                "criterion": "Low clinical activity in indication",
                "why_it_matters": "Lack of clinical activity may suggest low commercial interest, difficult recruitment, or weak translational confidence.",
                "suggested_resolution": "Check adjacent indications, patient subgroups, and biomarker-defined populations.",
            }
        )

    return criteria


def next_best_experiments(
    target_score: Dict[str, Any],
    chem_score: Dict[str, Any],
    precedent_score: Dict[str, Any],
    trial_score: Dict[str, Any],
) -> List[Dict[str, str]]:
    experiments = []

    if target_score["score"] < 60:
        experiments.append(
            {
                "priority": "High",
                "experiment": "Target-disease validation assay",
                "purpose": "Confirm whether modulating the target affects disease-relevant biology.",
                "output": "Pathway modulation, disease phenotype rescue, or biomarker shift.",
            }
        )

    if chem_score["score"] < 70:
        experiments.append(
            {
                "priority": "High",
                "experiment": "In vitro ADME panel",
                "purpose": "Assess solubility, permeability, metabolic stability, CYP inhibition, and plasma protein binding.",
                "output": "Go/no-go chemistry liability profile.",
            }
        )

    experiments.append(
        {
            "priority": "Medium",
            "experiment": "Target engagement assay",
            "purpose": "Confirm that the compound engages the intended target in a relevant cellular context.",
            "output": "Dose-response curve and cellular potency estimate.",
        }
    )

    if trial_score["score"] >= 60:
        experiments.append(
            {
                "priority": "Medium",
                "experiment": "Competitive trial landscape review",
                "purpose": "Identify differentiation opportunities against active clinical programs.",
                "output": "Patient subgroup, endpoint, and biomarker strategy.",
            }
        )

    if precedent_score["score"] < 60:
        experiments.append(
            {
                "priority": "Medium",
                "experiment": "Mechanistic literature deep dive",
                "purpose": "Resolve sparse or contradictory public evidence before wet-lab investment.",
                "output": "Evidence map with support and contradiction claims.",
            }
        )

    return experiments[:5]


def build_sources(
    ot_data: Dict[str, Any],
    pubchem: Dict[str, Any],
    chembl: Dict[str, Any],
    papers: List[Dict[str, Any]],
    trials: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sources = []
    idx = 1

    if ot_data.get("efo_id"):
        sources.append(
            {
                "id": idx,
                "source": "OpenTargets",
                "label": f"{ot_data.get('disease_name')} target associations",
                "url": f"https://platform.opentargets.org/disease/{ot_data.get('efo_id')}",
            }
        )
        idx += 1

    if pubchem.get("cid"):
        sources.append(
            {
                "id": idx,
                "source": "PubChem",
                "label": f"{pubchem.get('compound')} compound properties",
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem.get('cid')}",
            }
        )
        idx += 1

    if chembl.get("chembl_id"):
        sources.append(
            {
                "id": idx,
                "source": "ChEMBL",
                "label": f"{chembl.get('compound')} ChEMBL record",
                "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl.get('chembl_id')}/",
            }
        )
        idx += 1

    for paper in papers:
        if paper.get("pmid"):
            sources.append(
                {
                    "id": idx,
                    "source": "PubMed",
                    "label": paper.get("title", "PubMed paper"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/",
                }
            )
            idx += 1

    for trial in trials:
        if trial.get("nct_id"):
            sources.append(
                {
                    "id": idx,
                    "source": "ClinicalTrials.gov",
                    "label": trial.get("title", "Clinical trial"),
                    "url": f"https://clinicaltrials.gov/study/{trial.get('nct_id')}",
                }
            )
            idx += 1

    return sources


def build_claims(
    target: str,
    compound: str,
    indication: str,
    target_score: Dict[str, Any],
    chem_score: Dict[str, Any],
    precedent_score: Dict[str, Any],
    trial_score: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "claim": f"{target} has disease-association evidence for {indication}.",
            "stance": "support" if target_score["score"] >= 60 else "weak_support",
            "confidence": target_score["score"],
            "evidence_type": "genetic / target association",
            "source_refs": [
                s["id"]
                for s in sources
                if s["source"] == "OpenTargets"
            ],
        },
        {
            "claim": f"{compound} has an acceptable drug-likeness profile.",
            "stance": "support" if chem_score["score"] >= 70 else "concern",
            "confidence": chem_score["score"],
            "evidence_type": "compound properties",
            "source_refs": [
                s["id"]
                for s in sources
                if s["source"] in {"PubChem", "ChEMBL"}
            ],
        },
        {
            "claim": f"There is public literature precedent for {target}, {compound}, or {indication}.",
            "stance": "support" if precedent_score["score"] >= 60 else "weak_support",
            "confidence": precedent_score["score"],
            "evidence_type": "literature",
            "source_refs": [
                s["id"]
                for s in sources
                if s["source"] == "PubMed"
            ],
        },
        {
            "claim": f"The clinical-trial landscape for {indication} shows translational activity.",
            "stance": "support" if trial_score["score"] >= 60 else "weak_support",
            "confidence": trial_score["score"],
            "evidence_type": "clinical trials",
            "source_refs": [
                s["id"]
                for s in sources
                if s["source"] == "ClinicalTrials.gov"
            ],
        },
    ]


def recommendation_from_score(
    score: int,
    kill_criteria: List[Dict[str, Any]],
) -> Dict[str, str]:
    high_kills = [
        item
        for item in kill_criteria
        if item.get("severity") == "high"
    ]

    if score >= 75 and not high_kills:
        return {
            "decision": "Proceed",
            "risk_level": "Low external evidence risk",
            "summary": "Public evidence is broadly supportive. Proceed to normal experimental validation.",
        }

    if score >= 58 and len(high_kills) <= 1:
        return {
            "decision": "Proceed with caution",
            "risk_level": "Moderate external evidence risk",
            "summary": "Evidence is mixed. Resolve flagged risks before major wet-lab or clinical investment.",
        }

    if score >= 45:
        return {
            "decision": "Pause / redesign",
            "risk_level": "Elevated external evidence risk",
            "summary": "Important validation gaps or liabilities exist. Redesign or run focused de-risking experiments.",
        }

    return {
        "decision": "Do not advance yet",
        "risk_level": "High external evidence risk",
        "summary": "Public evidence is currently too weak or too risky to justify advancement without additional validation.",
    }


@app.post("/dossier-v2")
async def dossier_v2(req: DossierRequest):
    target = req.target.strip()
    compound = req.compound.strip()
    indication = req.indication.strip()

    if not target or not compound or not indication:
        return {
            "error": "target, compound, and indication are required.",
        }

    literature_query = (
        f'("{target}" OR "{compound}") AND "{indication}" '
        f'AND (drug OR therapy OR inhibitor OR biomarker OR clinical)'
    )

    (
        ot_data,
        pubchem,
        chembl,
        papers,
        trials,
    ) = await asyncio.gather(
        opentargets_disease_and_targets(indication),
        pubchem_properties(compound),
        chembl_lookup(compound),
        pubmed_evidence(literature_query, max_results=8),
        clinical_trials(indication, max_results=12),
    )

    target_score = score_target_validation(target, ot_data)
    chem_score = score_chemistry(pubchem, chembl)
    precedent_score = score_precedent(papers)
    trial_score = score_trials(trials)

    weights = {
        "target_validation": 0.32,
        "chemistry": 0.30,
        "precedent": 0.18,
        "clinical_landscape": 0.20,
    }

    external_score = clamp(
        target_score["score"] * weights["target_validation"]
        + chem_score["score"] * weights["chemistry"]
        + precedent_score["score"] * weights["precedent"]
        + trial_score["score"] * weights["clinical_landscape"]
    )

    kill_criteria = build_kill_criteria(
        target_score,
        chem_score,
        precedent_score,
        trial_score,
    )

    recommendation = recommendation_from_score(
        external_score,
        kill_criteria,
    )

    experiments = next_best_experiments(
        target_score,
        chem_score,
        precedent_score,
        trial_score,
    )

    sources = build_sources(
        ot_data,
        pubchem,
        chembl,
        papers,
        trials,
    )

    claims = build_claims(
        target,
        compound,
        indication,
        target_score,
        chem_score,
        precedent_score,
        trial_score,
        sources,
    )

    ai_prompt = f"""
You are DrugIQ, a rigorous external evidence reviewer for drug discovery.

Create an executive scientific dossier for this candidate.

Candidate:
Target: {target}
Compound: {compound}
Indication: {indication}
Context: {req.context or "General drug discovery"}

Structured scores:
Target validation:
{json.dumps(target_score, indent=2)}

Chemistry / ADME:
{json.dumps(chem_score, indent=2)}

Precedent literature:
{json.dumps(precedent_score, indent=2)}

Clinical trial landscape:
{json.dumps(trial_score, indent=2)}

Kill criteria:
{json.dumps(kill_criteria, indent=2)}

Claims:
{json.dumps(claims, indent=2)}

Return ONLY a JSON object with exactly these keys:
{{
  "executive_summary": "3-4 sentence board-level summary",
  "scientific_verdict": "clear scientific verdict",
  "why_it_might_work": ["point 1", "point 2", "point 3"],
  "why_it_might_fail": ["point 1", "point 2", "point 3"],
  "contradictions_or_uncertainties": ["uncertainty 1", "uncertainty 2"],
  "recommended_decision": "Proceed OR Proceed with caution OR Pause / redesign OR Do not advance yet",
  "score_explanation": "explain why the External Evidence Score is what it is",
  "next_best_action": "single most important next action"
}}

Rules:
- Do not give medical advice.
- Do not invent clinical claims.
- Stay grounded in the structured evidence.
- Do not include markdown.
- Do not include text outside the JSON object.
""".strip()

    ai_summary = await groq_json(
        ai_prompt,
        system=(
            "You are a JSON API for drug discovery evidence review. "
            "Output only valid JSON. No markdown."
        ),
    )

    return {
        "candidate": {
            "target": target,
            "compound": compound,
            "indication": indication,
        },
        "external_evidence_score": external_score,
        "recommendation": recommendation,
        "sections": {
            "target_validation": target_score,
            "chemistry_adme": chem_score,
            "precedent_literature": precedent_score,
            "clinical_trial_landscape": trial_score,
        },
        "kill_criteria": kill_criteria,
        "claims": claims,
        "next_best_experiments": experiments,
        "ai_summary": ai_summary,
        "sources": sources,
        "raw_evidence": {
            "opentargets": ot_data,
            "pubchem": pubchem,
            "chembl": chembl,
            "pubmed": papers,
            "clinical_trials": trials,
        },
        "disclaimer": (
            "Research use only. Not medical advice. "
            "Not a prediction of clinical success."
        ),
    }
