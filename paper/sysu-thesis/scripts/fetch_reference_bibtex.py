#!/usr/bin/env python3
"""Fetch topic-relevant BibTeX entries for the thesis.

The script only collects public literature. Local experiment artifacts are never
written into the bibliography.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]


REFERENCES = [
    {"key": "theodoridis2025evaluating", "title": "Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception"},
    {"key": "theodoridis2025dtpqa", "arxiv": "2511.13397"},
    {"key": "caesar2020nuscenes", "title": "nuScenes: A Multimodal Dataset for Autonomous Driving"},
    {"key": "dosovitskiy2017carla", "title": "CARLA: An Open Urban Driving Simulator"},
    {"key": "liu2024surveyadatasets", "title": "A Survey on Autonomous Driving Datasets: Statistics, Annotation Quality, and a Future Outlook"},
    {"key": "kaur2021surveyselfdrivingsimulators", "title": "A Survey on Simulators for Testing Self-Driving Cars"},
    {"key": "marcu2024lingoqa", "title": "LingoQA: Visual Question Answering for Autonomous Driving"},
    {"key": "xie2025vlmsreadyad", "arxiv": "2501.04003"},
    {"key": "tom2023readingbetweenthelanes", "title": "Reading Between the Lanes: Text VideoQA on the Road"},
    {"key": "chen2025automatedevaluationcornercases", "title": "Automated Evaluation of Large Vision-Language Models on Self-Driving Corner Cases"},
    {"key": "sima2024drivelm", "arxiv": "2312.14150"},
    {"key": "ding2024holisticautonomousdriving", "title": "Holistic Autonomous Driving Understanding by Bird’s-Eye-View Injected Multi-Modal Large Models"},
    {"key": "charoenpitaks2025tbbench", "arxiv": "2501.05733"},
    {"key": "guo2024drivemllm", "arxiv": "2411.13112"},
    {"key": "choudhary2023talk2bev", "arxiv": "2310.02251"},
    {"key": "gopalkrishnan2024multiframe", "arxiv": "2403.19838"},
    {"key": "jiang2024senna", "arxiv": "2410.22313"},
    {"key": "zheng2024simplellm4ad", "arxiv": "2407.21293"},
    {"key": "jiao2024lavidadrive", "arxiv": "2411.12980"},
    {"key": "wang2024omnidrive", "arxiv": "2405.01533"},
    {"key": "lubberstedt2025v3lma", "title": "V3LMA: Visual 3D-Enhanced Language Model for Autonomous Driving"},
    {"key": "qiao2025lightemma", "arxiv": "2505.00284"},
    {"key": "zhou2025dynrslvlm", "arxiv": "2503.11265"},
    {"key": "rahmanzadehgervi2024vlmsblind", "title": "Vision Language Models Are Blind"},
    {"key": "tong2024eyeswideshut", "title": "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"},
    {"key": "gou2024imagedetails", "arxiv": "2408.03940"},
    {"key": "kamoi2024visonlyqa", "arxiv": "2412.00947"},
    {"key": "kaduri2025whatsintheimage", "title": "What's in the Image? A Deep-Dive into the Vision of Vision Language Models"},
    {"key": "chen2024rightwayevaluatingvlm", "title": "Are We on the Right Way for Evaluating Large Vision-Language Models?"},
    {"key": "guan2024hallusionbench", "title": "HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models"},
    {"key": "liu2024mmbench", "title": "MMBench: Is Your Multi-Modal Model an All-Around Player?"},
    {"key": "yue2024mmmu", "title": "MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI"},
    {"key": "duan2024vlmevalkit", "title": "VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models"},
    {"key": "radford2021clip", "arxiv": "2103.00020"},
    {"key": "liu2023visualinstructiontuning", "arxiv": "2304.08485"},
    {"key": "alayrac2022flamingo", "arxiv": "2204.14198"},
    {"key": "chen2024internvl", "title": "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"},
    {"key": "zhu2025internvl3", "arxiv": "2504.10479"},
    {"key": "bai2025qwen25vl", "arxiv": "2502.13923"},
    {"key": "wang2024qwen2vl", "arxiv": "2409.12191"},
    {"key": "bai2023qwenvl", "arxiv": "2308.12966"},
    {"key": "lu2024ovis", "arxiv": "2405.20797"},
    {"key": "wu2024deepseekvl2", "arxiv": "2412.10302"},
    {"key": "lu2024deepseekvl", "arxiv": "2403.05525"},
    {"key": "oquab2024dinov2", "title": "DINOv2: Learning Robust Visual Features without Supervision"},
    {"key": "tong2024cambrian1", "title": "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs"},
    {"key": "chen2025spatialreasoninghard", "title": "Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas"},
    {"key": "yao2023react", "arxiv": "2210.03629"},
    {"key": "shinn2023reflexion", "arxiv": "2303.11366"},
    {"key": "madaan2023selfrefine", "arxiv": "2303.17651"},
    {"key": "schick2023toolformer", "arxiv": "2302.04761"},
    {"key": "lewis2020rag", "arxiv": "2005.11401"},
    {"key": "guu2020realm", "arxiv": "2002.08909"},
    {"key": "wei2022cot", "arxiv": "2201.11903"},
    {"key": "dosovitskiy2021vit", "arxiv": "2010.11929"},
    {"key": "su2024roformer", "arxiv": "2104.09864"},
    {"key": "shazeer2017moe", "arxiv": "1701.06538"},
    {"key": "dao2024flashattention2", "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"},
    {"key": "wolf2020transformers", "title": "Transformers: State-of-the-Art Natural Language Processing"},
    {"key": "noreen1989computerintensive", "title": "Computer-Intensive Methods for Testing Hypotheses"},
]


def fetch_url(url: str, headers: dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_bibtex_via_doi(doi: str) -> str:
    return fetch_url(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex", "User-Agent": "Mozilla/5.0"},
    )


def fetch_crossref_item(title: str) -> dict | None:
    query = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?rows=5&query.title={query}"
    raw = fetch_url(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(raw)
    items = payload.get("message", {}).get("items", [])
    title_norm = re.sub(r"[^a-z0-9]+", "", title.lower())
    for item in items:
        item_title = " ".join(item.get("title", []))
        if re.sub(r"[^a-z0-9]+", "", item_title.lower()) == title_norm:
            return item
    return items[0] if items else None


def normalize_bibtex_key(bibtex: str, key: str) -> str:
    return re.sub(r"@\w+\{[^,]+,", lambda m: m.group(0).split("{")[0] + "{" + key + ",", bibtex, count=1)


def main() -> None:
    output_entries: list[str] = []
    failures: list[dict] = []

    for ref in REFERENCES:
        try:
            if "arxiv" in ref:
                doi = f"10.48550/arXiv.{ref['arxiv']}"
                bibtex = fetch_bibtex_via_doi(doi)
            else:
                item = fetch_crossref_item(ref["title"])
                if not item or "DOI" not in item:
                    raise RuntimeError("DOI not found via Crossref")
                bibtex = fetch_bibtex_via_doi(item["DOI"])
            bibtex = normalize_bibtex_key(bibtex.strip(), ref["key"])
            output_entries.append(bibtex)
            time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": ref["key"], "title": ref.get("title", ref.get("arxiv")), "error": str(exc)})

    out_path = PAPER_ROOT / "reference.bib"
    out_path.write_text("\n\n".join(output_entries) + "\n", encoding="utf-8")
    (PAPER_ROOT / "data" / "reference_fetch_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Fetched {len(output_entries)} references; failures: {len(failures)}")


if __name__ == "__main__":
    main()
