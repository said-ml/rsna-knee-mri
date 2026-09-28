from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/workspace")
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "baseline_003"
    / "strict_evidence_58.csv"
)

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Conservative sentence splitting.
    Reports are mostly prose; retain each sentence as an evidence unit.
    """
    text = normalize(text)

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------

UNCERTAINTY_PATTERNS = [
    r"\br/o\b",
    r"\brule\s+out\b",
    r"\bsuspicious\b",
    r"\bsuspicious\s+for\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bquestionable\b",
    r"\bcannot\s+exclude\b",
    r"\bcan't\s+exclude\b",
    r"\bmay\s+represent\b",
    r"\bmay\s+be\b",
    r"\bmay\s+indicate\b",
    r"\bconcerning\s+for\b",
    r"\blikely\b",
    r"\bprobable\b",
    r"\bprobably\b",
]


def is_uncertain(sentence: str) -> bool:
    return any(
        re.search(p, sentence, flags=re.IGNORECASE)
        for p in UNCERTAINTY_PATTERNS
    )


# ---------------------------------------------------------------------
# Target terminology
# ---------------------------------------------------------------------

TARGET_PATTERNS = {
    "ACL": [
        r"\banterior\s+cruciate\s+ligament\b",
        r"\bacl\b",
    ],
    "MCL": [
        r"\bmedial\s+collateral\s+ligament\b",
        r"\bmcl\b",
    ],
    "Medial Meniscus": [
        r"\bmedial\s+meniscus\b",
    ],
    "Lateral Meniscus": [
        r"\blateral\s+meniscus\b",
    ],
    "Medial OA": [
        r"\bmedial\s+compartment\s+osteoarthritis\b",
        r"\bmedial\s+compartment\s+osteoarthritic\b",
        r"\bmedial\s+compartment\s+oa\b",
        r"\bosteoarthritis\b[^.]{0,100}\bmedial\s+compartment\b",
        r"\bmedial\s+compartment\b[^.]{0,100}\bosteoarthritis\b",
        r"\bmedial\s+compartment\b[^.]{0,100}\boa\b",
    ],
    "Lateral OA": [
        r"\blateral\s+compartment\s+osteoarthritis\b",
        r"\blateral\s+compartment\s+osteoarthritic\b",
        r"\blateral\s+compartment\s+oa\b",
        r"\bosteoarthritis\b[^.]{0,100}\blateral\s+compartment\b",
        r"\blateral\s+compartment\b[^.]{0,100}\bosteoarthritis\b",
        r"\blateral\s+compartment\b[^.]{0,100}\boa\b",
    ],
    "PF OA": [
        r"\bpatellofemoral\s+osteoarthritis\b",
        r"\bpatellofemoral\s+osteoarthritic\b",
        r"\bpatellofemoral\s+oa\b",
        r"\bpatellofemoral\s+compartment\b[^.]{0,100}\bosteoarthritis\b",
        r"\bpatellofemoral\s+compartment\b[^.]{0,100}\boa\b",
        r"\bosteoarthritis\b[^.]{0,100}\bpatellofemoral\b",
        r"\boa\b[^.]{0,100}\bpatellofemoral\b",
    ],
    "Effusion": [
        r"\beffusion\b",
        r"\bjoint\s+effusion\b",
        r"\bknee\s+effusion\b",
        r"\bderrame\s+articular\b",
        r"\bderrame\b",
        r"\bépanchement\b",
        r"\bepanchement\b",
        r"\bgelenkerguss\b",
    ],
    "Synovitis": [
        r"\bsynovitis\b",
        r"\bsynovial\s+hypertrophy\b",
        r"\bsynovial\s+thickening\b",
        r"\bhypertrophy\s+of\s+the\s+synovium\b",
        r"\bthickening\s+of\s+the\s+synovium\b",
        r"\bsinovitis\b",
    ],
    "Baker's": [
        r"\bbaker'?s\s+cyst\b",
        r"\bpopliteal\s+cyst\b",
        r"\bquiste\s+popl[ií]teo\b",
    ],
    "Contusion": [
        r"\bbone\s+contusion\b",
        r"\bbone\s+bruise\b",
        r"\bcontusion\b",
        r"\bbone\s+marrow\s+contusion\b",
    ],
    "Fracture": [
        r"\bfracture\b",
        r"\bfractures\b",
        r"\bfracture\s+line\b",
        r"\bfractura\b",
        r"\bfraktur\b",
    ],
}


# ---------------------------------------------------------------------
# Positive terminology
# ---------------------------------------------------------------------

POSITIVE_PATTERNS = {
    "ACL": [
        r"\btear\b[^.]{0,100}\b(?:anterior\s+cruciate\s+ligament|acl)\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\btear\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\bru[pt]ture\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\bsprain\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\binjur(?:y|ed)\b",
    ],
    "MCL": [
        r"\btear\b[^.]{0,100}\b(?:medial\s+collateral\s+ligament|mcl)\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\btear\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\bru[pt]ture\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\bsprain\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\binjur(?:y|ed)\b",
    ],
    "Medial Meniscus": [
        r"\bmedial\s+meniscus\b[^.]{0,100}\btear\b",
        r"\btear\b[^.]{0,100}\bmedial\s+meniscus\b",
        r"\bmedial\s+meniscus\b[^.]{0,100}\b(?:rupture|injury)\b",
    ],
    "Lateral Meniscus": [
        r"\blateral\s+meniscus\b[^.]{0,100}\btear\b",
        r"\btear\b[^.]{0,100}\blateral\s+meniscus\b",
        r"\blateral\s+meniscus\b[^.]{0,100}\b(?:rupture|injury)\b",
    ],
    "Effusion": [
        r"\b(?:moderate|mild|small|large|massive|trace)\s+(?:joint\s+|knee\s+)?effusion\b",
        r"\b(?:joint\s+|knee\s+)?effusion\s+(?:is|are|was|were)\s+(?:present|seen|observed)\b",
        r"\beffusion\s+is\s+present\b",
        r"\bderrame\s+articular\b",
        r"\bépanchement\b",
        r"\bepanchement\b",
        r"\bgelenkerguss\b",
    ],
    "Synovitis": [
        r"\bsynovitis\b",
        r"\bsynovial\s+hypertrophy\b",
        r"\bsynovial\s+thickening\b",
        r"\bhypertrophy\s+of\s+the\s+synovium\b",
        r"\bthickening\s+of\s+the\s+synovium\b",
        r"\bsinovitis\b",
    ],
    "Baker's": [
        r"\bbaker'?s\s+cyst\b",
        r"\bpopliteal\s+cyst\b",
        r"\bquiste\s+popl[ií]teo\b",
    ],
    "Contusion": [
        r"\bbone\s+contusion\b",
        r"\bbone\s+bruise\b",
        r"\bcontusion\b",
        r"\bbone\s+marrow\s+contusion\b",
    ],
    "Fracture": [
        r"\bfracture\b",
        r"\bfractures\b",
        r"\bfracture\s+line\b",
        r"\bfractura\b",
        r"\bfraktur\b",
    ],
}


# ---------------------------------------------------------------------
# Explicit negative terminology
# ---------------------------------------------------------------------

NEGATIVE_PATTERNS = {
    "ACL": [
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\bintact\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\bnormal\b",
        r"\bno\s+(?:tear\s+of\s+)?(?:the\s+)?(?:anterior\s+cruciate\s+ligament|acl)\b",
        r"\bno\s+evidence\s+of\s+(?:a\s+)?(?:tear\s+of\s+)?(?:the\s+)?(?:anterior\s+cruciate\s+ligament|acl)\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\bwithout\s+(?:a\s+)?tear\b",
        r"\b(?:anterior\s+cruciate\s+ligament|acl)\b[^.]{0,100}\bpreserved\b",
    ],
    "MCL": [
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\bintact\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\bnormal\b",
        r"\bno\s+(?:tear\s+of\s+)?(?:the\s+)?(?:medial\s+collateral\s+ligament|mcl)\b",
        r"\bno\s+evidence\s+of\s+(?:a\s+)?(?:tear\s+of\s+)?(?:the\s+)?(?:medial\s+collateral\s+ligament|mcl)\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\bwithout\s+(?:a\s+)?tear\b",
        r"\b(?:medial\s+collateral\s+ligament|mcl)\b[^.]{0,100}\bpreserved\b",
    ],
    "Medial Meniscus": [
        r"\bmedial\s+meniscus\b[^.]{0,100}\bintact\b",
        r"\bmedial\s+meniscus\b[^.]{0,100}\bnormal\b",
        r"\bmedial\s+meniscus\b[^.]{0,100}\bwithout\s+(?:a\s+)?tear\b",
        r"\bno\s+(?:evidence\s+of\s+)?(?:a\s+)?tear[^.]{0,100}\bmedial\s+meniscus\b",
        r"\bno\s+tear[^.]{0,100}\bmedial\s+meniscus\b",
    ],
    "Lateral Meniscus": [
        r"\blateral\s+meniscus\b[^.]{0,100}\bintact\b",
        r"\blateral\s+meniscus\b[^.]{0,100}\bnormal\b",
        r"\blateral\s+meniscus\b[^.]{0,100}\bwithout\s+(?:a\s+)?tear\b",
        r"\bno\s+(?:evidence\s+of\s+)?(?:a\s+)?tear[^.]{0,100}\blateral\s+meniscus\b",
        r"\bno\s+tear[^.]{0,100}\blateral\s+meniscus\b",
    ],
    "Effusion": [
        r"\bno\s+(?:joint\s+|knee\s+)?effusion\b",
        r"\bno\s+significant\s+(?:joint\s+|knee\s+)?effusion\b",
        r"\bno\s+evidence\s+of\s+(?:joint\s+|knee\s+)?effusion\b",
        r"\bwithout\s+(?:joint\s+|knee\s+)?effusion\b",
        r"\bkein(?:en|e)?\s+gelenkerguss\b",
    ],
    "Synovitis": [
        r"\bno\s+synovitis\b",
        r"\bwithout\s+synovitis\b",
        r"\bno\s+evidence\s+of\s+synovitis\b",
    ],
    "Baker's": [
        r"\bno\s+baker'?s\s+cyst\b",
        r"\bno\s+popliteal\s+cyst\b",
        r"\bwithout\s+(?:a\s+)?baker'?s\s+cyst\b",
        r"\bwithout\s+(?:a\s+)?popliteal\s+cyst\b",
    ],
    "Contusion": [
        r"\bno\b[^.]{0,150}\bbone\s+bruise\b",
        r"\bno\b[^.]{0,150}\bbone\s+contusion\b",
        r"\bwithout\b[^.]{0,150}\bbone\s+bruise\b",
        r"\bwithout\b[^.]{0,150}\bbone\s+contusion\b",
    ],
    "Fracture": [
        r"\bno\s+fracture\b",
        r"\bno\s+acute\s+fracture\b",
        r"\bno\s+evidence\s+of\s+(?:a\s+)?fracture\b",
        r"\bwithout\s+(?:a\s+)?fracture\b",
        r"\bkein(?:e|en)?\s+fraktur\b",
    ],
}


# ---------------------------------------------------------------------
# OA
# ---------------------------------------------------------------------

def extract_oa(sentence: str, target: str):
    """
    Strict OA policy:
    only explicit OA/osteoarthritis terminology plus the target compartment.
    Generic chondrosis/cartilage loss/degeneration is NOT enough.
    """
    s = sentence.lower()

    if target == "Medial OA":
        compartment = r"(?:medial\s+compartment|medial)"
    elif target == "Lateral OA":
        compartment = r"(?:lateral\s+compartment|lateral)"
    elif target == "PF OA":
        compartment = r"(?:patellofemoral|pf)"

    oa = r"(?:osteoarthritis|osteoarthritic|\boa\b)"

    # Explicit negative OA statements.
    negative = [
        rf"\bno\s+{oa}[^.]*{compartment}\b",
        rf"\b{compartment}\b[^.]*\bno\s+{oa}\b",
    ]

    for p in negative:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return "negative", m.group(0)

    if is_uncertain(s):
        return None, None

    positive = [
        rf"\b{compartment}\b[^.]*{oa}",
        rf"{oa}[^.]*\b{compartment}\b",
    ]

    for p in positive:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return "positive", m.group(0)

    return None, None


# ---------------------------------------------------------------------
# Generic target extraction
# ---------------------------------------------------------------------

def extract_target(sentence: str, target: str):
    """
    Returns:
        state: positive / negative / None
        evidence: exact matched text
    """
    s = sentence

    if target in {"Medial OA", "Lateral OA", "PF OA"}:
        return extract_oa(s, target)

    # Uncertainty always wins.
    if is_uncertain(s):
        return None, None

    # Negative before positive.
    for pattern in NEGATIVE_PATTERNS.get(target, []):
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return "negative", m.group(0)

    for pattern in POSITIVE_PATTERNS.get(target, []):
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return "positive", m.group(0)

    return None, None


# ---------------------------------------------------------------------
# Per-target report extraction
# ---------------------------------------------------------------------

def extract_report(report: str) -> list[dict]:
    """
    Produce exactly one auditable record per target.

    A label is emitted only from explicit target-specific evidence.
    """
    sentences = split_sentences(report)
    records = []

    for target in TARGETS:
        candidates = []

        for sentence in sentences:
            # The sentence must actually mention the target concept.
            mentioned = any(
                re.search(p, sentence, flags=re.IGNORECASE)
                for p in TARGET_PATTERNS[target]
            )

            if not mentioned:
                continue

            state, evidence = extract_target(sentence, target)

            if state is None:
                continue

            candidates.append(
                {
                    "target": target,
                    "label": 1 if state == "positive" else 0,
                    "polarity": state,
                    "evidence": sentence,
                    "matched_evidence": evidence,
                }
            )

        # -------------------------------------------------------------
        # Resolution
        # -------------------------------------------------------------
        #
        # Explicit conflict inside a report is NOT resolved by guessing.
        # If both polarities are independently explicit, return NaN.
        #
        positive = [x for x in candidates if x["polarity"] == "positive"]
        negative = [x for x in candidates if x["polarity"] == "negative"]

        if positive and negative:
            records.append(
                {
                    "target": target,
                    "label": pd.NA,
                    "polarity": "conflict",
                    "evidence": " | ".join(
                        dict.fromkeys(x["evidence"] for x in candidates)
                    ),
                    "matched_evidence": " | ".join(
                        dict.fromkeys(
                            x["matched_evidence"] for x in candidates
                        )
                    ),
                    "confidence": "conflict",
                }
            )

        elif positive:
            # Preserve all independent positive evidence.
            records.append(
                {
                    "target": target,
                    "label": 1,
                    "polarity": "positive",
                    "evidence": " | ".join(
                        dict.fromkeys(x["evidence"] for x in positive)
                    ),
                    "matched_evidence": " | ".join(
                        dict.fromkeys(
                            x["matched_evidence"] for x in positive
                        )
                    ),
                    "confidence": "high",
                }
            )

        elif negative:
            records.append(
                {
                    "target": target,
                    "label": 0,
                    "polarity": "negative",
                    "evidence": " | ".join(
                        dict.fromkeys(x["evidence"] for x in negative)
                    ),
                    "matched_evidence": " | ".join(
                        dict.fromkeys(
                            x["matched_evidence"] for x in negative
                        )
                    ),
                    "confidence": "high",
                }
            )

        else:
            records.append(
                {
                    "target": target,
                    "label": pd.NA,
                    "polarity": "unknown",
                    "evidence": "",
                    "matched_evidence": "",
                    "confidence": "unknown",
                }
            )

    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    train = pd.read_csv(TRAIN_PATH)

    required = ["StudyInstanceUID", "Report", *TARGETS]
    missing = [c for c in required if c not in train.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    complete = train[TARGETS].notna().all(axis=1)
    report_only = train[TARGETS].isna().all(axis=1)
    mixed = ~(complete | report_only)

    assert len(train) == 4407
    assert complete.sum() == 58
    assert report_only.sum() == 4349
    assert mixed.sum() == 0

    # -------------------------------------------------------------
    # AUDIT ONLY: 58 complete-label studies
    # -------------------------------------------------------------

    rows = []

    gold = train.loc[complete, ["StudyInstanceUID", "Report", *TARGETS]]

    for _, row in gold.iterrows():
        extracted = extract_report(row["Report"])

        for result in extracted:
            target = result["target"]

            rows.append(
                {
                    "StudyInstanceUID": row["StudyInstanceUID"],
                    "target": target,
                    "gold": row[target],
                    "pred": result["label"],
                    "polarity": result["polarity"],
                    "evidence": result["evidence"],
                    "matched_evidence": result["matched_evidence"],
                    "confidence": result["confidence"],
                }
            )

    out = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("BASELINE-003 STRICT-DIRECT-EVIDENCE")
    print("=" * 70)
    print(f"Total studies:       {len(train)}")
    print(f"Complete-label:      {complete.sum()}")
    print(f"Report-only:         {report_only.sum()}")
    print(f"Mixed:               {mixed.sum()}")
    print()
    print("AUDIT POPULATION:    58")
    print(f"Output:              {OUTPUT_PATH}")
    print(f"Output rows:         {len(out)}")
    print()
    print("Predicted label counts:")
    print(
        out.groupby(["target", "pred"], dropna=False)
        .size()
        .to_string()
    )
    print()
    print("BASELINE-003 AUDIT EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()