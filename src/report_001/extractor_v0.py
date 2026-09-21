from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# REPORT-001 v0
# Conservative report-to-label extractor
#
# Output:
#   1 = explicit positive evidence
#   0 = explicit negative evidence
#   NaN = insufficient / ambiguous evidence
#
# Important:
# - Evidence is retained.
# - No absence-of-mention => negative.
# - No severity inference.
# - No clinical inference.
# - v0 intentionally favors precision over coverage.
# ============================================================


PROJECT_ROOT = Path("/workspace")

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "report_001_gold.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "report_001_v0_predictions.csv"
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


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:
    """
    Conservative normalization.

    We preserve the original report separately.
    This normalized version is only used for matching.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize common Unicode variants.
    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence segmentation.

    Medical reports often use bullets and line breaks rather than
    conventional punctuation, so we split on punctuation and
    common report bullet boundaries.
    """
    text = normalize_text(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+|"
        r"\s+[>-]\s+|"
        r"\s*\n\s*",
        text,
    )

    return [p.strip() for p in parts if p.strip()]


# ============================================================
# Language hints
# ============================================================

def detect_language_hint(text: str) -> str:
    """
    Very conservative language hint.

    This is NOT intended to be a production language detector.
    It only records obvious lexical evidence useful during audit.
    """

    text_lower = normalize_text(text).lower()

    spanish_markers = [
        "hallazgos",
        "impresión",
        "derrame articular",
        "menisco",
        "rotura",
        "ligamentos",
        "sin signos",
        "sin alteraciones",
        "condropatía",
    ]

    german_markers = [
        "knie",
        "meniskus",
        "band",
        "fraktur",
        "erguss",
    ]

    french_markers = [
        "constatations",
        "fractures",
        "aucune",
        "ménisque",
        "ligament",
    ]

    turkish_markers = [
        "menisküs",
        "bağ",
        "yırtık",
        "diz",
        "eklem",
    ]

    scores = {
        "Spanish": sum(x in text_lower for x in spanish_markers),
        "German": sum(x in text_lower for x in german_markers),
        "French": sum(x in text_lower for x in french_markers),
        "Turkish": sum(x in text_lower for x in turkish_markers),
    }

    best_language = max(scores, key=scores.get)

    if scores[best_language] >= 2:
        return best_language

    return "English/Other"


# ============================================================
# Pattern helpers
# ============================================================

def any_match(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(0)
    return None


# ============================================================
# Target-specific extraction
# ============================================================

def extract_acl(sentence: str):
    """
    ACL:
      explicit intact/normal -> 0
      explicit tear -> 1

    Deliberately does NOT classify vague/minor 'injury' language.
    """

    positive = [
        r"\bcomplete tear\b.*\banterior cruciate ligament\b",
        r"\btear\b.*\banterior cruciate ligament\b",
        r"\banterior cruciate ligament\b.*\bcomplete tear\b",
        r"\banterior cruciate ligament\b.*\btear\b",
        r"\bACL\b.*\bcomplete tear\b",
        r"\bACL\b.*\btear\b",
        r"\bcomplete tear\b.*\bACL\b",
    ]

    negative = [
        r"\bACL\b.*\bintact\b",
        r"\bACL\b.*\bnormal\b",
        r"\banterior cruciate ligament\b.*\bintact\b",
        r"\banterior cruciate ligament\b.*\bnormal\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


def extract_mcl(sentence: str):
    """
    MCL:
      explicit intact/normal -> 0
      explicit tear -> 1

    Sprain/injury alone is deliberately not classified in v0.
    """

    positive = [
        r"\bcomplete tear\b.*\bMCL\b",
        r"\btear\b.*\bMCL\b",
        r"\bMCL\b.*\bcomplete tear\b",
        r"\bMCL\b.*\btear\b",
        r"\bmedial collateral ligament\b.*\bcomplete tear\b",
        r"\bmedial collateral ligament\b.*\btear\b",
    ]

    negative = [
        r"\bMCL\b.*\bintact\b",
        r"\bMCL\b.*\bnormal\b",
        r"\bmedial collateral ligament\b.*\bintact\b",
        r"\bmedial collateral ligament\b.*\bnormal\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


def extract_meniscus(sentence: str, side: str):
    """
    Explicit meniscal tear only.

    Degeneration without a tear is NOT positive.
    """

    if side == "medial":
        target = "Medial Meniscus"
        terms = [
            r"\bmedial meniscus\b",
            r"\bmeniscus medialis\b",
        ]
    else:
        target = "Lateral Meniscus"
        terms = [
            r"\blateral meniscus\b",
            r"\bmeniscus lateralis\b",
        ]

    target_pattern = r"(?:" + "|".join(terms) + r")"

    positive = [
        rf"\btear\b[^.]*{target_pattern}",
        rf"{target_pattern}[^.]*\btear\b",
        rf"\bradial tear\b[^.]*{target_pattern}",
        rf"\bcomplex tear\b[^.]*{target_pattern}",
        rf"{target_pattern}[^.]*\bcomplex tear\b",
    ]

    negative = [
        rf"{target_pattern}[^.]*\bnot torn\b",
        rf"\bno\b[^.]*\btear\b[^.]*{target_pattern}",
        rf"{target_pattern}[^.]*\bno frank tear\b",
        rf"{target_pattern}[^.]*\bwithout\b[^.]*\btear\b",
        rf"{target_pattern}[^.]*\bintact\b",
        rf"{target_pattern}[^.]*\bnormal\b",
    ]

    if any_match(sentence, negative):
        return target, 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return target, 1, first_match(sentence, positive), "high"

    return target, None, None, None


def extract_effusion(sentence: str):
    positive = [
        r"\bjoint effusion\b",
        r"\bknee effusion\b",
        r"\bjoint fluid\b",
        r"\bfluid accumulation\b",
        r"\bderrame articular\b",
        r"\bderrame\b",
    ]

    negative = [
        r"\bno knee effusion\b",
        r"\bno joint effusion\b",
        r"\bno effusion\b",
        r"\bsin derrame\b",
        r"\bsin derrame articular\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


def extract_synovitis(sentence: str):
    positive = [
        r"\bsynovitis\b",
        r"\bsynovial thickening\b",
        r"\bthickened synovial tissue\b",
        r"\bthickening of the synovial membrane\b",
        r"\bsynovial membrane thickening\b",
    ]

    negative = [
        r"\bno synovitis\b",
        r"\bwithout synovitis\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


def extract_bakers(sentence: str):
    positive = [
        r"\bBaker'?s cyst\b",
        r"\bpopliteal cyst\b",
        r"\bpopliteal cyst measuring\b",
        r"\bquiste popl[ií]teo\b",
    ]

    negative = [
        r"\bno Baker'?s cyst\b",
        r"\bno popliteal cyst\b",
        r"\bno popliteal cysts\b",
        r"\bno quiste popl[ií]teo\b",
        r"\bsin quistes popl[ií]teos\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


def extract_contusion(sentence: str):
    """
    v0 only accepts explicit bone/bone-contusion language.

    Generic marrow edema is deliberately NOT enough.
    """

    positive = [
        r"\bbone contusion\b",
        r"\bbone bruise\b",
        r"\bbone marrow contusion\b",
        r"\bcontusion\b.*\bbone\b",
        r"\bcontusion\b.*\bbone marrow\b",
        r"\bcontusi[oó]n\b.*\b[oó]sea\b",
    ]

    negative = [
        r"\bno bone contusion\b",
        r"\bno bone bruise\b",
        r"\bno contusion\b",
        r"\bsin contusi[oó]n\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


def extract_fracture(sentence: str):
    positive = [
        r"\bfracture\b",
        r"\bfractures\b",
        r"\bosteochondral fracture\b",
        r"\binsufficiency fracture\b",
        r"\bfractura\b",
        r"\bfracturas\b",
    ]

    negative = [
        r"\bno fracture\b",
        r"\bno fractures\b",
        r"\bwithout fracture\b",
        r"\bsin fractura\b",
        r"\bsin fracturas\b",
        r"\bfractures\s*:\s*none\b",
    ]

    if any_match(sentence, negative):
        return 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return 1, first_match(sentence, positive), "high"

    return None, None, None


# ============================================================
# OA extraction
# ============================================================

def extract_oa(sentence: str, compartment: str):
    """
    v0 is intentionally conservative.

    We only classify explicit osteoarthritis / degenerative
    disease terminology tied to the target compartment.

    We do NOT yet convert every cartilage defect/fissure into OA.
    """

    if compartment == "medial":
        target = "Medial OA"
        compartment_patterns = [
            r"\bmedial compartment\b",
            r"\bmedial femoral condyle\b",
            r"\bmedial tibial plateau\b",
            r"\bmedial tibiofemoral\b",
        ]

    elif compartment == "lateral":
        target = "Lateral OA"
        compartment_patterns = [
            r"\blateral compartment\b",
            r"\blateral femoral condyle\b",
            r"\blateral tibial plateau\b",
            r"\blateral tibiofemoral\b",
        ]

    else:
        target = "PF OA"
        compartment_patterns = [
            r"\bpatellofemoral\b",
            r"\bpatellar facet\b",
            r"\btrochlea\b",
            r"\btrochlear\b",
        ]

    compartment_pattern = r"(?:" + "|".join(compartment_patterns) + r")"

    positive = [
        rf"{compartment_pattern}[^.]*\bosteoarthritis\b",
        rf"\bosteoarthritis\b[^.]*{compartment_pattern}",
        rf"{compartment_pattern}[^.]*\bOA\b",
        rf"\bOA\b[^.]*{compartment_pattern}",
        rf"{compartment_pattern}[^.]*\bdegenerative\b",
        rf"{compartment_pattern}[^.]*\bchondrosis\b",
        rf"\bchondrosis\b[^.]*{compartment_pattern}",
        rf"{compartment_pattern}[^.]*\bcondropat[ií]a\b",
    ]

    negative = [
        rf"{compartment_pattern}[^.]*\bno\b[^.]*\bchondrosis\b",
        rf"{compartment_pattern}[^.]*\bno\b[^.]*\bchondral injury\b",
        rf"{compartment_pattern}[^.]*\bwithout\b[^.]*\bchondrosis\b",
        rf"{compartment_pattern}[^.]*\bwithout\b[^.]*\bchondral lesion\b",
    ]

    if any_match(sentence, negative):
        return target, 0, first_match(sentence, negative), "high"

    if any_match(sentence, positive):
        return target, 1, first_match(sentence, positive), "medium"

    return target, None, None, None


# ============================================================
# Sentence-level extraction
# ============================================================

def extract_report(report: str) -> dict:
    """
    Extract all 12 targets from one report.

    Multiple sentences may contain evidence.
    If both positive and negative evidence occur, the evidence
    is retained rather than silently resolving the conflict.
    """

    sentences = split_sentences(report)

    result = {
        target: None
        for target in TARGETS
    }

    evidence = {
        target: []
        for target in TARGETS
    }

    confidence = {
        target: None
        for target in TARGETS
    }

    for sentence in sentences:

        # ----------------------------------------------------
        # ACL
        # ----------------------------------------------------
        value, match, conf = extract_acl(sentence)

        if value is not None:
            result["ACL"] = value
            evidence["ACL"].append(sentence)
            confidence["ACL"] = conf

        # ----------------------------------------------------
        # MCL
        # ----------------------------------------------------
        value, match, conf = extract_mcl(sentence)

        if value is not None:
            result["MCL"] = value
            evidence["MCL"].append(sentence)
            confidence["MCL"] = conf

        # ----------------------------------------------------
        # Menisci
        # ----------------------------------------------------
        target, value, match, conf = extract_meniscus(
            sentence,
            "medial",
        )

        if value is not None:
            result[target] = value
            evidence[target].append(sentence)
            confidence[target] = conf

        target, value, match, conf = extract_meniscus(
            sentence,
            "lateral",
        )

        if value is not None:
            result[target] = value
            evidence[target].append(sentence)
            confidence[target] = conf

        # ----------------------------------------------------
        # Effusion
        # ----------------------------------------------------
        value, match, conf = extract_effusion(sentence)

        if value is not None:
            result["Effusion"] = value
            evidence["Effusion"].append(sentence)
            confidence["Effusion"] = conf

        # ----------------------------------------------------
        # Synovitis
        # ----------------------------------------------------
        value, match, conf = extract_synovitis(sentence)

        if value is not None:
            result["Synovitis"] = value
            evidence["Synovitis"].append(sentence)
            confidence["Synovitis"] = conf

        # ----------------------------------------------------
        # Baker's
        # ----------------------------------------------------
        value, match, conf = extract_bakers(sentence)

        if value is not None:
            result["Baker's"] = value
            evidence["Baker's"].append(sentence)
            confidence["Baker's"] = conf

        # ----------------------------------------------------
        # Contusion
        # ----------------------------------------------------
        value, match, conf = extract_contusion(sentence)

        if value is not None:
            result["Contusion"] = value
            evidence["Contusion"].append(sentence)
            confidence["Contusion"] = conf

        # ----------------------------------------------------
        # Fracture
        # ----------------------------------------------------
        value, match, conf = extract_fracture(sentence)

        if value is not None:
            result["Fracture"] = value
            evidence["Fracture"].append(sentence)
            confidence["Fracture"] = conf

        # ----------------------------------------------------
        # OA
        # ----------------------------------------------------
        for compartment in ["medial", "lateral", "pf"]:
            target, value, match, conf = extract_oa(
                sentence,
                compartment,
            )

            if value is not None:
                result[target] = value
                evidence[target].append(sentence)
                confidence[target] = conf

    # Convert evidence lists to readable strings.
    for target in TARGETS:
        evidence[target] = " || ".join(
            dict.fromkeys(evidence[target])
        )

    return {
        "labels": result,
        "evidence": evidence,
        "confidence": confidence,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(f"Loading: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Reports: {len(df)}")

    rows = []

    for _, row in df.iterrows():

        report = row["Report"]

        extracted = extract_report(report)

        output = {
            "StudyInstanceUID": row["StudyInstanceUID"],
            "language_hint": detect_language_hint(report),
        }

        for target in TARGETS:
            output[target] = extracted["labels"][target]

        for target in TARGETS:
            output[f"{target}__evidence"] = (
                extracted["evidence"][target]
            )

        for target in TARGETS:
            output[f"{target}__confidence"] = (
                extracted["confidence"][target]
            )

        rows.append(output)

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("REPORT-001 v0 complete")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {result.shape}")

    print()
    print("Coverage:")
    for target in TARGETS:
        coverage = result[target].notna().mean()
        positives = (result[target] == 1).sum()
        negatives = (result[target] == 0).sum()

        print(
            f"{target:20s} "
            f"coverage={coverage:6.1%} "
            f"positive={positives:2d} "
            f"negative={negatives:2d}"
        )


if __name__ == "__main__":
    main()
    