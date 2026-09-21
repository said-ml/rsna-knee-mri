from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/workspace")
GOLD_PATH = PROJECT_ROOT / "data" / "reports" / "report_001_gold.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reports" / "report_001_v2_predictions.csv"


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
# Text normalization
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter.

    We intentionally avoid a heavyweight NLP dependency here.
    Reports contain bullets, headings, and short fragments, so we
    split on sentence punctuation and newlines.
    """
    text = normalize_text(text)

    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)

    return [
        p.strip(" -*>\t")
        for p in pieces
        if p.strip(" -*>\t")
    ]


# ---------------------------------------------------------------------
# Language hint
# ---------------------------------------------------------------------

def language_hint(text: str) -> str:
    t = text.lower()

    scores = {
        "Spanish": 0,
        "French": 0,
        "German": 0,
        "Turkish": 0,
    }

    spanish = [
        "leve", "derrame", "articular", "sinovitis", "rodilla",
        "menisco", "fractura", "hueso", "ligamento",
        "hallazgos", "constataciones",
    ]

    french = [
        "genou", "ménisque", "fracture", "épanchement",
        "ligament", "constatations", "aucune", "œdème",
    ]

    german = [
        "knie", "meniskus", "band", "fraktur", "erguss",
        "befund", "keine", "knochen",
    ]

    turkish = [
        "diz", "menisküs", "bağ", "kırık", "eklem",
        "ödem", "sinovit",
    ]

    for word in spanish:
        if word in t:
            scores["Spanish"] += 1

    for word in french:
        if word in t:
            scores["French"] += 1

    for word in german:
        if word in t:
            scores["German"] += 1

    for word in turkish:
        if word in t:
            scores["Turkish"] += 1

    best = max(scores, key=scores.get)

    if scores[best] >= 2:
        return best

    return "English/Other"


# ---------------------------------------------------------------------
# Negation / uncertainty
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Negation / uncertainty
# ---------------------------------------------------------------------

NEGATIVE_PATTERNS = [
    r"\bno tear\b",
    r"\bno obvious tear\b",
    r"\bnot torn\b",
    r"\bwithout tear\b",
    r"\bno evidence of tear\b",
    r"\bno injury\b",
    r"\bwithout injury\b",
    r"\bintact\b",
    r"\bnormal\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bsuspicious\b",
    r"\bsuspicious for\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\bprobably\b",
    r"\bquestionable\b",
    r"\bcannot exclude\b",
    r"\bcan'?t exclude\b",
    r"\br/o\b",
    r"\brule out\b",
    r"\bmay represent\b",
    r"\bmay be\b",
    r"\bmay indicate\b",
    r"\bconcerning for\b",
    r"\blikely\b",
]


def has_uncertainty(text: str) -> bool:
    t = text.lower()

    return any(
        re.search(pattern, t)
        for pattern in UNCERTAINTY_PATTERNS
    )


def negation_before_match(
    text: str,
    match_start: int,
    window: int = 60,
) -> bool:
    """
    Detect local negation before the target phrase.

    A local window is used so that a negative statement elsewhere
    in the sentence does not automatically negate the target.
    """

    start = max(0, match_start - window)
    prefix = text[start:match_start].lower()

    return any(
        re.search(pattern, prefix)
        for pattern in NEGATIVE_PATTERNS
    )


def target_state(sentence: str, match) -> str | None:
    """
    Return:

        "positive"   explicit positive
        "negative"   explicit negative
        "uncertain"  uncertain evidence
        None         no usable evidence

    v2 policy:
        uncertainty has priority over positive/negative evidence.
    """

    s = normalize_text(sentence)

    # -------------------------------------------------------------
    # C-policy:
    # uncertain language is NOT converted into a positive label.
    #
    # Examples:
    #   suspicious tear -> uncertain
    #   R/O tear        -> uncertain
    #   likely contusion -> uncertain
    # -------------------------------------------------------------
    if has_uncertainty(s):
        return "uncertain"

    # -------------------------------------------------------------
    # Requested v2 A-fixes
    # -------------------------------------------------------------

    # "without surfacing tear"
    if re.search(r"\bwithout\b.{0,80}\btear\b", s, flags=re.IGNORECASE):
        return "negative"

    # "No evidence of tear ..."
    if re.search(
        r"\bno evidence of\b.{0,80}\btear\b",
        s,
        flags=re.IGNORECASE,
    ):
        return "negative"

    # Existing explicit-negative terminology.
    if any(
        re.search(pattern, s, flags=re.IGNORECASE)
        for pattern in NEGATIVE_PATTERNS
    ):
        return "negative"

    # Explicit positive target match.
    if match:
        return "positive"

    return None

# ---------------------------------------------------------------------
# Generic matcher
# ---------------------------------------------------------------------

def find_state(
    sentence: str,
    patterns: list[str],
) -> tuple[str | None, str | None]:
    """
    Return (state, matched_text).

    The first explicit target occurrence is used.
    """

    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match is None:
            continue

        state = target_state(sentence, match)

        return state, match.group(0)

    return None, None


# ---------------------------------------------------------------------
# Target-specific extraction
# ---------------------------------------------------------------------

def extract_target(sentence: str, target: str):
    s = sentence

    patterns = {

        "ACL": [
            r"\banterior cruciate ligament\b",
            r"\bACL\b",
        ],

        "MCL": [
            r"\bmedial collateral ligament\b",
            r"\bMCL\b",
        ],

        "Medial Meniscus": [
            r"\bmedial meniscus\b",
        ],

        "Lateral Meniscus": [
            r"\blateral meniscus\b",
        ],

        "Effusion": [
            r"\beffusion\b",
            r"\bjoint effusion\b",
            r"\bknee effusion\b",
            r"\bderrame articular\b",
            r"\bépanchement\b",
            r"\bgelenkerguss\b",
        ],

        "Synovitis": [
            r"\bsynovitis\b",
            r"\bsynovial hypertrophy\b",
            r"\bsynovial thickening\b",
            r"\bhypertrophy of the synovium\b",
            r"\bthickening of the synovium\b",
            r"\bsinovitis\b",
        ],

        "Baker's": [
            r"\bbaker'?s cyst\b",
            r"\bpopliteal cyst\b",
            r"\bquiste popl[íi]teo\b",
        ],

        "Contusion": [
            r"\bbone contusion\b",
            r"\bbone bruise\b",
            r"\bcontusion\b",
        ],

        "Fracture": [
            r"\bfracture\b",
            r"\bfractures\b",
            r"\bfracture line\b",
            r"\bfractura\b",
            r"\bfraktur\b",
        ],
    }

    if target in patterns:
        return find_state(s, patterns[target])

    return None, None


# ---------------------------------------------------------------------
# Meniscus interpretation
# ---------------------------------------------------------------------

def extract_meniscus(sentence: str, target: str):
    if target == "Medial Meniscus":
        name = r"medial meniscus"

    elif target == "Lateral Meniscus":
        name = r"lateral meniscus"

    else:
        return None, None

    # Explicit negative statements.
    negative_patterns = [
        rf"\b{name}\b[^.]*\bno tear\b",
        rf"\b{name}\b[^.]*\bno obvious tear\b",
        rf"\b{name}\b[^.]*\bnot torn\b",
        rf"\b{name}\b[^.]*\bwithout (?:a )?tear\b",
        rf"\b{name}\b[^.]*\bno evidence of tear\b",
    ]

    for pattern in negative_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match:
            return "negative", match.group(0)

    # Uncertain tear.
    uncertain_patterns = [
        rf"\b{name}\b[^.]*\bsuspicious\b[^.]*\btear\b",
        rf"\b{name}\b[^.]*\bpossible\b[^.]*\btear\b",
        rf"\b{name}\b[^.]*\bquestionable\b[^.]*\btear\b",
        rf"\b{name}\b[^.]*\br/o\b[^.]*\btear\b",
        rf"\b{name}\b[^.]*\bcannot exclude\b[^.]*\btear\b",
    ]

    for pattern in uncertain_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match:
            return "uncertain", match.group(0)

    # Explicit tear.
    positive_patterns = [
        rf"\b{name}\b[^.]*\btear\b",
        rf"\btear\b[^.]*\b{name}\b",
    ]

    for pattern in positive_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match:
            return "positive", match.group(0)

    return None, None


# ---------------------------------------------------------------------
# Ligament interpretation
# ---------------------------------------------------------------------

def extract_ligament(sentence: str, target: str):
    if target == "ACL":
        name = r"(?:anterior cruciate ligament|ACL)"

    elif target == "MCL":
        name = r"(?:medial collateral ligament|MCL)"

    else:
        return None, None

    # Explicit negatives.
    negative_patterns = [
        rf"{name}[^.]*\bno tear\b",
        rf"{name}[^.]*\bintact\b",
        rf"{name}[^.]*\bnormal\b",
        rf"{name}[^.]*\bwithout tear\b",
        rf"{name}[^.]*\bno injury\b",
    ]

    for pattern in negative_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match:
            return "negative", match.group(0)

    # Uncertain.
    uncertain_patterns = [
        rf"{name}[^.]*\bsuspicious\b",
        rf"{name}[^.]*\bpossible\b",
        rf"{name}[^.]*\bquestionable\b",
        rf"{name}[^.]*\bcannot exclude\b",
        rf"{name}[^.]*\br/o\b",
    ]

    for pattern in uncertain_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match:
            return "uncertain", match.group(0)

    # Explicit tear.
    positive_patterns = [
        rf"{name}[^.]*\btear\b",
        rf"\btear\b[^.]*{name}",
        rf"{name}[^.]*\brupture\b",
        rf"\brupture\b[^.]*{name}",
    ]

    for pattern in positive_patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if match:
            return "positive", match.group(0)

    return None, None


# ---------------------------------------------------------------------
# OA
# ---------------------------------------------------------------------

def extract_oa(sentence: str, target: str):
    """
    v1 deliberately does NOT map generic cartilage findings such as:

        chondrosis
        cartilage loss
        cartilage defect
        cartilage fissuring

    to OA.

    We require explicit osteoarthritis/OA terminology.
    """

    s = sentence.lower()

    explicit_oa = [
        r"\bosteoarthritis\b",
        r"\bosteoarthritic\b",
        r"\bdegenerative osteoarthritis\b",
        r"\bOA\b",
    ]

    if not any(re.search(p, sentence, flags=re.IGNORECASE)
               for p in explicit_oa):
        return None, None

    if target == "Medial OA":
        compartment_patterns = [
            r"\bmedial compartment\b",
            r"\bmedial tibiofemoral\b",
            r"\bmedial tibiofemoral compartment\b",
        ]

    elif target == "Lateral OA":
        compartment_patterns = [
            r"\blateral compartment\b",
            r"\blateral tibiofemoral\b",
            r"\blateral tibiofemoral compartment\b",
        ]

    elif target == "PF OA":
        compartment_patterns = [
            r"\bpatellofemoral\b",
            r"\bpatellofemoral compartment\b",
            r"\bpf compartment\b",
        ]

    else:
        return None, None

    for cp in compartment_patterns:
        match = re.search(cp, sentence, flags=re.IGNORECASE)

        if match:
            state = target_state(sentence, match)

            return state, sentence.strip()

    return None, None


# ---------------------------------------------------------------------
# Full extraction
# ---------------------------------------------------------------------

def extract_report(report: str) -> dict:
    sentences = split_sentences(report)

    result = {
        "language_hint": language_hint(report),
    }

    for target in TARGETS:
        result[target] = None
        result[f"{target}__evidence"] = ""
        result[f"{target}__confidence"] = ""

    for target in TARGETS:

        candidates = []

        for sentence in sentences:

            if target in {"ACL", "MCL"}:
                state, evidence = extract_ligament(sentence, target)

            elif target in {"Medial Meniscus", "Lateral Meniscus"}:
                state, evidence = extract_meniscus(sentence, target)

            elif target in {"Medial OA", "Lateral OA", "PF OA"}:
                state, evidence = extract_oa(sentence, target)

            else:
                state, evidence = extract_target(sentence, target)

            if state is not None:
                candidates.append((state, evidence, sentence))

        if not candidates:
            continue

        # -------------------------------------------------------------
        # Resolution policy
        #
        # Explicit positive/negative evidence outranks uncertainty.
        # If both positive and negative explicit evidence exist,
        # preserve the conflict as unknown rather than forcing a label.
        # -------------------------------------------------------------

        positive = [x for x in candidates if x[0] == "positive"]
        negative = [x for x in candidates if x[0] == "negative"]
        uncertain = [x for x in candidates if x[0] == "uncertain"]

        if positive and negative:
            value = None
            confidence = "conflict"

            evidence = " || ".join(
                x[2] for x in candidates
            )

        elif positive:
            value = 1
            confidence = "high"

            evidence = " || ".join(
                x[2] for x in positive
            )

        elif negative:
            value = 0
            confidence = "high"

            evidence = " || ".join(
                x[2] for x in negative
            )

        elif uncertain:
            value = None
            confidence = "uncertain"

            evidence = " || ".join(
                x[2] for x in uncertain
            )

        else:
            continue

        result[target] = value
        result[f"{target}__evidence"] = evidence
        result[f"{target}__confidence"] = confidence

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    gold = pd.read_csv(GOLD_PATH)

    print(f"Gold reports: {len(gold)}")

    rows = []

    for _, row in gold.iterrows():

        result = extract_report(row["Report"])

        result["StudyInstanceUID"] = row["StudyInstanceUID"]

        rows.append(result)

    predictions = pd.DataFrame(rows)

    columns = (
        ["StudyInstanceUID", "language_hint"]
        + TARGETS
        + [f"{t}__evidence" for t in TARGETS]
        + [f"{t}__confidence" for t in TARGETS]
    )

    predictions = predictions[columns]

    predictions.to_csv(OUTPUT_PATH, index=False)

    print()
    print("REPORT-001 v2 complete")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Shape: {predictions.shape}")

    print()
    print("Coverage:")

    for target in TARGETS:
        mask = predictions[target].notna()

        coverage = mask.mean()

        positive = (predictions.loc[mask, target] == 1).sum()
        negative = (predictions.loc[mask, target] == 0).sum()

        print(
            f"{target:18s} "
            f"coverage={coverage:6.1%} "
            f"positive={positive:2d} "
            f"negative={negative:2d}"
        )


if __name__ == "__main__":
    main()