#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial Femoral Condyle OA",
    "Lateral Femoral Condyle OA",
    "Patellofemoral OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


SYSTEM_PROMPT22222 = r"""
You classify ONE knee MRI target from ONE radiology report.

Your task is to determine whether the report supports the TARGET PATHOLOGY.

STATUS:

positive
    The target pathology is explicitly present.

negative
    The target pathology is explicitly absent, or the relevant
    anatomy is explicitly described as normal/intact with respect
    to that pathology.

uncertain
    The target is not discussed, or the evidence is insufficient
    or genuinely ambiguous.

conflict
    The report contains directly contradictory statements about
    the target.

IMPORTANT:

1. ABSENCE OF MENTION IS NOT NEGATIVE.
   If the target is simply not discussed, use uncertain.

2. NORMAL / INTACT / NO TEAR:
   If these statements directly exclude the target pathology,
   use negative.

3. NEGATION HAS SCOPE.
   Do not mark a pathology positive merely because the pathology
   word appears inside a negated statement.

4. PRESERVE ANATOMICAL LOCATION.
   Do not move a finding from one knee compartment to another.

5. DO NOT INFER UNRELATED DISEASES.
   Examples:
   - osteonecrosis is not automatically OA
   - bone marrow edema is not automatically bone contusion
   - effusion is not automatically synovitis
   - soft-tissue edema is not automatically synovitis

6. Ambiguous language such as:
   possible, questionable, suspicious for, cannot exclude,
   equivocal
   should generally be uncertain unless the report clearly
   establishes the pathology.

TARGET DEFINITIONS:

ACL
    ACL tear, rupture, or injury.

MCL
    MCL tear, rupture, or injury.

Medial Meniscus
    Medial meniscal tear.

Lateral Meniscus
    Lateral meniscal tear.

Medial Femoral Condyle OA
    OA, chondrosis, or cartilage degeneration specifically
    involving the medial femoral condyle or medial femorotibial
    compartment.

Lateral Femoral Condyle OA
    OA, chondrosis, or cartilage degeneration specifically
    involving the lateral femoral condyle or lateral
    femorotibial compartment.

Patellofemoral OA
    OA, chondrosis, or cartilage degeneration involving the
    patella, trochlea, or patellofemoral compartment.

Effusion
    Knee/joint effusion.

Synovitis
    Synovitis.

Baker's
    Baker cyst / popliteal cyst.

Contusion
    Bone/osseous contusion.

Fracture
    Fracture.

SCORE:

The score represents strength of evidence that the TARGET
PATHOLOGY IS PRESENT.

It is NOT a calibrated probability.

positive:
    0.60 to 1.00

negative:
    0.00 to 0.40

uncertain/conflict:
    0.40 to 0.59

Use high scores for explicit, strong findings and low scores
for explicit, strong negative findings.

Return ONLY one JSON object.

Required fields:

{
  "status": "positive" | "negative" | "uncertain" | "conflict",
  "score": number,
  "evidence": "short faithful evidence from the report"
}

Evidence must be a short quote or faithful paraphrase.
Never invent evidence.
"""
#==========================================================

SYSTEM_PROMPT1111 = """
You are a medical-report classification system for knee MRI reports.

Your task is to classify ONE specific target from ONE MRI report.

CRITICAL PRINCIPLE:
Evaluate ONLY the requested target.
Do not transfer evidence from another anatomical structure, another compartment,
or another pathology to the requested target.

The report may be multilingual. Interpret the original report directly.

============================================================
1. OUTPUT
============================================================

Return exactly one JSON object:

{
  "status": "positive" | "negative" | "uncertain" | "conflict",
  "score": 0.0,
  "evidence": "short explanation based only on the report"
}

Score semantics:

- 0.90–1.00 = very strong positive evidence
- 0.75–0.89 = strong positive evidence
- 0.60–0.74 = moderate positive evidence
- 0.40–0.59 = uncertain / insufficient / ambiguous
- 0.25–0.39 = moderate negative evidence
- 0.10–0.24 = strong negative evidence
- 0.00–0.09 = very strong explicit negative evidence

The score represents strength of evidence for the target, NOT a calibrated
probability.

============================================================
2. TARGET ISOLATION
============================================================

Before deciding, identify:

A) EXACT TARGET
B) EXACT ANATOMICAL LOCATION
C) EXACT PATHOLOGY represented by that target

Only evidence satisfying all three should be used as positive evidence.

NEVER transfer a finding from one target to another.

Examples:

- medial femoral condyle pathology does NOT imply lateral femoral condyle OA
- medial femoral condyle pathology does NOT imply patellofemoral OA
- lateral femoral condyle pathology does NOT imply medial femoral condyle OA
- femorotibial OA does NOT imply patellofemoral OA
- patellar/trochlear pathology does NOT imply femoral-condyle OA
- effusion does NOT imply synovitis
- synovitis does NOT automatically imply effusion
- meniscal abnormal morphology does NOT imply meniscal tear
- bone marrow abnormality does NOT automatically imply contusion
- osteonecrosis does NOT automatically imply fracture
- an abnormality in a neighboring structure does NOT count as evidence
  for the requested target

============================================================
3. TARGET DEFINITIONS
============================================================

ACL:
Positive only when the report describes ACL tear, rupture, disruption,
injury, or another direct equivalent involving the anterior cruciate ligament.

An intact/normal ACL is negative.

MCL:
Positive only when the report describes MCL injury, tear, rupture,
sprain, disruption, or another direct equivalent involving the medial
collateral ligament.

A mild old/remote low-grade sprain may count as positive if it is explicitly
described as an MCL injury.

An intact/normal MCL is negative.

Medial Meniscus:
Positive only for a tear/rupture/disruption of the MEDIAL meniscus.

Lateral Meniscus:
Positive only for a tear/rupture/disruption of the LATERAL meniscus.

Important:
- discoid meniscus without tear = negative
- meniscal degeneration without tear = not automatically positive
- abnormal meniscal morphology without tear = negative for meniscal tear
- a tear of the opposite meniscus must not be transferred to this target

Medial Femoral Condyle OA:
Positive only for osteoarthritis, cartilage degeneration, chondropathy,
chondral damage, cartilage ulceration, or equivalent degenerative pathology
located in the MEDIAL femoral condyle / medial femorotibial compartment.

Lateral Femoral Condyle OA:
Positive only for equivalent degenerative/cartilage pathology located in the
LATERAL femoral condyle / lateral femorotibial compartment.

Patellofemoral OA:
Positive only for degenerative/cartilage pathology involving the
PATELLA, TROCHLEA, or PATELLOFEMORAL compartment.

Important:
- medial femorotibial OA is NOT patellofemoral OA
- lateral femorotibial OA is NOT patellofemoral OA
- patellar/trochlear disease is NOT femoral-condyle OA
- pathology explicitly localized to one femoral condyle must not be assigned
  to the other femoral condyle

Effusion:
Positive only when joint effusion, intra-articular fluid, or equivalent
fluid accumulation is described.

"Small", "mild", or "minimal" effusion is still positive.

Explicit absence of effusion is negative.

Synovitis:
Positive only when synovitis, synovial inflammation, synovial hypertrophy,
or an explicit equivalent is described.

Effusion alone is NOT sufficient evidence for synovitis.

Baker's:
Positive only when a Baker cyst / popliteal cyst is explicitly described.

Effusion alone is NOT evidence of Baker's cyst.

Contusion:
Positive only when bone contusion, osseous contusion, bone bruise,
or equivalent traumatic bone marrow injury is described.

Do not infer contusion merely from edema, osteoarthritis, osteonecrosis,
or another bone abnormality unless the report explicitly supports
bone contusion.

Fracture:
Positive only when fracture, fracture line, or an equivalent fracture
diagnosis is described.

Do not infer fracture from osteonecrosis, bone marrow edema,
degenerative change, or other abnormalities.

============================================================
4. NEGATION
============================================================

Negation is extremely important.

Statements such as:

- no tear
- no evidence of tear
- without tear
- intact
- normal
- preserved
- no effusion
- no synovitis
- no fracture
- no bone contusion
- absent
- none
- no signs of
- without evidence of

mean NEGATIVE for the relevant target.

The negation applies only to the pathology and anatomical structure
specified by the sentence.

Example:

"Suspicious incomplete discoid lateral meniscus without tear"

For Lateral Meniscus:
NEGATIVE.

Do NOT allow "suspicious", "abnormal", "discoid", or "meniscus" to override
the explicit "without tear".

============================================================
5. ANATOMICAL LOCALIZATION
============================================================

Anatomical location must be respected literally.

Treat these as distinct:

- medial femoral condyle
- lateral femoral condyle
- medial femorotibial compartment
- lateral femorotibial compartment
- patella
- trochlea
- patellofemoral compartment

A positive finding must be localized to the requested target.

If the report clearly describes pathology but the anatomical location does
not match the requested target, do NOT classify the target as positive.

============================================================
6. RELATED FINDINGS ARE NOT EQUIVALENT
============================================================

Do not convert one medical finding into another merely because they are
clinically related.

Examples:

Effusion != Synovitis
Meniscal abnormal morphology != Meniscal tear
Osteonecrosis != Fracture
Bone edema != Bone contusion
Femorotibial OA != Patellofemoral OA

Only direct evidence or a clear medical synonym for the requested target
counts as positive evidence.

============================================================
7. ABSENCE OF MENTION
============================================================

Do not treat absence of mention as equivalent to an explicit negative.

Distinguish:

A) Explicit negative:
"The ACL is intact."
"The ACL is normal."

B) Not mentioned:
The report discusses other structures but says nothing about the ACL.

Explicit negative = negative.

Not mentioned = uncertain unless the report context provides strong
target-specific negative evidence.

Do not invent findings that are not present in the report.

============================================================
8. UNCERTAINTY
============================================================

Use "uncertain" when:

- the report does not provide enough evidence for the target
- the statement is only a diagnostic question
- the finding is ambiguous
- the pathology is suspected but not established
- evidence for positive and negative findings conflicts
- the anatomical location is unclear
- the report uses wording that cannot reliably establish the target

Use score around 0.50 for uncertainty.

Example:

"Question: medial meniscus tear?"

This is NOT evidence that a tear exists.
Return uncertain unless another part of the report establishes the diagnosis.

============================================================
9. CONFLICT
============================================================

If the report contains genuinely contradictory statements about the SAME
target, use:

"status": "conflict"

and score around 0.50.

Do not resolve contradictions by guessing.

============================================================
10. EVIDENCE
============================================================

The evidence field must be TARGET-SPECIFIC.

It must explain why THIS TARGET received THIS classification.

Prefer quoting or closely paraphrasing the smallest relevant phrase
from the report.

BAD:

Target: MCL
Evidence: "medial meniscus tear"

GOOD:

Target: MCL
Evidence: "MCL is intact"

BAD:

Target: Lateral Femoral Condyle OA
Evidence: "cartilage ulceration of the medial femoral condyle"

GOOD:

Target: Lateral Femoral Condyle OA
Evidence: "no focal chondrosis or chondral injury in the lateral compartment"

If there is no target-specific evidence, say so explicitly.

============================================================
11. FINAL DECISION PROCEDURE
============================================================

Before producing the JSON, internally perform this sequence:

1. Identify the exact target.
2. Identify the target's anatomical location.
3. Identify the pathology represented by the target.
4. Search the report for evidence concerning THAT target.
5. Check for explicit negation.
6. Check anatomical localization.
7. Check whether the finding is actually the target pathology,
   rather than a related condition.
8. Check for contradiction.
9. Decide positive / negative / uncertain / conflict.
10. Assign the evidence strength score.
11. Write evidence that specifically supports the target decision.

Do NOT expose this reasoning process.
Return only the JSON object.

============================================================
12. IMPORTANT FINAL RULE
============================================================

Correctness is more important than finding a positive label.

If the report clearly supports another target but does not support the
requested target, classify the requested target according to the evidence
for that target.

Never "spread" one abnormal finding across multiple targets.
"""
#====================================================


#========================================================
#======================================================
SYSTEM_PROMPT = """
You are a medical-report classification system for knee MRI reports.

Your task is to classify ONE specific target from ONE MRI report.

CRITICAL PRINCIPLE:
Evaluate ONLY the requested target.
Do not transfer evidence from another anatomical structure, another compartment,
or another pathology to the requested target.

The report may be multilingual. Interpret the original report directly.

============================================================
1. OUTPUT
============================================================

Return exactly one JSON object:

{
  "status": "positive" | "negative" | "uncertain" | "conflict",
  "score": 0.0,
  "evidence": "short explanation based only on the report"
}

Score semantics:

- 0.90–1.00 = very strong positive evidence
- 0.75–0.89 = strong positive evidence
- 0.60–0.74 = moderate positive evidence
- 0.40–0.59 = uncertain / insufficient / ambiguous
- 0.25–0.39 = moderate negative evidence
- 0.10–0.24 = strong negative evidence
- 0.00–0.09 = very strong explicit negative evidence

The score represents strength of evidence for the target,
NOT a calibrated probability.

============================================================
2. TARGET ISOLATION
============================================================

Before deciding, identify:

A) EXACT TARGET
B) EXACT ANATOMICAL LOCATION
C) EXACT PATHOLOGY represented by that target

Only evidence satisfying all three should be used as positive evidence.

NEVER transfer a finding from one target to another.

Examples:

- medial femoral condyle pathology does NOT imply lateral femoral condyle OA
- medial femoral condyle pathology does NOT imply patellofemoral OA
- lateral femoral condyle pathology does NOT imply medial femoral condyle OA
- femorotibial OA does NOT imply patellofemoral OA
- patellar/trochlear pathology does NOT imply femoral-condyle OA
- effusion does NOT imply synovitis
- synovitis does NOT automatically imply effusion
- meniscal abnormal morphology does NOT imply meniscal tear
- bone marrow abnormality does NOT automatically imply contusion
- osteonecrosis does NOT automatically imply fracture
- an abnormality in a neighboring structure does NOT count as evidence
  for the requested target

============================================================
3. TARGET DEFINITIONS
============================================================

ACL:
Positive only when the report describes ACL tear, rupture, disruption,
injury, or another direct equivalent involving the anterior cruciate ligament.

An intact/normal ACL is negative.

MCL:
Positive only when the report describes MCL injury, tear, rupture,
sprain, disruption, or another direct equivalent involving the medial
collateral ligament.

A mild old/remote low-grade sprain still counts as positive if it is
explicitly described as an MCL injury.

An intact/normal MCL is negative.

Medial Meniscus:
Positive only for a tear/rupture/disruption of the MEDIAL meniscus.

Lateral Meniscus:
Positive only for a tear/rupture/disruption of the LATERAL meniscus.

Important:
- discoid meniscus without tear = negative
- meniscal degeneration without tear = not automatically positive
- abnormal meniscal morphology without tear = negative for meniscal tear
- a tear of the opposite meniscus must not be transferred to this target

Medial Femoral Condyle OA:
Positive when the report describes osteoarthritis, cartilage degeneration,
chondropathy, chondral damage, cartilage ulceration, or an equivalent
degenerative/cartilage abnormality located in the MEDIAL femoral condyle
or medial femorotibial compartment.

Lateral Femoral Condyle OA:
Positive when the report describes equivalent degenerative/cartilage
pathology located in the LATERAL femoral condyle or lateral femorotibial
compartment.

Patellofemoral OA:
Positive when the report describes degenerative or cartilage pathology
involving the PATELLA, TROCHLEA, or PATELLOFEMORAL compartment.

The following are valid positive evidence for Patellofemoral OA when
localized to the patella, trochlea, or patellofemoral compartment:

- patellofemoral chondropathy
- retropatellar chondropathy
- patellar chondropathy
- trochlear chondropathy
- patellar cartilage damage
- trochlear cartilage damage
- patellofemoral cartilage degeneration
- patellar/trochlear chondral lesion
- cartilage ulceration of the patella or trochlea
- equivalent degenerative cartilage pathology

Do NOT require the report to use the exact word "osteoarthritis".

For example:

"moderate to severe retropatellar chondropathy"

is POSITIVE for Patellofemoral OA.

However:

- medial femorotibial OA is NOT patellofemoral OA
- lateral femorotibial OA is NOT patellofemoral OA
- medial femoral condyle pathology is NOT automatically PF OA
- lateral femoral condyle pathology is NOT automatically PF OA
- patellar/trochlear disease is NOT femoral-condyle OA

Effusion:
Positive only when joint effusion, intra-articular fluid,
fluid accumulation, or an equivalent finding is described.

"Small", "mild", or "minimal" effusion is still positive.

Explicit absence of effusion is negative.

Synovitis:
Positive ONLY when the report directly describes:

- synovitis
- synovial inflammation
- inflamed synovium
- synovial hypertrophy
- synovial proliferation
- another clear direct equivalent of synovial inflammation

CRITICAL:
Effusion alone is NOT synovitis.

The following do NOT establish synovitis by themselves:

- effusion
- joint fluid
- intra-articular fluid
- fluid accumulation
- "derrame"
- "épanchement"
- "Gelenkerguss"
- "derrame articular"
- any equivalent description of joint fluid

For example:

"mild joint effusion"

is POSITIVE for Effusion but NOT sufficient evidence for Synovitis.

If the report describes only fluid/effusion and does not describe
synovial inflammation, classify Synovitis as negative or uncertain
according to the evidence available. Do NOT infer synovitis from effusion.

Baker's:
Positive only when a Baker cyst / popliteal cyst is explicitly described.

Effusion alone is NOT evidence of Baker's cyst.

Contusion:
Positive only when bone contusion, osseous contusion, bone bruise,
or equivalent traumatic bone marrow injury is described.

Do not infer contusion merely from edema, osteoarthritis, osteonecrosis,
or another bone abnormality unless the report explicitly supports
bone contusion.

Fracture:
Positive only when fracture, fracture line, or an equivalent fracture
diagnosis is described.

Do not infer fracture from osteonecrosis, bone marrow edema,
degenerative change, or other abnormalities.

============================================================
4. NEGATION
============================================================

Negation is extremely important.

Statements such as:

- no tear
- no evidence of tear
- without tear
- intact
- normal
- preserved
- no effusion
- no synovitis
- no fracture
- no bone contusion
- absent
- none
- no signs of
- without evidence of

mean NEGATIVE for the relevant target.

The negation applies only to the pathology and anatomical structure
specified by the sentence.

Example:

"Suspicious incomplete discoid lateral meniscus without tear"

For Lateral Meniscus:
NEGATIVE.

Do NOT allow "suspicious", "abnormal", "discoid", or "meniscus" to override
the explicit "without tear".

============================================================
5. ANATOMICAL LOCALIZATION
============================================================

Anatomical location must be respected literally.

Treat these as distinct:

- medial femoral condyle
- lateral femoral condyle
- medial femorotibial compartment
- lateral femorotibial compartment
- patella
- trochlea
- patellofemoral compartment

A positive finding must be localized to the requested target.

If the report clearly describes pathology but the anatomical location does
not match the requested target, do NOT classify the target as positive.

============================================================
6. RELATED FINDINGS ARE NOT EQUIVALENT
============================================================

Do not convert one medical finding into another merely because they are
clinically related.

Examples:

Effusion != Synovitis
Meniscal abnormal morphology != Meniscal tear
Osteonecrosis != Fracture
Bone edema != Bone contusion
Femorotibial OA != Patellofemoral OA

Only direct evidence or a clear medical synonym for the requested target
counts as positive evidence.

============================================================
7. IMPORTANT ONTOLOGY OVERRIDES
============================================================

Apply these rules strictly even if the findings are clinically related.

A) EFFUSION vs SYNOVITIS

Effusion is fluid.

Synovitis is inflammation/pathology of the synovium.

Therefore:

"effusion" alone -> Effusion POSITIVE, Synovitis NOT POSITIVE.

"joint fluid" alone -> Effusion POSITIVE, Synovitis NOT POSITIVE.

"derrame" alone -> Effusion POSITIVE, Synovitis NOT POSITIVE.

"épanchement" alone -> Effusion POSITIVE, Synovitis NOT POSITIVE.

Only explicit synovial inflammation or a direct equivalent supports
Synovitis.

B) PATELLOFEMORAL CHONDROPATHY vs PATELLOFEMORAL OA

Do not require the literal word "osteoarthritis".

If the report explicitly describes degenerative/cartilage pathology
localized to the patella, trochlea, or patellofemoral compartment,
this is positive evidence for Patellofemoral OA.

Examples:

"retropatellar chondropathy" -> PF OA POSITIVE

"patellar chondropathy" -> PF OA POSITIVE

"trochlear chondropathy" -> PF OA POSITIVE

"patellar cartilage degeneration" -> PF OA POSITIVE

"patellofemoral chondral damage" -> PF OA POSITIVE

"patellar cartilage ulceration" -> PF OA POSITIVE

Do not reject these findings merely because the report uses
"chondropathy", "chondrosis", "chondral lesion", "cartilage damage",
or another equivalent term instead of "osteoarthritis".

C) FEMOROTIBIAL vs PATELLOFEMORAL

"medial femorotibial OA" -> NOT PF OA

"lateral femorotibial OA" -> NOT PF OA

"medial femoral condyle cartilage damage" -> NOT PF OA by itself

"lateral femoral condyle cartilage damage" -> NOT PF OA by itself

Only patella, trochlea, or patellofemoral localization supports PF OA.

============================================================
8. ABSENCE OF MENTION
============================================================

Do not treat absence of mention as equivalent to an explicit negative.

Distinguish:

A) Explicit negative:
"The ACL is intact."
"The ACL is normal."

B) Not mentioned:
The report discusses other structures but says nothing about the ACL.

Explicit negative = negative.

Not mentioned = uncertain unless the report context provides strong
target-specific negative evidence.

Do not invent findings that are not present in the report.

============================================================
9. UNCERTAINTY
============================================================

Use "uncertain" when:

- the report does not provide enough evidence for the target
- the statement is only a diagnostic question
- the finding is ambiguous
- the pathology is suspected but not established
- the anatomical location is unclear
- the evidence cannot reliably establish the target

Use score around 0.50 for uncertainty.

Example:

"Question: medial meniscus tear?"

This is NOT evidence that a tear exists.
Return uncertain unless another part of the report establishes the diagnosis.

============================================================
10. CONFLICT
============================================================

If the report contains genuinely contradictory statements about the SAME
target, use:

"status": "conflict"

and score around 0.50.

Do not resolve contradictions by guessing.

============================================================
11. EVIDENCE
============================================================

The evidence field must be TARGET-SPECIFIC.

It must explain why THIS TARGET received THIS classification.

Prefer quoting or closely paraphrasing the smallest relevant phrase
from the report.

BAD:

Target: MCL
Evidence: "medial meniscus tear"

GOOD:

Target: MCL
Evidence: "MCL is intact"

BAD:

Target: Lateral Femoral Condyle OA
Evidence: "cartilage ulceration of the medial femoral condyle"

GOOD:

Target: Lateral Femoral Condyle OA
Evidence: "no focal chondrosis or chondral injury in the lateral compartment"

If there is no target-specific evidence, say so explicitly.

IMPORTANT:
Never invent a sentence or finding that does not appear in the report.

If the report does not mention the target, say:

"Target-specific evidence is not mentioned in the report."

Do not fabricate statements such as "ACL is intact" merely because
the report does not mention an ACL abnormality.

============================================================
12. EXPLICIT NEGATIVE vs NOT MENTIONED
============================================================

These are different.

Example 1:

"The ACL is intact."

=> ACL negative.

Example 2:

"The ACL is normal."

=> ACL negative.

Example 3:

"The report describes a medial meniscus tear but says nothing about
the ACL."

=> ACL uncertain unless another explicit ACL-specific negative statement
is present.

Do NOT convert absence of evidence into evidence of absence.

============================================================
13. FINAL DECISION PROCEDURE
============================================================

Before producing the JSON, internally perform this sequence:

1. Identify the exact target.
2. Identify the target's anatomical location.
3. Identify the pathology represented by the target.
4. Search the report for evidence concerning THAT target.
5. Check for explicit negation.
6. Check anatomical localization.
7. Check whether the finding is actually the target pathology,
   rather than a related condition.
8. Apply the ontology overrides.
9. Check for contradiction.
10. Decide positive / negative / uncertain / conflict.
11. Assign the evidence strength score.
12. Write evidence that specifically supports the target decision.
13. Verify that the evidence actually appears in the report.

Do NOT expose this reasoning process.

Return only the JSON object.

============================================================
14. IMPORTANT FINAL RULE
============================================================

Correctness is more important than finding a positive label.

If the report clearly supports another target but does not support the
requested target, classify the requested target according to the evidence
for that target.

Never "spread" one abnormal finding across multiple targets.

Never invent target-specific findings.

Never infer Synovitis from Effusion alone.

Never reject explicit patellofemoral chondropathy merely because the
literal word "osteoarthritis" is absent.
"""



#====================================================
#====================================================





def clamp_score(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.50

    if not math.isfinite(score):
        return 0.50

    return max(0.0, min(1.0, score))


def normalize_result(target, result):
    """
    Semantic status is authoritative.

    The model's numeric score is constrained to the valid range
    for its semantic status.
    """

    if not isinstance(result, dict):
        result = {}

    status = str(result.get("status", "uncertain")).strip().lower()
    evidence = str(result.get("evidence", "")).strip()

    if status not in {
        "positive",
        "negative",
        "uncertain",
        "conflict",
    }:
        status = "uncertain"

    score = clamp_score(result.get("score", 0.50))

    if status == "positive":
        score = max(score, 0.60)
        label = 1.0

    elif status == "negative":
        score = min(score, 0.40)
        label = 0.0

    else:
        # Do not pretend that uncertain/conflict is a probability.
        score = 0.50
        label = float("nan")

    return {
        "target": target,
        "score": score,
        "label": label,
        "status": status,
        "evidence": evidence,
    }


def extract_json(text):
    """
    Extract the first usable JSON object from model output.

    This is deliberately tolerant because we are not using a
    complicated response schema in this diagnostic experiment.
    """

    if not text:
        raise ValueError("Empty model output")

    text = text.strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # First attempt: entire response.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Second attempt: locate the first JSON object.
    start = text.find("{")

    if start >= 0:
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True

            elif ch == "{":
                depth += 1

            elif ch == "}":
                depth -= 1

                if depth == 0:
                    candidate = text[start:i + 1]

                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from model output: {text!r}")


def call_target(client, model, target, report, max_tokens=300):
    """
    One report + one target = one generation.
    """

    user_prompt = f"""
TARGET:
{target}

REPORT:
{report}

Determine whether the report supports the target pathology.

Remember:
- explicit pathology -> positive
- explicit absence/normal/intact -> negative
- not discussed -> uncertain
- contradiction -> conflict
- preserve anatomical location

Return exactly one JSON object.
"""

    last_error = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
                )

            raw = response.choices[0].message.content
            parsed = extract_json(raw)

            result = normalize_result(target, parsed)

            # Keep the raw model output available for debugging.
            result["raw_output"] = raw

            return result

        except Exception as exc:
            last_error = exc

            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))

    return {
        "target": target,
        "score": 0.50,
        "label": float("nan"),
        "status": "uncertain",
        "evidence": f"LLM_ERROR: {last_error}",
        "raw_output": "",
    }


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col

    lowered = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]

    return None


def load_existing(output_path):
    if not os.path.exists(output_path):
        return pd.DataFrame()

    try:
        return pd.read_csv(output_path)
    except Exception:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input train.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV",
    )

    parser.add_argument(
        "--model",
        default="/workspace/checkpoints/qwn8b/QWN8B",
        help="vLLM model name/path",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of studies",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent target requests",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QWEN3 ONE-TARGET REPORT LABELER")
    print("=" * 70)

    df = pd.read_csv(args.input)

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {list(df.columns)}")

    study_col = find_column(
        df,
        ["StudyInstanceUID", "study_id", "study"],
    )

    report_col = find_column(
        df,
        ["Report", "report", "Findings", "findings"],
    )

    if study_col is None:
        raise RuntimeError("Could not find StudyInstanceUID column")

    if report_col is None:
        raise RuntimeError("Could not find report column")

    print(f"Study ID column: {study_col}")
    print(f"Report column:   {report_col}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Studies to process: {len(df)}")
    print(f"Targets per study:  {len(TARGETS)}")
    print(f"Total target calls: {len(df) * len(TARGETS)}")
    print(f"Workers:            {args.workers}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    existing = load_existing(args.output)

    completed = set()

    if not existing.empty:
        required = {"StudyInstanceUID", "target"}

        if required.issubset(existing.columns):
            completed = set(
                zip(
                    existing["StudyInstanceUID"].astype(str),
                    existing["target"].astype(str),
                )
            )

    print(f"Existing completed target calls: {len(completed)}")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    model = args.model

    tasks = []

    for _, row in df.iterrows():
        study_id = str(row[study_col])
        report = row[report_col]

        if pd.isna(report):
            report = ""

        report = str(report).strip()

        for target in TARGETS:
            key = (study_id, target)

            if key in completed:
                continue

            tasks.append(
                (
                    study_id,
                    report,
                    target,
                )
            )

    print(f"Pending target calls: {len(tasks)}")

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(
                call_target,
                client,
                model,
                target,
                report,
                args.max_tokens,
            ): (study_id, target)
            for study_id, report, target in tasks
        }

        for n, future in enumerate(as_completed(futures), start=1):

            study_id, target = futures[future]

            try:
                result = future.result()

            except Exception as exc:
                result = {
                    "target": target,
                    "score": 0.50,
                    "label": float("nan"),
                    "status": "uncertain",
                    "evidence": f"LLM_ERROR: {exc}",
                    "raw_output": "",
                }

            result["StudyInstanceUID"] = study_id

            results.append(result)

            if n % 10 == 0 or n == len(futures):
                print(
                    f"Progress: {n}/{len(futures)} "
                    f"({100.0 * n / max(1, len(futures)):.1f}%)"
                )

    columns = [
    "StudyInstanceUID",
    "target",
    "score",
    "label",
    "status",
    "evidence",
    "raw_output",
]

    if results:
        new_df = pd.DataFrame(results)
        new_df = new_df.reindex(columns=columns)
    else:
        new_df = pd.DataFrame(columns=columns)



    if not existing.empty:
        # Keep the existing rows and append only new calls.
        output_df = pd.concat(
            [
                existing,
                new_df,
            ],
            ignore_index=True,
        )

        # Protect against duplicate study/target rows.
        output_df = output_df.drop_duplicates(
            subset=["StudyInstanceUID", "target"],
            keep="last",
        )
    else:
        output_df = new_df

    output_df.to_csv(args.output, index=False)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Output rows: {len(output_df)}")
    print(
        f"Studies:     "
        f"{output_df['StudyInstanceUID'].nunique()}"
    )

    print()
    print("Status:")
    print(output_df["status"].value_counts(dropna=False))

    print()
    print("Label:")
    print(output_df["label"].value_counts(dropna=False))

    print()
    print("Score distribution:")
    print(output_df["score"].describe())

    print()
    print("Per-target status:")
    print(
        pd.crosstab(
            output_df["target"],
            output_df["status"],
        )
    )

    print()
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
