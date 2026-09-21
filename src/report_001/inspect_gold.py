import pandas as pd

path = "/workspace/data/reports/report_001_gold.csv"
gold = pd.read_csv(path)

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

print("Shape:", gold.shape)

print("\nPositive counts:")
print(gold[TARGETS].sum().sort_values(ascending=False))

print("\nNegative counts:")
print((gold[TARGETS] == 0).sum().sort_values(ascending=False))

gold["report_chars"] = gold["Report"].fillna("").str.len()
gold["report_words"] = gold["Report"].fillna("").str.split().str.len()

print("\nReport length:")
print(gold[["report_chars", "report_words"]].describe())
