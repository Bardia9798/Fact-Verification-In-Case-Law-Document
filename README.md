# Fact Verification In Case-Law Documents: An Industry Use
Repository created for presenting the work performed for the Master's thesis
# Overview
This repository belongs to the Master project "Fact Verification in Case-Law Documents” by Bardia Kashani, with the supervision of Dr. Pia Sommerauer and Dr. Ilia Markov. The project was carried out in collaboration with International Bureau of Fiscal Documentation (IBFD).

This thesis addresses the automated fact verification of atomic facts extracted from international case law summaries. The task is framed as a binary classification problem, where is fact is either Verified or Contradicted, to determine whether a given fact is supported or rejected by the source award document. The task examines the challenges of verifying highly formal, complex legal texts where contradicted facts are often minimally edited paraphrases of the source text, differing only by small details such as numeric values, dates, or negations.

The proposed pipeline evaluates and compares four lighter-weight scoring approaches against an LLM-based baseline developed by IBFD. The evaluated approaches include: (1) string-based comparison (Levenshtein and RapidFuzz); (2) dense semantic embedding similarity (all-MiniLM-L6-v2); (3) paraphrase detection (paraphrase-mpnet-base-v2); and (4) natural language inference (NLI) using DeBERTa-based cross-encoders. These approaches are evaluated at both the sentence-level and chunk-level to investigate the trade-off between computational cost, architectural complexity, and how well they perform on fact verification.

It must be mentioned that the data cannot be shared with third parties due to the confidentiality agreement with the company, so it is not published in this repository. Any outputs that would reveal its contents have been removed.
