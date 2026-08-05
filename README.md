# Fact Verification In Case-Law Documents: An Industry Use
Repository created for presenting the work performed for the Master's thesis
# Overview
This repository belongs to the Master project "Fact Verification in Case-Law Documents” by Bardia Kashani, with the supervision of Dr. Pia Sommerauer and Dr. Ilia Markov. The project was carried out in collaboration with International Bureau of Fiscal Documentation (IBFD).

This thesis addresses the automated fact verification of atomic facts extracted from international case law summaries. The task is framed as a binary classification problem, where is fact is either Verified or Contradicted, to determine whether a given fact is supported or rejected by the source award document. The task examines the challenges of verifying highly formal, complex legal texts where contradicted facts are often minimally edited paraphrases of the source text, differing only by small details such as numeric values, dates, or negations.

The proposed pipeline evaluates and compares four lighter-weight scoring approaches against an LLM-based baseline developed by IBFD. The evaluated approaches include: (1) string-based comparison (Levenshtein and RapidFuzz); (2) dense semantic embedding similarity (all-MiniLM-L6-v2); (3) paraphrase detection (paraphrase-mpnet-base-v2); and (4) natural language inference (NLI) using DeBERTa-based cross-encoders. In addition, an optional step is used. Named Entity Recognition as a pre-filtering step is used to only keep the candidates that contain at least one named entity in them. These approaches are evaluated at both the sentence level and chunk level to investigate the trade-off between computational cost, architectural complexity, and how well they perform on fact verification.

It must be mentioned that the data cannot be shared with third parties due to the confidentiality agreement with the company, so it is not published in this repository. Any outputs that would reveal its contents have been removed.

# Project Structure

```text
Fact Verification In Case-Law Documents
L───Documents
|   L───Figures
|   |   |   [example_confusion_matrix].png   # Contains confusion matrices displays
|   L───Results
|   |   |   [example_result_file].csv        # Contains output data and results
|   |   sentence-level-fact-verification.ipynb                     # [evaluates the four lighter-weight approaches on sentence level with NER pre-filter]
|   |   sentence-level-fact-verification-no-ner.ipynb              # [evaluates the four lighter-weight approaches on sentence level without NER pre-filter]
|   |   chunk-level-fact-verification.ipynb              # [evaluates the four lighter-weight approaches on chunk level with NER pre-filter]
|   |   scoring-validation-hard-code.ipynb              # [to manually validate and verify the score each scoring approach provides for each fact=candidate pair]
|   |   fact_check_utils.py                         # [helper functions for the sentence level notebooks]
|   |   chunk_fact_check_utils.py                   # [helper function for the chunk level notebook]
|   .gitignore
|   LICENSE
|   README.md
|   requirements.txt
