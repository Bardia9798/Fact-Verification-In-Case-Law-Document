# Fact Verification In Case-Law Documents: An Industry Use
Repository created for presenting the work performed for the Master's thesis
# Overview
This repository belongs to the Master project "Fact Verification in Case-Law Documents” by Bardia Kashani, with the supervision of Dr. Pia Sommerauer and Dr. Ilia Markov. The project was carried out in collaboration with International Bureau of Fiscal Documentation (IBFD).

This thesis addresses the automated fact verification of atomic facts extracted from international case law summaries. The task is framed as a binary classification problem, where the fact is either Verified or Contradicted, to determine whether a given fact is supported or rejected by the source award document. The task examines the challenges of verifying highly formal, complex legal texts where contradicted facts are often minimally edited paraphrases of the source text, differing only by small details such as numeric values, dates, or negations.

The proposed pipeline evaluates and compares four lighter-weight scoring approaches against an LLM-based baseline developed by IBFD. The evaluated approaches include: (1) string-based comparison (Levenshtein and RapidFuzz); (2) dense semantic embedding similarity (all-MiniLM-L6-v2); (3) paraphrase detection (paraphrase-mpnet-base-v2); and (4) natural language inference (NLI) using DeBERTa-based cross-encoders. In addition, an optional step is used. Named Entity Recognition as a pre-filtering step is used to only keep the candidates that contain at least one named entity in them. These approaches are evaluated at both the sentence level and chunk level to investigate the trade-off between computational cost, architectural complexity, and how well they perform on fact verification.

It must be mentioned that the data cannot be shared with third parties due to the confidentiality agreement with the company, so it is not published in this repository. Any outputs that would reveal its contents have been removed.

# Project Structure

```
Fact Verification In Case-Law Documents
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── Documents
    ├── sentence-level-fact-verification.ipynb
    ├── sentence-level-fact-verification-no-ner.ipynb
    ├── chunk-level-fact-verification.ipynb
    ├── scoring-validation-hard-code.ipynb 
    ├── fact_check_utils.py
    ├── chunk_fact_check_utils.py 
    ├── Results
    │   ├── sentence-level
    │   └── chunk-level
    ├── Confusion Matrices
    │   ├── Sentence-level optimal confusion matrices
    │   ├── NLI_chunk confusion matrices
    │   └── sentence-level confusion matrices
    └── Figure
        └── five-category-error-analysis
```
# Notebooks
1. **sentence-level-fact-verification.ipynb**: This notebook illustrates how the lighter-weight fact verification was performed at sentence level. The sentences here go through NER pre-filtering, and only the sentences with at least one named entity are considered as a candidate for fact verification by each 4 scoring approaches.
2. **sentence-level-fact-verification-no-ner.ipynb**: This notebook illustrates how the lighter-weight fact verification was performed at sentence level. There is no NER pre-filtering step here, so all the sentences are considered as a candidate for fact verification by each 4 scoring approaches.
3. **chunk-level-fact-verification.ipynb**: This notebook illustrates how the lighter-weight fact verification was performed at chunk level. The candidates here are multi-sentences chunks and they go through NER pre-filtering step.
4. **scoring-validation-hard-code.ipynb**: This notebook is used to validate and verify the performance of each model. Here we can double check the scores by providing the fact and the sentence/chunk.

# Scripts
1. **fact_check_utils.py**: Contains the helper functions required for sentence level evaluation
2. **chunk_fact_check_utils.py**: Contains the helper function required for chunk level evaluation

# Results
The **Results** folder contains the classification reports for sentence level and chunk level evaluation. Chunk level are provided for NLI scoring approach, for both small and large models.

# Confusion Matrices
The **Confusion Matrices** folder contains the confusion matrices.

# Figure
The **Figure** folder contains the errors distribution.

# Requirements 
The **requirements.txt** contains the Python packages required to run the codes. Also, for running the spaCy NER filtering, you need to install the large CNN model and the transformer model:
```Bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```
# Thesis Report
The thesis report is fully available in PDF format, completed in Overleaf.
