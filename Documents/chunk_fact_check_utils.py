"""
chunk_fact_check_utils.py
===================
Helper functions for the fact-verification pipeline.
Chunk-level using NER pre-filter.

"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from rapidfuzz import fuzz
from Levenshtein import ratio, setratio, seqratio, jaro, jaro_winkler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# NLI label-order maps  (id2label of each model, inverted to name -> index)
# ---------------------------------------------------------------------------
# cross-encoder/nli-deberta-v3-base       : {0: contradiction, 1: entailment, 2: neutral}
NLI_DEBERTA = {"contradiction": 0, "entailment": 1, "neutral": 2}
# MoritzLaurer/DeBERTa-v3-large-mnli-...   : {0: entailment, 1: neutral, 2: contradiction}
MORITZLAURER = {"entailment": 0, "neutral": 1, "contradiction": 2}


# ===========================================================================
# 1. Data loading / preprocessing
# ===========================================================================
def sent_sent_reading(text_file_path):
    """Read a raw text file and return it split into stripped, non-empty lines."""
    all_sents = []
    with open(text_file_path, "r", encoding="utf-8") as new_text:
        for sentences in new_text.readlines():
            sents = sentences.strip()
            if sents:
                all_sents.append(sents)
    return all_sents


# --- chunk-based preprocessing  --------------
# These regexes strip inline reference markers / leading numbers / bullets
# before sentence splitting.
_INLINE_REF_DOT = re.compile(r"(?<=[A-Za-z\)\]\"'])\.(\d{1,3})(?=[\s,;:.\)\]]|$)")
_INLINE_REF_YEAR_DOT = re.compile(r"\b((?:19|20)\d{2})\.(\d{1,3})(?=[\s,;:.\)\]]|$)")
_INLINE_REF_YEAR_NODOT = re.compile(r"\b((?:19|20)\d{2})(\d{1,3})(?=\s|$)")
_LEADING_NUM = re.compile(r"^\s*\d{1,4}\.?\s*$|^\s*\d{1,4}\.?\s+")
_BULLET = re.compile(r"\s*[;:,]?\s*·\s*")


def _clean_line(line):
    """Strip inline reference markers, leading numbers, and bullets from a line."""
    line = _INLINE_REF_DOT.sub(".", line)
    line = _INLINE_REF_YEAR_DOT.sub(r"\1.", line)
    line = _INLINE_REF_YEAR_NODOT.sub(r"\1 ", line)
    line = _LEADING_NUM.sub("", line)
    return line


def sent_chunk_reading(text_file_path, nlp, count_tokens,
                       target_tokens=150, max_tokens=250, overlap_tokens=30):
    """Token-budget chunking (the chunk-based counterpart to sent_sent_reading).

    Groups sentences until target_tokens and adds overlap by carrying the tail
    of the previous chunk into the next. `nlp` is a spaCy model used only for
    sentence splitting (e.g. en_core_web_sm) and `count_tokens` is based on the 
    embedding model's tokenizer).

    Currently, only `target_tokens` is effecting the results and `max_tokens` is added as
    a future option to work with.
    """
    with open(text_file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    cleaned = [_clean_line(l) for l in lines]
    text = " ".join(cleaned)
    text = _BULLET.sub(". ", text)
    text = re.sub(r"\s+", " ", text).strip()

    sents = [(s.text.strip(), count_tokens(s.text.strip()))
             for s in nlp(text).sents if s.text.strip()]

    chunks = []
    current_sents, current_tokens = [], 0
    overlap_buffer = []  # carry tail of last chunk into next

    for sent, n_tok in sents:
        # Flush if adding this sentence would exceed the budget
        if current_tokens + n_tok > target_tokens and current_sents:
            chunk_text = " ".join(s for s, _ in current_sents)
            chunks.append(chunk_text)
            # Build overlap: take last sentences worth ~overlap_tokens
            overlap_buffer, ot = [], 0
            for s, t in reversed(current_sents):
                if ot + t > overlap_tokens:
                    break
                overlap_buffer.insert(0, (s, t))
                ot += t
            current_sents = list(overlap_buffer)
            current_tokens = sum(t for _, t in current_sents)

        current_sents.append((sent, n_tok))
        current_tokens += n_tok

    if current_sents:
        chunks.append(" ".join(s for s, _ in current_sents))

    return chunks


def read_json(file_path):
    """Read a JSON file and return the parsed object."""
    with open(file_path, "r", encoding="utf-8") as json_read:
        full_json = json.load(json_read)
    return full_json


def json_feature(full_json):
    """Pull required information from the full json file. 
    Returns a list of dictionaries containing document_name, fact, labels,
    and a derived text_name.
    """
    info_results_json = []
    for projects in full_json["projects"]:
        for info in projects["facts"]:
            document_name = info["document_name"]
            fact = info["verification_data"]["fact"]["text"]
            machine_verification_status = info["machine_verification_status"]
            human_verification_status = info["human_verification_status"]

            info_results_json.append({
                "document_name": document_name,
                "fact": fact,
                "machine_verification_status": machine_verification_status,
                "human_verification_status": human_verification_status,
                "text_name": document_name.replace(".docx", ".txt"),
            })

    return info_results_json


def build_all_facts(info_results_json):
    """Flat list of every fact string, in annotation order."""
    return [info["fact"] for info in info_results_json]


def build_gold_labels(info_results_json):
    """Gold labels in annotation order: 1 = verified, 0 = contradicted.

    This is the single source of truth for gold labels. Every method used to
    rebuild its own identical list (gold_label_lev, gold_label_rf, ...); now
    they all share this one.
    """
    return [1 if info["human_verification_status"] == "verified" else 0
            for info in info_results_json]


# ===========================================================================
# 2. NER preprocessing
# ===========================================================================
def extract_ner_batch(sentences, nlp):
    """Extract named entities from a list of sentences using the given spaCy model.

    Returns a list of (sentence, {(entity_text_lower, label), ...}) tuples.
    """
    docs = list(nlp.pipe(sentences))
    ner_results = []
    for sent, doc in zip(sentences, docs):
        ent_info = set()
        for ent in doc.ents:
            clean_text = "".join(
                token.text_with_ws for token in ent if not token.is_space
            ).strip()
            if clean_text:
                ent_info.add((clean_text.lower(), ent.label_))
        ner_results.append((sent, ent_info))
    return ner_results


def ner_per_file(info_results_json, file_path, nlp):
    """Read + NER-tag every source text file referenced in the annotations.

    Returns {absolute_path: [(sentence, entity_set), ...]}.
    """
    file_ner = {}
    for info in info_results_json:
        actual_path = os.path.join(file_path, info["text_name"])
        if actual_path not in file_ner and os.path.exists(actual_path):
            sentences = sent_sent_reading(actual_path)
            file_ner[actual_path] = extract_ner_batch(sentences, nlp)
            print(f"  Cached: {actual_path} ({len(sentences)} sentences)")
    print(f"Total files cached: {len(file_ner)}")
    return file_ner


def filter_ner_sentences(per_file_ner):
    """
    Keep only sentences that contain at least one named entity.

    Returns:
        dict: {path: [sentence, ...]}
    """
    filtered_sentences = {}
    for path, sents_with_ner in per_file_ner.items():
        sentences = []
        for sent, ents in sents_with_ner:
            if ents:
                sentences.append(sent)
        filtered_sentences[path] = sentences

    return filtered_sentences

# ===========================================================================
# 3. Lexical / fuzzy similarity
# ===========================================================================
def exact_match_score(fact, sentences):
    """Substring exact-match baseline: 1 if the fact appears verbatim
     inside any candidate chunk, else 0.
    """
    for sentence in sentences:
        if fact.strip().lower() in sentence.strip().lower():
            return 1
    return 0


def get_best_match_ner_lev(facts_text, sentences_with_ner, method, threshold):
    """First NER-filtered sentence whose Levenshtein-family score clears the
    threshold. Returns (label, score, sentence); (0, None, None) if none clear.
    """
    lev_method_type = {
        "og_ratio": ratio,
        "set_ratio": setratio,
        "seq_ratio": seqratio,
        "jaro": jaro,
        "jaro_winkler": jaro_winkler,
    }
    score_method = lev_method_type.get(method)
    for sentence in sentences_with_ner:
        score = score_method(sentence, facts_text)
        if score >= threshold:
            return 1, score, sentence
    return 0, None, None


def get_best_match_ner_rapfuz(facts_text, sentences_with_ner, method, threshold):
    """First NER-filtered sentence whose RapidFuzz score clears the threshold.
    Returns (label, score, sentence); (0, None, None) if none clear.
    """
    rapfuz_method_type = {
        "token_sort": fuzz.token_sort_ratio,
        "token_set": fuzz.token_set_ratio,
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_ratio": fuzz.partial_token_ratio,
        "w_ratio": fuzz.WRatio,
    }
    score_method = rapfuz_method_type.get(method)
    for sentence in sentences_with_ner:
        score = score_method(sentence, facts_text)
        if score >= threshold:
            return 1, score, sentence
    return 0, None, None


# ===========================================================================
# 4. Embedding similarity approache and Paraphrase detection approach
# ---------------------------------------------------------------------------
#   embedding similarity   -> all-MiniLM-L6-v2         (get_first_above_thresh)
#   paraphrase detection   -> paraphrase-mpnet-base-v2 (get_first_above_thresh_paraph)
# They share only the generic encoding helper (build_embedding_cache). The matching
# functions are separate so each approach can evolve on its own.
# ===========================================================================
def build_embedding_cache(per_file_ner_filtered, model, show_progress_bar=False):
    """Encode every file's NER-filtered sentences and cache them by path.

    Shared by both embedding and paraphrase approaches; the specific model is passed in.
    """
    cache = {}
    for path, sentences in per_file_ner_filtered.items():
        if sentences:
            cache[path] = model.encode(sentences, show_progress_bar=show_progress_bar)
            print(f"  Embedded: {os.path.basename(path)} ({len(sentences)} sentences)")
        else:
            print(f"NO SENTENCES: {os.path.basename(path)}")
    return cache


# --- embedding similarity approach (all-MiniLM-L6-v2) --------------------
def get_first_above_thresh(facts_txt, sentences, sentence_embedding, model, threshold):
    """Embedding similarity approach (all-MiniLM-L6-v2).

    First sentence whose cosine similarity to the fact clears the threshold.
    Returns (score, sentence, found); if nothing clears, returns the best
    match with found=False.
    """
    fact_embedding = model.encode([facts_txt])
    cos_scores = cosine_similarity(fact_embedding, sentence_embedding)[0]

    for i, score in enumerate(cos_scores):
        if score >= threshold:
            return (float(score), sentences[i], True)

    best = cos_scores.argmax()
    return (float(cos_scores[best]), sentences[best], False)


# --- paraphrase detection approach (paraphrase-mpnet-base-v2) --------------------
def get_first_above_thresh_paraph(facts_txt, sentences, sentence_embedding,
                                  model_paraph, threshold):
    """Paraphrase detection approach (paraphrase-mpnet-base-v2).

    Same matching rule as get_first_above_thresh, kept as its own function so
    the paraphrase idea stays clearly separated from the sentence-embedding
    idea in both the script and the notebook.
    """
    fact_embedding = model_paraph.encode([facts_txt])
    cos_scores = cosine_similarity(fact_embedding, sentence_embedding)[0]

    for i, score in enumerate(cos_scores):
        if score >= threshold:
            return (float(score), sentences[i], True)

    best = cos_scores.argmax()
    return (float(cos_scores[best]), sentences[best], False)


# ===========================================================================
# 5. NLI cross-encoder
# ===========================================================================
def get_all_pairs_cross_encoder(facts_txt, sentences_with_ner, model,
                                batch_size=64, apply_softmax=True):
    """Score every (chunk, fact) pair with the given CrossEncoder.

    Returns (scores, sentences_with_ner). `model` is passed in so the same
    function can build both the nli-deberta and the MoritzLaurer checkpoints.

    `apply_softmax` controls what gets stored:
      * False  -> raw logits 
      * True   -> probabilities 
    """
    pairs = [[sent, facts_txt] for sent in sentences_with_ner]
    scores = model.predict(pairs, batch_size=batch_size, apply_softmax=apply_softmax)
    return scores, sentences_with_ner


def early_exit(entry, threshold, entail_index):
    """Early-exit search: index of the first sentence whose entailment
    probability clears the threshold.

    Returns (idx, prob) or (None, None). `entail_index` selects the entailment
    column for the model in use (NLI_DEBERTA['entailment'] vs
    MORITZLAURER['entailment']).
    """
    for i, s in enumerate(entry["scores"]):
        p = float(s[entail_index])
        if p >= threshold:
            return i, p
    return None, None


def predict_early_exit(entry, threshold, entail_index):
    """Predict 1 (verified) if any sentence's entailment prob clears the
    threshold, else 0.
    """
    idx, prob = early_exit(entry, threshold, entail_index)
    return 1 if idx is not None else 0


def early_exit_decision(entry, threshold, entail_index):
    """Walk chunks in order; verified the moment one crosses the entailment
    threshold. Returns (pred, chosen_chunk, chosen_score).

    For pred=1, chosen chunk is the FIRST one that crossed.
    For pred=0, no chunk crossed -> report the highest-entailment chunk as the
    closest near-miss (via np.argmax).
    """
    scores = entry["scores"]
    for idx, s in enumerate(scores):
        if s[entail_index] >= threshold:
            return 1, entry["sentences"][idx], s

    idx = int(np.argmax([s[entail_index] for s in scores]))
    return 0, entry["sentences"][idx], scores[idx]


def predict_asymmetric(entry, threshold, entail_index, contra_index):
    """Asymmetric rule: contradicted (0) only when the strongest contradiction
    clears the threshold AND the strongest entailment does not; otherwise 1.
    """
    scores = entry["scores"]
    max_entail = max(s[entail_index] for s in scores)
    max_contra = max(s[contra_index] for s in scores)
    if max_contra >= threshold and max_entail < threshold:
        return 0
    return 1


# ===========================================================================
# 6. Evaluation / reporting
# ===========================================================================
def report_at_threshold(gold_labels, predictions, threshold, digits=3,
                        target_names=("contradicted", "verified"),
                        display_labels=("Contradicted", "Verified"),
                        figsize=(3.5, 3.5), show_plot=True):
    """Print the classification report and (optionally) show the confusion
    matrix for one threshold. Replaces the copy-pasted report+plot block that
    appeared in every evaluation cell.
    """
    print(f"\nClassification Report for Threshold {threshold}")
    print(classification_report(gold_labels, predictions, digits=digits,
                                target_names=list(target_names)))
    print("=" * 60)
    if show_plot:
        cm = confusion_matrix(gold_labels, predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=list(display_labels))
        fig, ax = plt.subplots(figsize=figsize)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"threshold = {threshold}")
        plt.show()
        print("=" * 60)
