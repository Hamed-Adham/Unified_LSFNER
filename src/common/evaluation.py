# conll_evaluator.py

import sys
import re
from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'

class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0
        self.correct_tags = 0
        self.found_correct = 0
        self.found_guessed = 0
        self.token_counter = 0
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False
    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True
    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False
    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True
    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True
    return chunk_start

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def get_metrics(counts):
    """Calculates overall and by-type metrics from EvalCounts."""
    overall = calculate_metrics(
        counts.correct_chunk, counts.found_guessed, counts.found_correct
    )
    by_type = {}
    for t in set(counts.t_found_correct.keys()).union(counts.t_found_guessed.keys()):
        by_type[t] = calculate_metrics(
            counts.t_correct_chunk[t], counts.t_found_guessed[t], counts.t_found_correct[t]
        )
    return overall, by_type

def _evaluate_iterable_exact(iterable, delimiter=ANY_SPACE, boundary='-X-', otag='O'):
    """Core evaluation logic for exact matching that processes an iterable of lines."""
    counts = EvalCounts()
    num_features = None
    in_correct = False
    last_correct = 'O'
    last_correct_type = ''
    last_guessed = 'O'
    last_guessed_type = ''

    for line in iterable:
        line = line.rstrip('\r\n')

        if delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == boundary:
            features = [boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct, last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct, last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

def _evaluate_iterable_overlap(iterable, delimiter=ANY_SPACE, boundary='-X-', otag='O'):
    """Core evaluation logic for overlap matching that processes an iterable of lines."""
    # Step 1: Parse all chunks from the iterable. This part is correct.
    correct_chunks = []
    guessed_chunks = []
    current_correct_chunk = None
    current_guessed_chunk = None
    token_counter = 0
    correct_tags = 0
    num_features = None

    for line_num, line in enumerate(iterable):
        line = line.rstrip('\r\n')
        if delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' % (len(features), num_features))

        if len(features) == 0 or features[0] == boundary:
            features = [boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == boundary:
            guessed = 'O'

        if first_item != boundary:
            token_counter += 1
            if correct == guessed and correct_type == guessed_type:
                correct_tags += 1

        # Process correct chunk
        if correct in ('B', 'S'):
            if current_correct_chunk:
                correct_chunks.append(current_correct_chunk)
            current_correct_chunk = {'start': line_num, 'end': line_num, 'type': correct_type}
        elif correct == 'I' and current_correct_chunk and current_correct_chunk['type'] == correct_type:
            current_correct_chunk['end'] = line_num
        elif correct == 'O':
            if current_correct_chunk:
                correct_chunks.append(current_correct_chunk)
                current_correct_chunk = None
        elif correct == 'E':
            if current_correct_chunk and current_correct_chunk['type'] == correct_type:
                current_correct_chunk['end'] = line_num
                correct_chunks.append(current_correct_chunk)
                current_correct_chunk = None
            else:
                correct_chunks.append({'start': line_num, 'end': line_num, 'type': correct_type})

        # Process guessed chunk
        if guessed in ('B', 'S'):
            if current_guessed_chunk:
                guessed_chunks.append(current_guessed_chunk)
            current_guessed_chunk = {'start': line_num, 'end': line_num, 'type': guessed_type}
        elif guessed == 'I' and current_guessed_chunk and current_guessed_chunk['type'] == guessed_type:
            current_guessed_chunk['end'] = line_num
        elif guessed == 'O':
            if current_guessed_chunk:
                guessed_chunks.append(current_guessed_chunk)
                current_guessed_chunk = None
        elif guessed == 'E':
            if current_guessed_chunk and current_guessed_chunk['type'] == guessed_type:
                current_guessed_chunk['end'] = line_num
                guessed_chunks.append(current_guessed_chunk)
                current_guessed_chunk = None
            else:
                guessed_chunks.append({'start': line_num, 'end': line_num, 'type': guessed_type})

    if current_correct_chunk:
        correct_chunks.append(current_correct_chunk)
    if current_guessed_chunk:
        guessed_chunks.append(current_guessed_chunk)

    # Step 2: Calculate TP, FP, FN for overlap metrics. This is the key fix.
    # We will not use the standard counts object for the final metrics.
    
    # Group chunks by type for per-type evaluation
    correct_chunks_by_type = defaultdict(list)
    guessed_chunks_by_type = defaultdict(list)
    for chunk in correct_chunks:
        correct_chunks_by_type[chunk['type']].append(chunk)
    for chunk in guessed_chunks:
        guessed_chunks_by_type[chunk['type']].append(chunk)

    all_types = set(correct_chunks_by_type.keys()).union(guessed_chunks_by_type.keys())
    
    # We will build the by_type_metrics dictionary directly
    by_type_metrics = {}
    
    # We also need to calculate overall TP, FP, FN by summing up per-type values
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for chunk_type in all_types:
        correct_set = correct_chunks_by_type.get(chunk_type, [])
        guessed_set = guessed_chunks_by_type.get(chunk_type, [])
        
        # In overlap mode:
        # TP = Number of correct chunks overlapped by at least one guessed chunk
        # FP = Number of guessed chunks with NO overlap
        # FN = Number of correct chunks with NO overlap
        
        overlapped_correct_indices = set()
        overlapped_guessed_indices = set()
        
        for i, guessed_chunk in enumerate(guessed_set):
            for j, correct_chunk in enumerate(correct_set):
                if not (guessed_chunk['end'] < correct_chunk['start'] or guessed_chunk['start'] > correct_chunk['end']):
                    overlapped_correct_indices.add(j)
                    overlapped_guessed_indices.add(i)
        
        tp = len(overlapped_correct_indices)
        fp = len(guessed_set) - len(overlapped_guessed_indices)
        fn = len(correct_set) - tp
        
        # Calculate precision, recall, and f1 for this type
        p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
        r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
        f = 0 if p + r == 0 else 2 * p * r / (p + r)
        
        # Store the metrics for this type
        by_type_metrics[chunk_type] = Metrics(tp, fp, fn, p, r, f)
        
        # Add to overall counts
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    # Step 3: Calculate overall metrics from the summed TP, FP, FN
    overall_p = 0 if overall_tp + overall_fp == 0 else 1.*overall_tp / (overall_tp + overall_fp)
    overall_r = 0 if overall_tp + overall_fn == 0 else 1.*overall_tp / (overall_tp + overall_fn)
    overall_f = 0 if overall_p + overall_r == 0 else 2 * overall_p * overall_r / (overall_p + overall_r)
    overall_metrics = Metrics(overall_tp, overall_fp, overall_fn, overall_p, overall_r, overall_f)

    # Step 4: Return the correctly calculated metrics
    return overall_metrics, by_type_metrics, token_counter, correct_tags, len(correct_chunks), len(guessed_chunks)

def _evaluate_iterable(iterable, delimiter=ANY_SPACE, boundary='-X-', otag='O', mode='exact'):
    """Core evaluation logic that processes an iterable of lines."""
    if mode == 'exact':
        return _evaluate_iterable_exact(iterable, delimiter, boundary, otag)
    elif mode == 'overlap':
        return _evaluate_iterable_overlap(iterable, delimiter, boundary, otag)
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}. Use 'exact' or 'overlap'.")

def evaluate_from_file(file_path, delimiter=ANY_SPACE, boundary='-X-', otag='O', mode='exact'):
    """
    Evaluates a CoNLL format file and returns the results.

    Args:
        file_path (str): Path to the CoNLL format file.
        delimiter (str): The character delimiting items in input.
                         Defaults to any space.
        boundary (str): The sentence boundary marker.
        otag (str): The outside tag.
        mode (str): Evaluation mode, either 'exact' or 'overlap'.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        if mode == 'exact':
            counts = _evaluate_iterable_exact(f, delimiter, boundary, otag)
            overall_metrics, by_type_metrics = get_metrics(counts)
            # Use the original counts for the results
            found_correct = counts.found_correct
            found_guessed = counts.found_guessed
            correct_phrases = counts.correct_chunk
            token_counter = counts.token_counter
            correct_tags = counts.correct_tags
        elif mode == 'overlap':
            # The new overlap function returns everything we need directly
            overall_metrics, by_type_metrics, token_counter, correct_tags, found_correct, found_guessed = _evaluate_iterable_overlap(f, delimiter, boundary, otag)
            # For overlap, correct_phrases is the overall TP count
            correct_phrases = overall_metrics.tp
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}. Use 'exact' or 'overlap'.")

    results = {
        'processed_tokens': token_counter,
        'found_correct_phrases': found_correct,
        'found_guessed_phrases': found_guessed,
        'correct_phrases': correct_phrases,
        'accuracy': (100. * correct_tags / token_counter) if token_counter > 0 else 0,
        'overall': {
            'precision': overall_metrics.prec,
            'recall': overall_metrics.rec,
            'f1': overall_metrics.fscore
        },
        'by_type': {
            tag: {
                'TP': m.tp,
                'FP': m.fp,
                'FN': m.fn,
                'precision': m.prec,
                'recall': m.rec,
                'f1': m.fscore,
                'found_guessed': m.tp + m.fp # This is the number of guessed chunks for this type
            } for tag, m in by_type_metrics.items()
        }
    }
    return results

def print_report(results, out=None):
    """Prints a formatted report from the results dictionary."""
    if out is None:
        out = sys.stdout

    out.write(f"processed {results['processed_tokens']} tokens with {results['found_correct_phrases']} phrases; ")
    out.write(f"found: {results['found_guessed_phrases']} phrases; correct: {results['correct_phrases']}.\n")

    if results['processed_tokens'] > 0:
        out.write(f"accuracy: {results['accuracy']:6.2f}%%; ")
        out.write(f"precision: {results['overall']['precision']*100:6.2f}%%; ")
        out.write(f"recall: {results['overall']['recall']*100:6.2f}%%; ")
        out.write(f"FB1: {results['overall']['f1']*100:6.2f}\n")

    for i, m in sorted(results['by_type'].items()):
        out.write(f'{i:17s}: ')
        out.write(f"precision: {m['precision']*100:6.2f}%%; ")
        out.write(f"recall: {m['recall']*100:6.2f}%%; ")
        out.write(f"FB1: {m['f1']*100:6.2f}  {m['found_guessed']}\n")


import os
import json
from pathlib import Path

def _find_file(base_dirs, rel_paths):
    for b in base_dirs:
        for r in rel_paths:
            p = Path(b) / r
            if p.exists():
                return p
    return None

def combine_files_as_txt(file_num, ground_truth_sub, predicted_BIO_sub=None, folder_to_save=None):
    from src.common.config import PROJECT_ROOT, OUTPUT_DIR, DATA_DIR
    file_num_str = str(file_num).replace('.bio', '').replace('.txt', '')
    
    # Candidate search paths for ground truth BIO
    gt_candidates = [
        OUTPUT_DIR / "ground_truth_BIO" / ground_truth_sub / f"{file_num_str}.bio",
        DATA_DIR / "splits" / ground_truth_sub / f"{file_num_str}.bio",
        DATA_DIR / ground_truth_sub / f"{file_num_str}.bio",
        PROJECT_ROOT / "output" / "ground_truth_BIO" / ground_truth_sub / f"{file_num_str}.bio",
    ]
    gt_path = None
    for p in gt_candidates:
        if p.exists():
            gt_path = p
            break
            
    # Candidate search paths for predicted BIO
    pred_candidates = []
    if predicted_BIO_sub:
        pred_candidates.extend([
            OUTPUT_DIR / "predicted" / "BIO" / predicted_BIO_sub / f"{file_num_str}.bio",
            OUTPUT_DIR / predicted_BIO_sub / f"{file_num_str}.bio",
            Path(predicted_BIO_sub) / f"{file_num_str}.bio",
            PROJECT_ROOT / "output" / "predicted" / "BIO" / predicted_BIO_sub / f"{file_num_str}.bio",
        ])
    else:
        pred_candidates.extend([
            OUTPUT_DIR / "predicted" / "BIO" / f"{file_num_str}.bio",
            OUTPUT_DIR / "predicted" / f"{file_num_str}.bio",
            PROJECT_ROOT / "output" / "predicted" / "BIO" / f"{file_num_str}.bio",
        ])
    pred_path = None
    for p in pred_candidates:
        if p.exists():
            pred_path = p
            break

    ground_truth = []
    predicted = []
    final_input_to_eval = []

    if gt_path and gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as gf:
            for line in gf.read().strip().split("\n"):
                parts = tuple(line.split())
                if len(parts) >= 2:
                    ground_truth.append((parts[0], parts[-1]))

    if pred_path and pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as pf:
            for line in pf.read().strip().split("\n"):
                parts = tuple(line.split())
                if len(parts) >= 2:
                    predicted.append((parts[0], parts[-1]))

    max_len = max(len(ground_truth), len(predicted))
    for i in range(max_len):
        gt_tok, gt_tag = ground_truth[i] if i < len(ground_truth) else ("<UNK>", "O")
        pr_tok, pr_tag = predicted[i] if i < len(predicted) else ("<UNK>", "O")
        tok = gt_tok if gt_tok != "<UNK>" else pr_tok
        if gt_tok == pr_tok:
            final_input_to_eval.append(f"{tok}\t{gt_tag}\t{pr_tag}")
        else:
            final_input_to_eval.append(f"{tok}\t{gt_tag}\t{pr_tag}")

    save_dir = OUTPUT_DIR / "combined_to_eval"
    if folder_to_save:
        save_dir = save_dir / folder_to_save
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{file_num_str}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_input_to_eval))

    return str(out_path), file_num_str


def evaluate_one_file(file_name, ground_truth_sub="train_phase", mode="overlap", predicted_BIO_sub=None, folder_to_save=None):
    from src.common.config import OUTPUT_DIR
    out_path, base_id = combine_files_as_txt(file_name, ground_truth_sub, predicted_BIO_sub, folder_to_save)
    results = evaluate_from_file(out_path, mode=mode)
    
    save_dir = OUTPUT_DIR / "combined_to_eval"
    if folder_to_save:
        save_dir = save_dir / folder_to_save
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{base_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Result for {base_id} saved to {json_path}")
    return results


def evaluate_all_files(directory_path, ground_truth_sub="test_phase", mode="exact", predicted_BIO_sub=None, folder_to_save=None):
    from src.common.config import OUTPUT_DIR
    dir_p = Path(directory_path)
    if not dir_p.exists():
        dir_p = OUTPUT_DIR / directory_path
    
    files_list = sorted([f.name for f in dir_p.glob("*.bio")]) if dir_p.exists() else []
    if not files_list:
        print(f"No .bio files found in {dir_p}")
        return {}

    for fn in files_list:
        base = fn.replace(".bio", "")
        evaluate_one_file(base, ground_truth_sub, mode=mode, predicted_BIO_sub=str(dir_p) if not predicted_BIO_sub else predicted_BIO_sub, folder_to_save=folder_to_save)

    save_dir = OUTPUT_DIR / "combined_to_eval"
    if folder_to_save:
        save_dir = save_dir / folder_to_save
    
    total_tp, total_fp, total_fn = 0, 0, 0
    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)
    per_file_results = {}

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    for fn in files_list:
        base = fn.replace(".bio", "")
        json_file = save_dir / f"{base}.json"
        if not json_file.exists():
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            r = json.load(f)

        tp_f, fp_f, fn_f = 0, 0, 0
        for t, vals in r.get("by_type", {}).items():
            TP_t = vals.get("TP", 0)
            FP_t = vals.get("FP", 0)
            FN_t = vals.get("FN", 0)
            tp_f += TP_t
            fp_f += FP_t
            fn_f += FN_t
            type_tp[t] += TP_t
            type_fp[t] += FP_t
            type_fn[t] += FN_t

        prec_f = safe_div(tp_f, tp_f + fp_f)
        rec_f = safe_div(tp_f, tp_f + fn_f)
        f1_f = safe_div(2 * prec_f * rec_f, prec_f + rec_f)

        per_file_results[base] = {
            "tp": tp_f, "fp": fp_f, "fn": fn_f,
            "precision": prec_f, "recall": rec_f, "f1": f1_f
        }
        total_tp += tp_f
        total_fp += fp_f
        total_fn += fn_f

    overall_precision = safe_div(total_tp, total_tp + total_fp)
    overall_recall = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)

    per_type_results = {}
    for t in type_tp.keys():
        tp = type_tp[t]
        fp = type_fp[t]
        fn = type_fn[t]
        prec_t = safe_div(tp, tp + fp)
        rec_t = safe_div(tp, tp + fn)
        f1_t = safe_div(2 * prec_t * rec_t, prec_t + rec_t)
        per_type_results[t] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec_t, "recall": rec_t, "f1": f1_t
        }

    final_result = {
        "overall_micro": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": overall_precision, "recall": overall_recall, "f1": overall_f1
        },
        "per_file": per_file_results,
        "per_type": per_type_results
    }
    return final_result

