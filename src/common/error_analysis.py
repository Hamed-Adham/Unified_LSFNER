import os
import json

def generate_eval_txt_content(ground_truth_str, predicted_str):
    """
    Takes BIO formatted ground truth and predicted strings and generates
    the 4-column format (Token, Correct_Tag, Guessed_Tag, Error_Label)
    """
    ground_truth = []
    predicted = []
    
    for line in ground_truth_str.strip().split("\n"):
        parts = tuple(line.split())
        if len(parts) >= 2:
            ground_truth.append((parts[0], parts[1]))
            
    for line in predicted_str.strip().split("\n"):
        parts = tuple(line.split())
        if len(parts) >= 2:
            predicted.append((parts[0], parts[1]))
            
    final_input_to_eval = []
    # Match line by line
    max_len = max(len(ground_truth), len(predicted))
    for i in range(max_len):
        gt_token = ground_truth[i][0] if i < len(ground_truth) else "-"
        gt_tag = ground_truth[i][1] if i < len(ground_truth) else "O"
        
        pr_token = predicted[i][0] if i < len(predicted) else "-"
        pr_tag = predicted[i][1] if i < len(predicted) else "O"
        
        # Take the token from ground truth ideally, or predicted if gt is missing
        token = gt_token if gt_token != "-" else pr_token
        
        if gt_tag == 'O' and pr_tag == 'O':
            label = 'TN'
        elif gt_tag == 'O' and pr_tag != 'O':
            label = 'FP'
        elif gt_tag != 'O' and pr_tag == 'O':
            label = 'FN'
        elif gt_tag == pr_tag:
            label = 'TP'
        elif gt_tag[0] == pr_tag[0]:
            label = 'CLSS_ERR'
        else:
            label = 'BIO_ERR'
        
        final_input_to_eval.append(f"{token}\t{gt_tag}\t{pr_tag}\t{label}")
            
    return "\n".join(final_input_to_eval)


def extract_discrepancies_from_text(txt_content):
    """
    Parses a 4-column .txt content and groups the tokens by their error labels:
    TP, TN, FP, FN, BIO_ERR, CLSS_ERR.
    """
    lines = txt_content.strip().split('\n')
        
    discrepancies = {
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': [],
        'BIO_ERR': [],
        'CLSS_ERR': []
    }
    
    for i, line in enumerate(lines):
        parts = line.split('\t')
        if len(parts) < 4:
            parts = line.split()
            if len(parts) < 4:
                continue
            
        token = parts[0]
        correct_tag = parts[1]
        guessed_tag = parts[2]
        label = parts[3]
        
        if label in discrepancies:
            discrepancies[label].append({
                'line_number': i + 1,
                'token': token,
                'correct_tag': correct_tag,
                'guessed_tag': guessed_tag
            })
            
    return discrepancies


def generate_and_analyze_errors(ground_truth_str, predicted_str, output_txt_path=None, output_json_path=None):
    """
    Generates the text evaluation file and directly analyzes the errors,
    saving the result to JSON if path provided.
    Runs the full workflow requested.
    """
    txt_content = generate_eval_txt_content(ground_truth_str, predicted_str)
    
    if output_txt_path:
        os.makedirs(os.path.dirname(output_txt_path) or ".", exist_ok=True)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
            
    discrepancies = extract_discrepancies_from_text(txt_content)
    
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(discrepancies, f, indent=4)
            
    return txt_content, discrepancies
