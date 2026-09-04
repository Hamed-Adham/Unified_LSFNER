import os
import shutil
import time
import json
import random

# Import Smart Pre-Scan
from src.common.smart_pre_scan import calculate_global_density_stats

# --- REPORTING LOGIC ---
def generate_report(selected_files, output_folder, total_target_files):
    MAIN_CLASSES = [
        "Nutrition", "Physical_activities", "Substance_use",
        "Environmental_exposures", "Socioeconomic_factors", "Mental_health_practices",
        "Non_physical_leisure_time_activities", "Personal_care_products_and_cosmetic_procedures",
        "Sleep"
    ]
    
    target_per_main = total_target_files // 9
    target_remainder = total_target_files - (target_per_main * 9)
    
    baskets = {cls: [] for cls in MAIN_CLASSES}
    baskets["Remainder"] = []
    
    for entry in selected_files:
        # Convert entry format to report format
        # Entry: {file_name, majority_class, purity, total_entities, purity_on, majority_count}
        maj = entry["majority_class"]
        
        report_item = {
            "file_name": entry["file_name"],
            "label": maj,
            "purity_score": str(entry["purity"]),
            "weighted_purity": {
                "purity": entry["purity"],
                "majority_count": entry.get("majority_count", 0) # User requested: Count of entities matching the label
            },
            "density_metrics": {
                "density_raw": entry.get("density", 0.0),
                "combined_score": entry.get("combined_score", 0.0)
            },
            "total_entities": entry["total_entities"],
            "full_purity_desc": entry.get("purity_on", "")
        }
        
        if maj in MAIN_CLASSES:
            baskets[maj].append(report_item)
        else:
            baskets["Remainder"].append(report_item)
            
    # Verify
    report = {
        "summary": {
            "total_files": len(selected_files),
            "target_per_main": target_per_main,
            "target_remainder": target_remainder,
            "overall_status": "PENDING"
        },
        "verification_results": {},
        "baskets": {}
    }
    
    all_passed = True
    
    # Check Main
    for cls in MAIN_CLASSES:
        count = len(baskets[cls])
        status = "PASS" if count == target_per_main else "FAIL"
        if status == "FAIL": all_passed = False
        
        report["verification_results"][cls] = {"actual": count, "target": target_per_main, "status": status}
        report["baskets"][cls] = sorted(baskets[cls], key=lambda x: (x["weighted_purity"]["purity"], x["weighted_purity"]["majority_count"]), reverse=True)

    # Check Remainder
    rem_count = len(baskets["Remainder"])
    rem_status = "PASS" if rem_count == target_remainder else "FAIL"
    if rem_status == "FAIL": all_passed = False
    
    report["verification_results"]["Remainder"] = {"actual": rem_count, "target": target_remainder, "status": rem_status}
    report["baskets"]["Remainder"] = sorted(baskets["Remainder"], key=lambda x: (x["weighted_purity"]["purity"], x["weighted_purity"]["majority_count"]), reverse=True)
    
    report["summary"]["overall_status"] = "BALANCED" if all_passed else "UNBALANCED"
    
    with open(os.path.join(output_folder, "dataset_balance_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"Report generated: {report['summary']['overall_status']}")


# --- INLINED SELECTION LOGIC ---
def select_balanced_subset_inline(source_folder, output_folder, total_target_files, max_density_95=1.0, file_densities={}):
    """
    Selects balanced subset using Combined Score.
    Score = (0.5 * Purity/100) + (0.5 * Density/Max_Density_95)
    """
    # Defensive fix
    if isinstance(source_folder, list):
        if len(source_folder) == 1:
            source_folder = str(source_folder[0])
        else:
            source_folder = os.path.join(*source_folder)
    source_folder = str(source_folder)
    
    print(f"DEBUG: INLINE selection source='{source_folder}'")

    MAIN_CLASSES = [
        "Nutrition", "Physical_activities", "Substance_use",
        "Environmental_exposures", "Socioeconomic_factors", "Mental_health_practices",
        "Non_physical_leisure_time_activities", "Personal_care_products_and_cosmetic_procedures",
        "Sleep"
    ]
    
    target_per_main = total_target_files // 9
    target_remainder = total_target_files - (target_per_main * 9)
    
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return []

    bins = {cls: [] for cls in MAIN_CLASSES}
    bins["Remainder"] = [] 
    
    # SCAN
    files = os.listdir(source_folder)
    json_files = [f for f in files if f.endswith('.json')]
    
    for json_file in json_files:
        path = os.path.join(source_folder, json_file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            majority = data.get("majority_voted_class", "None")
            purity = float(data.get("purity", "0"))
            purity_on = data.get("purity_on", "")
            counts = data.get("entitiy_frequency_per_class", {})
            total_entities = sum(counts.values())
            
            # User Request: Weighted Purity Count should be count of Majority Label only
            majority_count = counts.get(majority, 0)
            
            # Retrieve Density
            base_name = data.get("file_name").replace('.ann', '')
            density = file_densities.get(base_name, 0.0)
            
            # Calculate Combined Score
            # Normalize Purity (0-100 -> 0-1)
            norm_purity = purity / 100.0
            
            # Normalize Density (0-Max -> 0-1, capped at 1.0)
            norm_density = 0.0
            if max_density_95 > 0:
                norm_density = density / max_density_95
                if norm_density > 1.0: norm_density = 1.0
                
            combined_score = (0.5 * norm_purity) + (0.5 * norm_density)
            
            entry = {
                "file_name": base_name,
                "majority_class": majority,
                "purity": purity,
                "total_entities": total_entities,
                "purity_on": purity_on,
                "majority_count": majority_count,
                "density": density,
                "combined_score": combined_score
            }
            
            if majority in MAIN_CLASSES:
                bins[majority].append(entry)
            else:
                bins["Remainder"].append(entry)
        except:
            pass

    # SELECT
    selected_files = []
    
    # NEW SORT KEY: Combined Score Descending
    def sort_key(e): 
        return e["combined_score"]
    
    for cls in MAIN_CLASSES:
        bin_files = sorted(bins[cls], key=sort_key, reverse=True)
        selected_files.extend(bin_files[:target_per_main])

    remainder_bin = sorted(bins["Remainder"], key=sort_key, reverse=True)
    selected_files.extend(remainder_bin[:target_remainder])
    
    # --- BACKFILL STRATEGY ---
    # User Requirement: Ensure total number equals given size, even if unbalanced.
    current_count = len(selected_files)
    shortage = total_target_files - current_count
    
    if shortage > 0:
        print(f"DEBUG: Shortage of {shortage} files for target {total_target_files}. Backfilling...")
        
        # Gather all candidates not yet selected
        # We need a set of selected filenames for fast lookup
        selected_names = set(e["file_name"] for e in selected_files)
        
        # Flatten all bins to search for extras
        all_candidates = []
        for cls in bins:
             all_candidates.extend(bins[cls])
             
        # Filter out already selected
        available_candidates = [e for e in all_candidates if e["file_name"] not in selected_names]
        
        # Sort by quality (Weighted Purity)
        available_candidates.sort(key=sort_key, reverse=True)
        
        # Take top 'shortage'
        backfill_files = available_candidates[:shortage]
        selected_files.extend(backfill_files)
        
        print(f"DEBUG: Backfilled {len(backfill_files)} files. New total: {len(selected_files)}")
        if len(selected_files) < total_target_files:
             print("WARNING: Pool exhausted! Could not reach target size.")

    # COPY
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Generate Report before returning
    generate_report(selected_files, output_folder, total_target_files)
        
    for file_entry in selected_files:
        base = file_entry["file_name"]
        for ext in ['.ann', '.txt']: # User modified this to skip json
            src = os.path.join(source_folder, base + ext)
            dst = os.path.join(output_folder, base + ext)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
    return [e["file_name"] for e in selected_files]


def safe_remove(path, retries=5, delay=1):
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            print(f"DEBUG: PermissionError removing {path}, retrying ({i+1}/{retries})...")
            time.sleep(delay)
    print(f"WARNING: Failed to remove {path} after retries.")

def safe_rmtree(path, retries=5, delay=1):
    for i in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return
        except PermissionError:
            print(f"DEBUG: PermissionError removing dir {path}, retrying ({i+1}/{retries})...")
            time.sleep(delay)
    print(f"WARNING: Failed to remove dir {path} after retries.")

# --- ORCHESTRATOR LOGIC ---
def orchestrate_unified(source_folder, subset_sizes, base_output_dir):
    print(f"--- Unified Orchestrator ---")
    
    timestamp = int(time.time())
    temp_folder = f"temp_processing_{timestamp}"
    
    safe_rmtree(temp_folder) # Ensure clean start
    shutil.copytree(source_folder, temp_folder)
    
    # --- PHASE 1: SMART PRE-SCAN ---
    print("\n--- Performing Smart Pre-Scan ---")
    max_density_95, file_densities = calculate_global_density_stats(temp_folder)
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    for i, size in enumerate(subset_sizes):
        subset_name = f"subset_{i+1}_size_{size}"
        out_path = os.path.join(base_output_dir, subset_name)
        
        print(f"Generating {subset_name}...")
        
        # Pass Pre-Scan Stats
        selected = select_balanced_subset_inline(
            temp_folder, 
            out_path, 
            size, 
            max_density_95=max_density_95, 
            file_densities=file_densities
        )
        
        # Remove
        for base in selected:
            for ext in ['.ann', '.txt', '.json']:
                p = os.path.join(temp_folder, base + ext)
                safe_remove(p)
                    
    # Cleanup
    safe_rmtree(temp_folder)
    print("Done.")

if __name__ == "__main__":
    SOURCE = "data/Relabaled_Train_folder_noDisease_statistics_Added"
    # TO create Only validation set and get Train - Validation Run below 
    # TOTAL_SUBSETS = [40, 120]
    TOTAL_SUBSETS = [40, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    BASE_OUT = "data/Genereted_Subsets"
    
    orchestrate_unified(SOURCE, TOTAL_SUBSETS, BASE_OUT)
