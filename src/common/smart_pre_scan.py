import os
import json
import math

def calculate_global_density_stats(source_folder):
    """
    Scans the source folder to calculate density statistics for all files.
    Density = Majority_Label_Count / Text_Length
    
    Returns:
        max_density_95 (float): The 95th percentile of density values (to handle outliers).
        file_densities (dict): {filename_no_ext: density_value}
    """
    print(f"--- Smart Pre-Scan: Analyzing Density in '{source_folder}' ---")
    
    if not os.path.exists(source_folder):
        print(f"Error: Folder '{source_folder}' not found.")
        return 1.0, {}

    files = os.listdir(source_folder)
    json_files = [f for f in files if f.endswith('.json')]
    
    densities = []
    file_densities = {}
    
    processed_count = 0
    
    for json_file in json_files:
        base_name = json_file.replace('.json', '')
        txt_file = base_name + '.txt'
        
        json_path = os.path.join(source_folder, json_file)
        txt_path = os.path.join(source_folder, txt_file)
        
        if not os.path.exists(txt_path):
            continue
            
        try:
            # 1. Get Text Length
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                text_len = len(text)
                
            if text_len == 0:
                file_densities[base_name] = 0.0
                densities.append(0.0)
                continue

            # 2. Get Majority Count
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            majority_class = data.get("majority_voted_class", "None")
            counts = data.get("entitiy_frequency_per_class", {})
            
            # Use specific count for majority class
            majority_count = counts.get(majority_class, 0)
            
            # 3. Calculate Density
            density = majority_count / text_len
            
            file_densities[base_name] = density
            densities.append(density)
            processed_count += 1
            
        except Exception as e:
            # Silently skip errors to avoid spamming
            pass

    if not densities:
        print("Warning: No valid densities calculated.")
        return 1.0, {}

    # 4. Calculate 95th Percentile
    densities.sort()
    count = len(densities)
    # index for 95th percentile
    idx = int(0.95 * count)
    if idx >= count:
        idx = count - 1
        
    max_density_95 = densities[idx]
    
    # Safety: avoid division by zero later if all are 0
    if max_density_95 == 0:
        if densities[-1] > 0:
            max_density_95 = densities[-1] # Fallback to absolute max
        else:
            max_density_95 = 1.0 # Default to 1.0 to avoid errors

    print(f"Processed {processed_count} files.")
    print(f"Global Density Stats -> 95th Percentile: {max_density_95:.6f} (Max Absolute: {densities[-1]:.6f})")
    
    return max_density_95, file_densities

if __name__ == "__main__":
    # Test run
    FOLDER = "data/Relabaled_Train_folder_noDisease_statistics_Added"
    stats, _ = calculate_global_density_stats(FOLDER)
