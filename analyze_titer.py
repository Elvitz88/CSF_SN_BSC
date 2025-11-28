import cv2
import numpy as np
import os

def get_titer_value(col_index):
    return f"1:{2**(col_index + 1)}"

def analyze_plate(image_path):
    print(f"Analyzing: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    output_img = img.copy()
    
    # 1. Resize
    target_width = 1600
    scale_ratio = target_width / img.shape[1]
    target_height = int(img.shape[0] * scale_ratio)
    img_resized = cv2.resize(img, (target_width, target_height))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. Find Contours (Focus on Red/Dark wells first)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 5)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_wells = []
    min_area = (target_width / 40)**2 
    max_area = (target_width / 12)**2 
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 0.7 < aspect_ratio < 1.3: 
                valid_wells.append([x + w/2, y + h/2, w]) 

    print(f"Detected {len(valid_wells)} potential well candidates.")
    
    if len(valid_wells) < 10:
        print("Not enough wells detected.")
        return

    # 3. Determine Grid Geometry (Pitch) from detected wells
    valid_wells = np.array(valid_wells)
    
    # Calculate average distance between nearest neighbors to find Pitch
    distances = []
    for i in range(len(valid_wells)):
        for j in range(i + 1, len(valid_wells)):
            dist = np.sqrt(np.sum((valid_wells[i][:2] - valid_wells[j][:2])**2))
            distances.append(dist)
            
    # Heuristic: The most common small distance is likely the Neighbor Pitch
    # Filter large distances
    distances = [d for d in distances if d < target_width/10]
    if not distances:
        print("Could not determine grid pitch.")
        return
        
    # Use histogram/median to find the pitch
    # Wells are typically ~width/14 apart
    expected_pitch = target_width / 14
    valid_distances = [d for d in distances if 0.8*expected_pitch < d < 1.2*expected_pitch]
    
    if not valid_distances:
        # Fallback if detection is messy
        avg_pitch = expected_pitch
        print(f"Using theoretical pitch: {avg_pitch:.2f}")
    else:
        avg_pitch = np.median(valid_distances)
        print(f"Detected Grid Pitch: {avg_pitch:.2f} px")

    # 4. Fit Grid (RANSAC-like approach)
    # We have the pitch. Now we need to find the "Top-Left" anchor (A1).
    # Since we mostly detected the RED wells (on the right), we need to shift back.
    
    # Find the bounding box of the DETECTED cluster
    cluster_min_x = np.min(valid_wells[:, 0])
    cluster_max_x = np.max(valid_wells[:, 0])
    cluster_min_y = np.min(valid_wells[:, 1])
    cluster_max_y = np.max(valid_wells[:, 1])
    
    # Snap the cluster to a virtual grid
    # Assume the right-most detected well is in column 12 (or close to it)
    # Or assume the top-most is Row A
    
    # Let's iterate through possible "Offset" adjustments
    # We want to minimize the error of detected wells against grid points
    
    best_start_x = 0
    best_start_y = 0
    min_error = float('inf')
    
    # Try anchoring the detected cluster to different columns
    # e.g., Maybe the cluster starts at Col 5? or Col 6?
    # We assume the cluster includes the RIGHT side (Col 12) generally.
    
    # Let's just define the grid based on the cluster bounds + pitch
    # We know the plate has 12 columns.
    # Estimate how many columns fit in the detected width
    detected_cols = round((cluster_max_x - cluster_min_x) / avg_pitch) + 1
    detected_rows = round((cluster_max_y - cluster_min_y) / avg_pitch) + 1
    
    print(f"Cluster spans approx {detected_cols} cols x {detected_rows} rows")
    
    # Align the grid:
    # We assume the Top-Left detected well is (A, StartCol)
    # But we don't know StartCol.
    # Heuristic: The plate is centered in the image usually?
    # OR: We assume the RIGHTMOST detected well is likely Col 12 (since it's usually positive/red)
    
    # Let's find the right-most well
    rightmost_idx = np.argmax(valid_wells[:, 0])
    rightmost_x = valid_wells[rightmost_idx, 0]
    rightmost_y = valid_wells[rightmost_idx, 1] # But this might not be row A
    
    # Find top-most well
    topmost_idx = np.argmin(valid_wells[:, 1])
    topmost_y = valid_wells[topmost_idx, 1]
    
    # Define Grid Start (A1) relative to these extremes
    # A1_y should be close to topmost_y
    # A1_x is harder. 
    # Let's assume the Rightmost well is in Column 12.
    estimated_A1_x = rightmost_x - (11 * avg_pitch)
    estimated_A1_y = topmost_y
    
    # Refine A1_x: Check if it pushes A1 too far left (off image)
    if estimated_A1_x < 0:
        # Maybe rightmost wasn't col 12.
        estimated_A1_x = 50 # Safety margin
        
    # Refine A1_y
    # Check if we missed row A? 
    # Usually finding Row A is easier than Col 1.
    
    # Let's generate the grid points based on this A1 estimate
    start_x = estimated_A1_x
    start_y = estimated_A1_y
    
    # Re-center optimization (optional but good)
    # Calculate average offset from all points modulo pitch
    
    # Final Grid Generation
    final_grid_points = []
    results_table = []
    row_labels = "ABCDEFGH"
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # ROI size
    roi_radius_resized = int(avg_pitch * 0.35)
    
    for r in range(8):
        row_result = []
        curr_y = start_y + (r * avg_pitch)
        
        for c in range(12):
            curr_x = start_x + (c * avg_pitch)
            
            orig_x = int(curr_x / scale_ratio)
            orig_y = int(curr_y / scale_ratio)
            orig_r = int(roi_radius_resized / scale_ratio)
            
            # Boundary Check
            if orig_x < 0 or orig_x > img.shape[1] or orig_y < 0 or orig_y > img.shape[0]:
                row_result.append("?")
                continue

            roi_r_analysis = int(orig_r * 0.7)
            roi = img[orig_y-roi_r_analysis:orig_y+roi_r_analysis, 
                      orig_x-roi_r_analysis:orig_x+roi_r_analysis]
            
            status = "?"
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
                percent_red = (cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])) * 100
                is_positive = percent_red > 40
                status = "+" if is_positive else "-"
                
                color = (0, 0, 255) if is_positive else (255, 200, 0)
                cv2.circle(output_img, (orig_x, orig_y), orig_r, color, 3)
                
                # Draw Symbol (+/-)
                symbol = "+" if is_positive else "-"
                text_color = (0, 255, 255) if is_positive else (0, 255, 0) # Yellow for +, Green for -
                
                # Adjust font size based on circle size
                font_scale = orig_r / 25.0 
                thickness = 2
                
                # Center the text
                (text_w, text_h), _ = cv2.getTextSize(symbol, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = int(orig_x - text_w / 2)
                text_y = int(orig_y + text_h / 2)
                
                cv2.putText(output_img, symbol, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                
                # Debug numbers
                # if r == 0:
                #     cv2.putText(output_img, str(c+1), (orig_x-10, orig_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            row_result.append(status)

        titer = "No Inhibition"
        last_neg_index = -1
        for i, res in enumerate(row_result):
            if res == "-": last_neg_index = i
        if last_neg_index != -1:
            titer = get_titer_value(last_neg_index)
        elif all(res == "+" for res in row_result):
            titer = "< 1:2"
            
        results_table.append({
            "Row": row_labels[r],
            "Wells": row_result,
            "Titer": titer
        })

    # Draw Anchor Point
    orig_start_x = int(start_x / scale_ratio)
    orig_start_y = int(start_y / scale_ratio)
    cv2.circle(output_img, (orig_start_x, orig_start_y), 10, (0, 255, 0), -1) # Green Dot at A1 estimate

    output_filename = "analyzed_" + os.path.basename(image_path)
    cv2.imwrite(output_filename, output_img)
    print(f"Saved analyzed image to: {output_filename}")
    
    print("\n--- Analysis Results ---")
    print("Row | Well Results (1-12) | Titer")
    print("-" * 60)
    for row in results_table:
        wells_str = " ".join(row["Wells"])
        print(f" {row['Row']}  | {wells_str} | {row['Titer']}")

if __name__ == "__main__":
    target_image = "D:/SN-BSC/picture/15.jpeg"
    if os.path.exists(target_image):
        analyze_plate(target_image)
