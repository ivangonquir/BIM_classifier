import trimesh
import os
import pandas as pd
import numpy as np


#####################################################
###################### PART A #######################
#####################################################

def extract_features_file(file_path):
        try:
            mesh = trimesh.load(file_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # AABB for orientation
            aabb = mesh.bounding_box.extents
            is_vertical = 1 if np.argmax(aabb) == 2 else 0
            
            # Robust OBB for proportions
            try:
                obb = np.sort(mesh.bounding_box_oriented.extents)
            except:
                obb = np.sort(aabb)
                
            t, w, l = obb[0], obb[1], obb[2] # Thick, Wide, Long
            
            return [
                l / t,          # Slenderness
                l / w,          # Aspect Ratio
                w / t,          # Flatness
                is_vertical,    # Orientation (0 or 1)
                mesh.area / mesh.volume # SA/Vol ratio
            ]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
        

def create_dataframe(data_dir):
    classes = ['Beams', 'Columns', 'Walls', 'Slabs']
    dataset = []

    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path): continue
        for file in os.listdir(class_path):
            if file.endswith(".obj"):
                path = os.path.join(class_path, file)
                feats = extract_features_file(path)
                if feats:
                    dataset.append(feats + [label_idx,path])

    df = pd.DataFrame(dataset, columns=['slenderness', 'aspect', 'flatness', 'vertical', 'sa_vol', 'label', 'file_path'])

    return df
        

def extract_features_from_mesh(mesh):
    """Calculates geometric ratios for classification."""
    aabb = mesh.bounding_box.extents
    # Verticality: Is the Z-axis the longest?
    is_vertical = 1 if np.argmax(aabb) == 2 else 0
    
    try:
        obb = np.sort(mesh.bounding_box_oriented.extents)
    except:
        obb = np.sort(aabb)
        
    t, w, l = obb[0], obb[1], obb[2]
    return {
        "slenderness": l / max(t, 1e-6),
        "aspect": l / max(w, 1e-6),
        "flatness": w / max(t, 1e-6),
        "vertical": is_vertical,
        "sa_vol": mesh.area / max(mesh.volume, 1e-6)
    }



#####################################################
###################### PART B #######################
#####################################################

def get_contextual_features(part_mesh, warehouse_aabb):
    """
    Analyzes where a part lives in the overall warehouse volume.
    """
    # 1. Global building dimensions
    warehouse_height = warehouse_aabb.extents[2]
    warehouse_min_z = warehouse_aabb.bounds[0][2]
    
    # 2. Local part dimensions
    part_min_z = part_mesh.bounds[0][2]
    
    # 3. Calculate relative height (0.0 = Ground, 1.0 = Roof)
    relative_elevation = (part_min_z - warehouse_min_z) / warehouse_height
    
    return {
        "elevation": relative_elevation,
        "is_on_ground": relative_elevation < 0.1, # Grounded if in bottom 10%
        "is_in_roof": relative_elevation > 0.7     # Likely truss if in top 30%
    }


def process_full_warehouse(file_path, model, classes):
    """
    Modular function to handle the entire Part B pipeline.
    Returns: A list of tuples (mesh, predicted_label, color, context_data)
    """
    # 1. Load and get Global Context
    warehouse_mesh = trimesh.load(file_path, force = 'mesh')
    warehouse_aabb = warehouse_mesh.bounding_box
    
    # 2. Segment
    parts = warehouse_mesh.split(only_watertight=False)
    
    processed_parts = []
    MAX_PARTS = 500
    color_map = {"Beams": "red", "Columns": "blue", "Walls": "green", "Slabs": "yellow"}
    print(f"MAX PARTS = {MAX_PARTS}")
    for p in parts[:MAX_PARTS]:
        # Optimization: Skip tiny objects to save memory and processing time
        if p.area < 0.2: continue

        # Optimization: Decimate the mesh if it has too many faces
        if len(p.faces) > 2000:
            p = p.simplify_quadratic_decimation(1000)

            
        # Geometry Features
        feats = extract_features_from_mesh(p)
        # Task 2: Contextual Features
        context = get_contextual_features(p, warehouse_aabb)
        
        if feats:
            pred_idx = model.predict([feats])[0]
            label = classes[pred_idx]
            
            # Solve Task 2: Refine Label/Color based on context
            final_color = color_map.get(label, "white")
            if label == "Columns" and not context['is_on_ground']:
                final_color = "cyan" # Flag as Truss member
                label = "Truss Stud (Contextual Refinement)"
            
            processed_parts.append({
                "mesh": p,
                "label": label,
                "color": final_color,
                "elevation": context['elevation']
            })
            
    return processed_parts