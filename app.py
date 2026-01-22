import os
import joblib 
import trimesh
import pandas as pd
import pyvista as pv
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from stpyvista import stpyvista
from core.geometry import extract_features_from_mesh, get_contextual_features
#from core.model_utils import load_test_results
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report



# --- CONFIG & CONSTANTS ---
# --- PAGE SETUP ---
st.set_page_config(page_title="BIM Element Classifier", layout="wide")
st.title("🏗️ 3D Structural Element Classifier")
CLASSES = ['Beams', 'Columns', 'Walls', 'Slabs']
COLOR_MAP = {"Beams": "red", "Columns": "blue", "Walls": "green", "Slabs": "yellow"}


# --- DATA LOADING ---
@st.cache_resource
def load_pipeline_assets():
    model = joblib.load("models/bim_model.joblib")
    test_results = pd.read_csv("data/processed/test_results.csv")

    return model, test_results 


model, test_results = load_pipeline_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("🏗️ BIM Navigation")
page = st.sidebar.radio("Go to", ["Performance Metrics", "Mesh Explorer", "Warehouse Battle Test"])


# --- PAGE: PERFORMANCE ---
if page == "Performance Metrics":

    st.header("📊 Model Evaluation & Geometric Justification")
    
    # 1. High-Level Metrics Row
    acc = accuracy_score(test_results['label'], test_results['preds'])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy", f"{acc:.1%}")
    c2.metric("Classes Evaluated", len(CLASSES))
    c3.metric("Test Samples", len(test_results))

    st.divider()

    # 2. Confusion Matrix & Separability
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(test_results['label'], test_results['preds'])
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues", ax=ax_cm)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig_cm)
        st.caption("This matrix shows which classes are being confused. Notice the high diagonal accuracy.")

    with col_right:
        st.subheader("Mathematical Separability")
        # Scatter plot proving classes occupy different 'geometric spaces'
        fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
        # Map labels to names for better legend
        test_results['Class'] = test_results['label'].map(lambda x: CLASSES[x])
        

        sns.scatterplot(
            data=test_results, 
            x='slenderness', 
            y='sa_vol', 
            hue='Class', 
            palette='bright',
            ax=ax_scatter
        )
        plt.yscale('log') # Ratios often look better in log scale
        plt.title("Slenderness vs SA/Vol Ratio")
        st.pyplot(fig_scatter)
        st.caption("Proof that classes are mathematically separable using extracted features.")

    st.divider()

    # 3. Feature Importance & Failure Analysis
    col_feat, col_fail = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance")
        # We extract importance from our loaded model
        importances = model.feature_importances_
        feature_names = ['Slenderness', 'Aspect Ratio', 'Flatness', 'Verticality', 'SA/Vol']
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        fig_feat, ax_feat = plt.subplots()
        sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis', ax=ax_feat)
        st.pyplot(fig_feat)
        st.info("The model relies heavily on Verticality and Slenderness to distinguish BIM elements.")

    with col_fail:
        st.subheader("⚠️ Failure Analysis")
        # Find 3 random errors
        errors = test_results[test_results['label'] != test_results['preds']]
        
        if len(errors) > 0:
            st.warning(f"Detected {len(errors)} misclassifications in the test set.")
            st.info(f"Showing {min(2, len(errors))} / {len(errors)} errors")
            for i in range(min(2, len(errors))):
                err = errors.iloc[i]
                st.write(f"**File:** `{os.path.basename(err['file_path'])}`")
                st.write(f"Actual: {CLASSES[int(err['label'])]} | Predicted: :red[{CLASSES[int(err['preds'])]}]")
                st.write("---")
        else:
            st.success("Perfect classification on the test set!")


elif page == "Mesh Explorer":
    st.header("🔍 Individual Mesh Inspection")
    
    # Filters
    selected_class = st.selectbox("Select Element Type", CLASSES)
    class_idx = CLASSES.index(selected_class)
    filtered_data = test_results[test_results['label'] == class_idx]
    
    # Show correct vs incorrect filter
    show_errors = st.checkbox("Show only misclassified elements")
    if show_errors:
        filtered_data = filtered_data[filtered_data['label'] != filtered_data['preds']]

    selected_file = st.selectbox("Select a file to inspect", filtered_data['file_path'].tolist())

    if selected_file:
        feat_data = filtered_data[filtered_data['file_path'] == selected_file].iloc[0]
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.info(f"Visualizing: {os.path.basename(selected_file)}")
            
            # --- WEB-NATIVE 3D RENDERING ---
            # 1. Load the mesh using PyVista
            mesh = pv.read(selected_file)
            
            # 2. Setup the plotter
            plotter = pv.Plotter(window_size=[600, 400])
            
            # 3. Determine color based on prediction
            color_map = {"Beams": "red", "Columns": "blue", "Walls": "green", "Slabs": "yellow"}
            pred_class = CLASSES[int(feat_data['preds'])]
            obj_color = color_map.get(pred_class, "white")
            
            plotter.add_mesh(mesh, color=obj_color, show_edges=True)
            plotter.view_isometric()
            plotter.background_color = "#f0f2f6" # Match Streamlit background
            
            # 4. Render directly in the browser
            stpyvista(plotter, key=f"pv_{selected_file}")

        with c2:
            st.subheader("Extracted Features")
            feat_data = filtered_data[filtered_data['file_path'] == selected_file].iloc[0]
            st.json({
                "Prediction": CLASSES[int(feat_data['preds'])],
                "Actual": CLASSES[int(feat_data['label'])],
                "Slenderness": f"{feat_data['slenderness']:.2f}",
                "SA/Vol Ratio": f"{feat_data['sa_vol']:.4f}",
                "Is Vertical": bool(feat_data['vertical'])
            })


elif page == "Warehouse Battle Test":
    st.header("🏢 Task 2: Warehouse Contextual Analysis")
    st.markdown("""
    **The Goal:** Use spatial heuristics to distinguish between structural columns 
    and truss components that have identical geometry.
    """)

    # 1. Setup paths and file selection
    PART_B_DIR = r"./data/Part B/Structures" # Ensure this matches your folder structure
    if not os.path.exists(PART_B_DIR):
        st.error(f"Directory not found: {PART_B_DIR}")
    else:
        warehouse_files = [f for f in os.listdir(PART_B_DIR) if f.endswith('.obj')]
        selected_warehouse = st.selectbox("Select Industrial Warehouse Assembly", warehouse_files)

        if st.button("Run Battle Test Pipeline"):
            full_path = os.path.join(PART_B_DIR, selected_warehouse)
            
            with st.spinner("Analyzing building hierarchy and segmenting mesh..."):
                # --- CALL MODULAR GEOMETRY ENGINE ---
                # We call the 'Manager' function we moved to geometry.py
                try:
                    import core.geometry as geom_engine
                    results = geom_engine.process_full_warehouse(full_path, model, CLASSES)
                except Exception as e:
                    st.error(f"Error in Geometry Engine: {e}")
                    results = []

            if results:
                st.success(f"Successfully segmented and classified {len(results)} elements.")
                
                # --- UI LAYOUT FOR RESULTS ---
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    # Visualization of the Battle Test
                    plotter = pv.Plotter(window_size=[800, 600])
                    plotter.background_color = "#ffffff"
                    
                    for item in results:
                        # item['mesh'] is the trimesh, item['color'] is the contextual color
                        plotter.add_mesh(pv.wrap(item['mesh']), color=item['color'], show_edges=True)
                    
                    plotter.view_isometric()
                    stpyvista(plotter, key=f"warehouse_{selected_warehouse}")

                with c2:
                    st.subheader("Inventory Stats")
                    # Count occurrences of each label
                    from collections import Counter
                    counts = Counter([item['label'] for item in results])
                    
                    for label, count in counts.items():
                        st.metric(label, count)
                    
                    st.info("""
                    **Legend:**
                    - 🟦 **Blue:** Structural Columns (Grounded)
                    - 🟦 **Cyan:** Truss Studs (Elevated)
                    - 🟥 **Red:** Beams
                    - 🟩 **Green:** Walls
                    - 🟨 **Yellow:** Slabs
                    """)
                    
                    st.download_button(
                        label="Download Classification Report",
                        data=pd.DataFrame([{"Label": i['label'], "Elevation": i['elevation']} for i in results]).to_csv(),
                        file_name=f"report_{selected_warehouse}.csv",
                        mime="text/csv"
                    )