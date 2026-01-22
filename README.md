# Maestro_repo

This project addresses the challenge of identifying structural elements within complex BIM (Building Information Modeling) assemblies. Using a modular Python architecture, the system extracts scale-invariant geometric features from 3D meshes to classify elements into four primary categories: Beams, Columns, Walls, and Slabs.

Beyond simple geometry, the pipeline implements spatial heuristics to solve "Real World" contextual ambiguity—distinguishing between primary structural supports and local components (like roof trusses) that share identical geometric profiles.

## Contents
- ```core/```: Contains utility functions for geometry processing, model manipulation, and visualization.

- ```geometry.py```: Contains functions for geometry-related operations.

- ```model_utils.py```: Provides utility functions to load, manage, and interact with machine learning models.

- ```visualization.py```: Includes functions for visualizing model outputs and data.

- ```models/```: Contains machine learning models used by the app (specific models are yet to be described).

- ```app.py```: The main Python script that runs the Streamlit app. This is the entry point for the interactive web application.

- ```notebook.ipynb```: A Jupyter notebook for exploring the models, performing analysis, and testing different parts of the code.

- ```requirements.txt```: A file listing the necessary Python dependencies required to run the app and other scripts.

## Installation Instructions
### 1. Clone the repository
```bash
git clone https://github.com/ivan-quirante/Maestro_repo.git
cd Maestro_repo
```
### 2. Add data folder
Add the data folder with Part A and Part B to the repository.

### 4. Set up the virtual environment
```bash
python -m venv venv

# On macOS/Linux
source .env/bin/activate
# On Windows
venv/Scripts/activate
```

### 5. Install the required dependencies
Once the virtual environment is activated, install the dependencies from
```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit App
```bash
streamlit run app.py
```

