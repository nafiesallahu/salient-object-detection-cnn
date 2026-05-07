# Notebook Workflow

These notebooks are thin orchestration layers around the project code. They use normal Python imports for exploration and code inspection, and use `subprocess` only when launching full project scripts such as preprocessing, training, evaluation, visualization, and the demo app.

Run them in this order:

1. `01_data_preprocessing.ipynb`
2. `02_data_exploration.ipynb`
3. `03_training_baseline.ipynb`
4. `04_experiments.ipynb`
5. `05_demo.ipynb`

Open the notebooks from this `notebooks/` folder or from the project root. The shared `notebook_utils.py` helper automatically resolves the project directory.
