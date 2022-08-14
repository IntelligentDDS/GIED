# GIED

This repo contains the simulation environment dataset and source code showpieces for the paper "Graph based Incident Extraction and Diagnosis in Large-Scale Online Systems" (ASE'22).

## Project Structure

- `./data` contains the simulation environment dataset used in the paper. The real-world dataset from the company cannot be provided here yet due to the confidentiality policy of the company.
- `./showpieces` contains ipython notebooks which run some code pieces of GIED to show how each step is performed. Their order is as follow:
    - `anomaly_detection_and_issue_extraction.ipynb` contains code pieces for KPI anomaly detection and issue extraction.
    - `data_labelling.ipynb` contains code pieces for data labelling using fault injection records.
    - `feature_engineering.ipynb` contains code pieces for feature engineering.
    - `SpatioDevNetPackage` contains the implemented graph neural networks based model.
    - `incident_detection.ipynb` contains code pieces for the graph neural networks based model training and testing for incident detection.
    - `incident_diagnosis.ipynb` contains code pieces for the root cause service localization.