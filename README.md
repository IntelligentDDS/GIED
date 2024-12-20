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

## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{DBLP:conf/kbse/HeCLYCYL22,
  author       = {Zilong He and
                  Pengfei Chen and
                  Yu Luo and
                  Qiuyu Yan and
                  Hongyang Chen and
                  Guangba Yu and
                  Fangyuan Li},
  title        = {Graph based Incident Extraction and Diagnosis in Large-Scale Online
                  Systems},
  booktitle    = {37th {IEEE/ACM} International Conference on Automated Software Engineering,
                  {ASE} 2022, Rochester, MI, USA, October 10-14, 2022},
  pages        = {48:1--48:13},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3551349.3556904},
  doi          = {10.1145/3551349.3556904},
  timestamp    = {Thu, 22 Jun 2023 07:45:51 +0200},
  biburl       = {https://dblp.org/rec/conf/kbse/HeCLYCYL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
