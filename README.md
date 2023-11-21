I. Implementation of KMC-CG
Pre-requisites: python with version later than 3.8, yaml, os, MDAnalysis, OpenMSCG and pandas are required to run the cg-mapping code, which is used to map an all-atom trajectory to its corresponding CG trajectory using the given CG mapping.

CG_cluster.py: KMC-CG source code.

arp23.ipynb: A jupyter notebook using Arp2/3 as an example.

CG_mapping.py: A sample code to generate the *.yaml file for mscg.cgmap function, which is used to map an all-atom trajectory to a CG trajectory. One can replace the variable names of the G-actin system to your own customed systems.

Reference:
    Please cite this paper if you think this notebook is useful for your research! Thank you very much!
    https://pubs.acs.org/doi/10.1021/acs.jctc.3c01053
    Jiangbo Wu, Weizhi Xue, and Gregory A. Voth
    K-Means Clustering Coarse-Graining (KMC-CG): A Next Generation Methodology for Determining Optimal Coarse-Grained Mappings of Large Biomolecules
    J. Chem. Theory Comput. 2023
