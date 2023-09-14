# CryoSegNet: Cryo-EM protein particles picking with foundational image segmentation model.

CryoSegNet is a method using foundational image segmentation model for picking protein particles in cryo-EM micrographs. It is trained on 22 different protein types including membrane protein, signaling protein, transport protein, viral protein, ribosomes, etc. It uses UNET and SAM's automatic mask generator for predicting the protein particles coordinates from the cryo-EM micrographs and generates output in the form of .star file which can be used in popular tools like RELION and CryoSPARC for generating 3D density maps. It has achieved the state-of-the-art performance and has surpassed the popular AI pickers like crYOLO and Topaz.

-----

## Overview
Figure below demonstrates the overview of particle picking process used by CryoSegNet.

![Alt text](<assets/General Outline.jpeg>)

## Installation
```
### Clone project
```
git clone https://github.com/jianlin-cheng/CryoSegNet.git
cd CryoSegNet/
```
### Download trained models
```
curl https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz --output pretrained_models.tar.gz
tar -xvf pretrained_models.tar.gz
rm pretrained_models.tar.gz
```
### Download training data (if required)
```
curl https://calla.rnet.missouri.edu/CryoSegNet/train_dataset.tar.gz --output train_dataset.tar.gz
tar -xvf train_dataset.tar.gz
rm train_dataset.tar.gz
```
### Download test data
```
curl https://calla.rnet.missouri.edu/CryoSegNet/test_dataset.tar.gz --output test_dataset.tar.gz
tar -xvf test_dataset.tar.gz
rm test_dataset.tar.gz
```
### Create conda environment
```
conda env create -f environment.yml
conda activate cryosegnet
```

## Training Data Statistics
| **SN** | **EMPAIR ID** | **Protein Type**        | **Image Size**          | **Total Structure Weight (kDa)** | **Training Images** | **Validation Images** | **Total Images** |
| ------ | ------------- | ----------------------- | ----------------------- | -------------------------------- | ------------------- | --------------------- | ---------------- | 
| 1      | 10005         | TRPV1 Transport Protein | (3710,3710)             | 272.97                           | 23                  | 6                     | 29               |
| 2      | 10059         | TRPV1 Transport Protein | (3838,3710)             | 317.88                           | 232                 | 59                    | 291              |
| 3      | 10075         | Bacteriophage MS2       | (4096,4096)             | 1000*                            | 239                 | 60                    | 299              |
| 4      | 10077         | Ribosome (70S)          | (4096,4096)             | 2198.78                          | 240                 | 60                    | 300              |
| 5      | 10096         | Viral Protein           | (3838,3710)             | 150*                             | 240                 | 60                    | 300              |
| 6      | 10184         | Aldolase                | (3838,3710)             | 150*                             | 236                 | 60                    | 296              |
| 7      | 10240         | Lipid Transport Protein | (3838,3710)             | 171.72                           | 239                 | 60                    | 299              |
| 8      | 10289         | Transport Protein       | (3710,3838)             | 361.39                           | 240                 | 60                    | 300              |
| 9      | 10291         | Transport Protein       | (3710,3838)             | 361.39                           | 240                 | 60                    | 300              |
| 10     | 10387         | Viral Protein           | (3710,3838)             | 185.87                           | 239                 | 60                    | 299              |
| 11     | 10406         | Ribosome (70S)          | (3838,3710)             | 632.89                           | 191                 | 48                    | 139              |
| 12     | 10444         | Membrane Protein        | (5760,4092)             | 295.89                           | 236                 | 60                    | 296              |
| 13     | 10526         | Ribosome (50S)          | (7676,7420)             | 1085.81                          | 176                 | 44                    | 220              |
| 14     | 10590         | TRPV1 Transport Protein | (3710,3838)             | 1000*                            | 236                 | 60                    | 296              |
| 15     | 10671         | Signaling Protein       | (5760,4092)             | 77.14                            | 238                 | 60                    | 298              |
| 16     | 10737         | Membrane Protein        | (5760,4092)             | 155.83                           | 233                 | 59                    | 292              |
| 17     | 10760         | Membrane Protein        | (3838,3710)             | 321.69                           | 240                 | 60                    | 300              |
| 18     | 10816         | Transport Protein       | (7676,7420)             | 166.62                           | 240                 | 60                    | 300              |
| 19     | 10852         | Signaling Protein       | (5760,4092)             | 157.81                           | 274                 | 69                    | 343              |
| 20     | 11051         | Transcription/DNA/RNA   | (3838,3710)             | 357.31                           | 240                 | 60                    | 300              |
| 21     | 11057         | Hydrolase               | (5760,4092)             | 149.43                           | 236                 | 59                    | 295              |
| 22     | 11183         | Signaling Protein       | (5760,4092)             | 139.36                           | 240                 | 60                    | 300              |
| Total  |               |                         |                         |                                  | 4,948               | 1,244                 | 6,192            |

## Test Data Statistics
| **SN** | **EMPAIR ID** | **Protein Type**        | **Image Size**          | **Total Structure Weight (kDa)** | **Total Images** |
| ------ | ------------- | ----------------------- | ----------------------- | -------------------------------- | ---------------- |
| 1      | 10017         | β -galactosidase        | (4096,4096)             | 450*                             | 84               |   
| 2      | 10028         | Ribosome (80S)          | (4096,4096)             | 2135.89                          | 300              |   
| 3      | 10081         | Transport Protein       | (3710,3838)             | 298.57                           | 300              |   
| 4      | 10093         | Membrane Protein        | (3838,3710)             | 779.4                            | 295              |   
| 5      | 10345         | Signaling Protein       | (3838,3710)             | 244.68                           | 295              |   
| 6      | 10532         | Viral Protein           | (4096,4096)             | 191.76                           | 300              |   
| 7      | 11056         | Transport Protein       | (5760,4092)             | 88.94                            | 305              |   
| Total  |               |                         |                         |                                  | 1879             |   

## Training (if required)



## Prediction
1. To predict protein function with protein structures in the PDB format as input (note: protein sequences are automatically extracted from the PDB files in the input pdb path).
```
    python predict.py --data-path path_to_store_intermediate_files --ontology GO_function_category --input-type pdb --pdb-path data/alphafold --output output_file --cut-off probability_threshold
```

2. To predict protein function with protein sequences in the fasta format and protein structures in the PDB format as input: 
```
    python predict.py --data-path path_to_store_intermediate_files --ontology GO_function_category --input-type fasta --pdb-path data/alphafold --fasta-path path_to_a_fasta_file --output result.txt --cut-off probability_threshold
```

3. Full prediction command: 
```
Predict protein functions with TransFun

optional arguments:
  -h, --help            Help message
  --data-path DATA_PATH
                        Path to store intermediate data files
  --ontology ONTOLOGY   GO function category: cellular_component, molecular_function, biological_process
  --no-cuda NO_CUDA     Disables CUDA training
  --batch-size BATCH_SIZE
                        Batch size
  --input-type {fasta,pdb}
                        Input data type: fasta file or PDB files
  --fasta-path FASTA_PATH
                        Path to a fasta containing one or more protein sequences
  --pdb-path PDB_PATH   Path to the directory of one or more protein structure files in the PDB format
  --cut-off CUT_OFF     Cut-off probability threshold to report function
  --output OUTPUT       A file to save output. All the predictions are stored in this file
  
```

4. An example of predicting cellular component of some proteins: 
```
    python predict.py --data-path data --ontology cellular_component --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

5. An example of predicting molecular function of some proteins: 
```
    python predict.py --data-path data --ontology molecular_function --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

-----

## Rights and Permissions
Open Access \
This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.



## Cite this work
If you use the code or data associated with this research work or otherwise find this data useful, please cite: \
@article {Gyawali2023, \
	author = {Gyawali, Rajan and Dhakal, Ashwin and Wang, Liguo and Cheng, Jianlin}, \
	title = {A large expert-curated cryo-EM image dataset for machine learning protein particle picking}, \
}