# MorphDG-Colon V2 Inference Requirements

This document describes the minimum frozen resources and input requirements
needed to reproduce MorphDG-Colon V2 inference.

MorphDG-Colon V2 is **not an image-only classifier**. Faithful inference
requires the image pathway together with the frozen morphology extraction,
scaling, and morphology-domain context resources released in this repository.

---

## 1. Primary Frozen Model

The primary manuscript model is:

```text
model/MorphDG_Colon_V2_Final_Seed42.pth

SHA256:
0b5e864aeae12e4093ec9e7ed08411d597d933700396a13570aa3beaa1b46bcd

Model Source Files:
src/model/stage10d_morphdg_colon_v2_model.py
src/model/stage8b1_morphdg_colon_model.py

Histopathology Image Input:
Input size: 224 x 224 pixels
Color mode: RGB
Normalization: ImageNet normalization

Output Tissue Classes:
0  ADI   Adipose tissue
1  BACK  Background
2  DEB   Debris
3  LYM   Lymphocytes
4  MUC   Mucus
5  MUS   Smooth muscle
6  NORM  Normal colon mucosa
7  STR   Cancer-associated stroma
8  TUM   Colorectal adenocarcinoma epithelium


Morphology Feature Input:
The frozen morphology extractor is: src/inference/stage9p2_reconstructed_morphology_extractor_v1.py

Its corresponding specification is: src/inference/resources/stage9p2_reconstructed_morphology_extractor_spec_v1.json

Frozen Morphology Scaler: src/inference/resources/stage7a3_training_only_hybrid_scaler.json

Morphology-Domain Context: src/inference/resources/stage7b3_final_label_free_domain_mapper.json

Frozen Runtime Contract: src/inference/resources/stage9p2_crcval_external_evaluator_contract_v1.json

Model Input Pathways:
RGB histopathology image
        |
        v
Image encoder and image prediction anchor
        |
        +-----------------------------+
                                      |
10-D morphology representation        |
        |                             |
        v                             |
Frozen morphology scaler              |
        |                             |
        v                             |
9-D morphology-domain context         |
        |                             |
        +------> Context fusion <------+
                     |
                     v
        Reliability-controlled
           residual correction
                     |
                     v
             Final nine-class
               prediction

Required Repository Assets:
model/
└── MorphDG_Colon_V2_Final_Seed42.pth

src/
├── model/
│   ├── stage10d_morphdg_colon_v2_model.py
│   └── stage8b1_morphdg_colon_model.py
│
└── inference/
    ├── stage9p2_reconstructed_morphology_extractor_v1.py
    │
    └── resources/
        ├── stage9p2_reconstructed_morphology_extractor_spec_v1.json
        ├── stage7a3_training_only_hybrid_scaler.json
        ├── stage7b3_final_label_free_domain_mapper.json
        └── stage9p2_crcval_external_evaluator_contract_v1.json

Software Environment: requirements.txt

Dataset Scope:
NCT-CRC-HE-100K
CRC-VAL-HE-7K
LC25000

Integrity Verification: metadata/model_checksums.txt




