# MorphDG-Colon

## Main Characteristics

- Nine-class colorectal histopathology classification.
- ImageNet-initialized EfficientNet-B0 image encoder.
- RandStainNA-based stain-robust training.
- Explicit morphology feature representation.
- Continuous morphology-domain context.
- Reliability-aware residual correction.
- Frozen image-based prediction anchor.
- Evaluation across source-domain and external datasets.

## Tissue Classes

The model uses the following nine-class order:

1. ADI - Adipose tissue
2. BACK - Background
3. DEB - Debris
4. LYM - Lymphocytes
5. MUC - Mucus
6. MUS - Smooth muscle
7. NORM - Normal colon mucosa
8. STR - Cancer-associated stroma
9. TUM - Colorectal adenocarcinoma epithelium

## Evaluation Datasets

### NCT-CRC-HE-100K

Used as the source-development dataset for model training and internal evaluation.

### CRC-VAL-HE-7K

Used for descriptive external comparison. This dataset informed MorphDG-Colon V2 development and is therefore not treated as an untouched independent validation set.

### LC25000 Colon Subset

Used as an independent task-compatible external evaluation restricted to the compatible NORM and TUM categories.

LC25000 is not treated as a full nine-class external validation dataset.

## Frozen Model

The primary reported model is the predefined **seed-42 MorphDG-Colon V2 checkpoint**.

SHA256:

```text
0b5e864aeae12e4093ec9e7ed08411d597d933700396a13570aa3beaa1b46bcd
