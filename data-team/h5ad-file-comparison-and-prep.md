# Template vs Prediction H5AD Files & VCC Prep Workflow

## Overview

This document summarizes our analysis of the differences between template and prediction H5AD files, and the VCC (Virtual Cell Challenge) preparation workflow using cell-eval.

## File Structure Analysis

### Basic Information

  * **Template File**: `competition_support_set/competition_val_template.h5ad`
  * **Prediction File**: `competition/prediction.h5ad`
  * **Both Files**: 98,927 cells x 18,080 genes

### HDF5 Internal Structure Comparison

Both files have identical HDF5 internal structure:

  * Same folder hierarchy (X, obs, var, obsm, varm, layers, uns)
  * Same metadata structure
  * Same categorical encodings for cell types and perturbations
  * Identical dimensions and data types

**Key finding**: The files are structurally identical but contain different data content.

### Main Differences: X Matrix Content

#### Statistical Comparison

| Metric                 | Template    | Prediction |
| :--------------------- | :---------- | :--------- |
| **Non-zero values**    | 48.40%      | 51.27%     |
| **Value range**        | [0, 7.60]   | [0, 12.37] |
| **Mean expression**    | 0.7833      | 0.7808     |
| **Standard deviation** | 0.9849      | 0.9891     |



#### Correlation

0.849 (strong but not identical)

### Key Insights

1.  **Template** = Input data for the model
2.  **Prediction** = STATE model output after processing
3.  **Different content, same structure**: Model preserves metadata while transforming expression values
4.  **Biological significance**: Represents input-output relationship of perturbation prediction

## Cell-Eval VCC Prep Workflow

### What is cell-eval?

Cell-eval is a comprehensive evaluation framework for single-cell perturbation prediction models, specifically designed for the Virtual Cell Challenge.

### Source Code Location

`cell-eval-main/src/cell_eval/`

  * `main.py`      # Main entry point
  * `cli/`         # VCC prep functionality # Evaluation runner # Scoring system
      * `prep.py`
      * `run.py`
      * `score.py`
      * `baseline.py`  # Baseline comparisons

### VCC Prep Command Analysis

```
uv tool run--from git+https://github.com/ArcInstitute/cell-eval@main \
cell-eval prep \
-i competition/prediction.h5ad \
-g competition_support_set/gene_names.csv
```



**Command Breakdown**:

  * **uv tool run**: Uses uv package manager to run tools
  * **--from git+...**: Installs and runs directly from GitHub
  * **cell-eval prep**: Runs the prep subcommand
  * **-i**: Input H5AD file (your predictions)
  * **-g**: Gene names CSV file for standardization

### Core Functions

#### 1\. `run_prep()` Function

Located in `_prep.py: 263`, this function:

  * Reads input AnnData file
  * Loads expected gene list from CSV
  * Calls `strip_anndata()` for core processing

#### 2\. `strip_anndata()` Function

The main data processing function that:

  * **Data Validation**:
      * Checks perturbation column exists
      * Validates gene list matches expected format
      * Ensures data dimensions are correct
      * Verifies negative control presence
  * **Data Standardization**:
      * Reorders genes to match expected list
      * Converts to sparse matrix format (CSR)
      * Applies normlog transformation
      * Sets appropriate float precision (32/64-bit)
  * **Metadata Simplification**:
      * Keeps only essential observation data
      * Strips unnecessary annotations
      * Standardizes column names
  * **Output Generation**:
      * Creates minimal AnnData object
      * Compresses using zstd algorithm
      * Packages into .vcc tar format
      * Adds watermark file for validation

### VCC Prep Workflow Steps

1.  **Input Validation**
    ```python
    # Check required columns exist
    if pert_col not in adata.obs:
        raise ValueError (f"Missing perturbation column: (pert_col '")
    ```
2.  **Gene Standardization**
    ```python
    # Reorder genes to match expected list
    if adata.var_names.tolist() ! genelist:
        adata adata(:, np.array(genelist)]
    ```
3.  **Data Type Optimization**
    ```python
    # Set precision based on encoding parameter
    dtype np.float64 if encoding ==64 else np.float32
    new_x csr_matrix(adata.X.astype (dtype))
    ```
4.  **Metadata Minimization**
    ```python
    # Keep only essential metadata
    new obs pd. DataFrame({
        output_pert_col: adata.obs [pert_col].values
    }, index-np.arange(adata.shape[0]).astype(str))
    ```
5.  **Normalization**
    ```python
    # Apply normlog transformation
    _convert_to_normlog (minimal, allow_discrete=allow_discrete)
    ```
6.  **Compression & Packaging**
    ```bash
    # Compress with zstd
    zstd -TO-frm pred.h5ad
    # Package into tar
    tar -cf output.vcc pred.h5ad.zst watermark.txt
    ```

### Expected Output

The prep command generates a .vcc file containing:

  * **pred.h5ad.zst**: Compressed, standardized prediction data
  * **watermark.txt**: Contains "vcc-prep" validation marker

### Virtual Cell Challenge Context

#### Purpose

VCC is a competition for predicting cellular responses to genetic perturbations at single-cell resolution.

#### Why VCC-Specific Prep?

1.  **Standardization**: Ensures all submissions use consistent format
2.  **Validation**: Checks data integrity before submission
3.  **Optimization**: Reduces file size for efficient processing
4.  **Compatibility**: Guarantees evaluation framework compatibility

#### Official Documentation Quote

> "This will strip the anndata to bare essentials, compress it, adjust naming conventions, and ensure compatibility with the evaluation framework. This step is optional for downstream usage, but recommended for optimal performance and compatibility."

### Key Takeaways

1.  **Template vs Prediction**: Same structure, different content - represents model input/output
2.  **VCC Prep is Essential**: Required for Virtual Cell Challenge submissions
3.  **cell-eval is Purpose-Built**: Specifically designed for perturbation prediction evaluation
4.  **Data Pipeline**: Raw predictions VCC prep Submission-ready format
5.  **Quality Assurance**: Multiple validation steps ensure submission compatibility

### Next Steps

After running VCC prep, you'll have a submission-ready .vcc file that can be uploaded directly to the Virtual Cell Challenge leaderboard for evaluation and scoring.
