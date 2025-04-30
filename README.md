# ECM3401 Coursework: Learning from Mislabelled Data

This project develops a novel Random Forest based technique for handling noisy labels in classification tasks. It includes experimentation with iterative reweighting, relabelling, bootstrapping, and visual performance analysis on synthetic and real datasets.

## Installation & Setup

1. Clone the repository or extract the contents of the ZIP file.
2. Create a conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate Dissertation
    ```

## How to Run

- To reproduce experiments relating to the final training scheme, run the notbooks in `/Final Design/`.
- To reproduce experiments relating to the initial training scheme, run the notbooks in `/Initial Design/`.
- To test relabelling algorithms, use `mislabelling.py` directly or via related notebooks.
Each notebook has a brief description of the experiments it runs

## Datasets

- Custom Gaussian Mixture Models (`gmm5train.txt`, `gmm5test.txt`)
- Additional benchmark datasets included in Scikit-learn (breast cancer, digits, and wine)

---

For questions or contributions, please contact the je497@exeter.ac.uk.