# Laplace Sample Information (LSI)

This project introduces **Laplace Sample Information (LSI)**, a novel measure of sample informativeness grounded in information theory. LSI leverages a Bayesian approximation to the weight posterior and the Kullback-Leibler (KL) divergence to quantify the unique contribution of individual samples to the parameters of a neural network.

---

## Key Features
- **Sample Informativeness**: LSI identifies typical and atypical samples, detects mislabeled data, and measures class-wise informativeness.
- **Dataset Analysis**: It assesses dataset difficulty and provides insights into data quality and informativeness.
- **Efficiency**: LSI can be computed efficiently using a probe model, enabling scalability to large datasets and architectures.
- **Broad Applicability**: LSI is applicable across supervised and unsupervised tasks, as well as image and text data.

---

## Applications
- Dataset curation and condensation
- Identifying mislabeled or redundant samples
- Measuring class-wise and dataset-level informativeness
- Improving model efficiency and generalization

---

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt