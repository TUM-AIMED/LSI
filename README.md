# Laplace Sample Information (LSI)

This project introduces **Laplace Sample Information (LSI)**, a novel measure of sample informativeness grounded in information theory. LSI leverages a Bayesian approximation to the weight posterior and the Kullback-Leibler (KL) divergence to quantify the unique contribution of individual samples to the parameters of a neural network.

![LSI Overview](Readme_data/concept.svg)

LSI fits a Bayesian posterior to the model using the Laplace approximation. By performing Leave-One-Out (LOO) retraining of the last layer, we can probe the features of the model and compute the Kullback-Leibler (KL) divergence between the posterior distributions with and without a specific data point. This KL divergence quantifies the informativeness of the data point.

Theoretically, LSI approximates an upper bound of the pointwise conditional mutual information between a data point and the weights of a neural network. The formula for LSI is given as:

$$
\text{LSI}(x_i) \approx D_{\text{KL}}\left( p(w \mid \mathcal{D}) \, \| \, p(w \mid \mathcal{D} \setminus \{x_i\}) \right)
$$
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