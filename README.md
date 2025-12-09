# Breast-Cancer-Machine-Learning-Analysis

This project performs a complete machine learning workflow on the Breast Cancer Wisconsin dataset using Python and scikit-learn. It covers dataset exploration, supervised learning, unsupervised learning, visualizations, and model evaluation.

---

## Features

### 1. Dataset Exploration
- Loads the breast cancer dataset from scikit-learn.
- Prints the number of records, attributes, and classes.

### 2. Scatterplot Visualization
- Plots mean area vs area error.
- Colors the points based on class labels.

### 3. Train–Test Split
- Uses 80% training, 20% testing.
- Records are shuffled with a fixed random seed.

### 4. Supervised Learning Models
Trains and evaluates the following:
- Decision Tree (depths 1–5)
- Logistic Regression (liblinear solver)
- Neural Network (MLP, 1 hidden layer with 5 neurons)

Selects and prints the best model based on test accuracy.

### 5. Confusion Matrix
- Computes the confusion matrix for the best model.
- Visualizes it using ConfusionMatrixDisplay.

### 6. Unsupervised Learning – K-Means
- Tests k = 2 to 30.
- Determines the optimal number of clusters with the Davies–Bouldin Index.

### 7. PCA-Based Clustering Visualization
- Reduces the dataset to 2 principal components.
- Visualizes K-Means clustering (k=4) in 2D.

---

## Technologies Used
- Python 3
- NumPy
- Matplotlib
- scikit-learn

---

## How to Run

1. Clone the repository:
git clone https://github.com/<your-username>/breast-cancer-ml-analysis.git

2. Install required libraries:
pip install numpy matplotlib scikit-learn

3. Run the script:
python main.py

---

## Output Includes

* Dataset summary
* Scatter plot
* Best classifier and accuracy
* Confusion matrix + visualization
* Optimal K via Davies–Bouldin score
* PCA visualization of clusters

---

## Project Structure

.
├── main.py

└── README.md

---

## License

This project is for educational purposes only.
