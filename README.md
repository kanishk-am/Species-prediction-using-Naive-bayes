# Iris Species Predictor using Gaussian Naive Bayes

## Overview

This project implements an Iris Species Predictor using the Gaussian Naive Bayes algorithm. It predicts the species of an Iris flower based on four input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The application accepts user input through the command line, predicts the flower species, displays prediction probabilities, and generates multiple visualizations for better understanding of the prediction.

---

## Features

- Interactive command-line interface
- Gaussian Naive Bayes classification
- Prediction confidence score
- Probability visualization
- Radar chart comparing features
- Feature comparison bar chart
- Prediction summary panel
- Automatic saving of visualization as an image
- Input validation for measurements

---

## Technologies Used

- Python 3
- NumPy
- Matplotlib

---

## Project Structure

```
Iris-Species-Predictor/
│
├── iris_predictor.py
├── README.md
├── iris_prediction.png
└── screenshots/
    ├── input.png
    ├── output.png
    ├── probability_chart.png
    └── visualization.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kanishk-am/Iris-Species-Predictor.git
```

Navigate to the project directory:

```bash
cd Iris-Species-Predictor
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

---

## Running the Project

Execute the Python script:

```bash
python iris_predictor.py
```


---

## Machine Learning Algorithm

The project uses the **Gaussian Naive Bayes** algorithm.

The prediction process consists of:

1. Computing the mean and variance for each feature of every Iris species.
2. Calculating Gaussian probability density for each feature.
3. Computing posterior probabilities.
4. Predicting the species with the highest probability.

---

## Dataset

The project uses the Iris dataset containing three flower species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each sample contains four numerical features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## Output

The application displays:

- Predicted species
- Prediction confidence
- Probability distribution of all classes
- Radar chart
- Feature comparison chart
- Prediction summary panel

The visualization is also saved automatically as:

```
iris_prediction.png
```
## Future Improvements

- GUI using Tkinter or PyQt
- Web application using Flask or Streamlit
- Support for CSV input
- Scikit-learn implementation
- Accuracy evaluation and confusion matrix
- Model comparison with other classifiers

---

## Author

**Kanishk A M**

GitHub: https://github.com/kanishk-am

---
