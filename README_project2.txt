Project 2: Image classification using different classification methods
Course: Medical Image Analysis (8DC00)
Assignment 2
Group 8

## Contributors
- Luuk Brouwer
- Karlijn Cornelissen
- Marit Janssen
- Manouk Mels

##Project Overview
This project is a Python-based tool for image classification using various techniques, including:
- Linear registration
- Logistic regression
- k-Nearest Neighbors (k-NN)
- Neural networks
The goal is to classify medical images using different classification methods to evaluate and compare their performance.


## Requirements
To run this project, you will need the following software and packages:
- Python 3.8 or later
- Conda (Anaconda or Miniconda)
- Visual Studio Code (not necessary, but preferred)
- Python packages: 
    - Ipython.display
    - sys
    - pandas
    - matplotlib.pyplot
To install required packages using conda:
conda install ipython pandas matplotlib

## Folder structure
In order for the code to run properly, a certain folder structure is expected. Since our work is handed in as a zip file under the name 'MIA_group_8_project2', let's suppose that when you unzip it, you will have a folder named 'MIA_group_8_project2'. This main folder should contain five items:

1. a folder named 'data'
2. a folder named 'code'
3. a folder named 'assets'
5. this markdown file named 'READ_ME.md'

#The code folder contains multiple  files, each serving a specific purpose:
-cad_PCA.py: Handles PCA (Principal Component Analysis) for dimensionality reduction.
-cad_project.py: Implements core project functionalities, including nuclei area measurement and classification.
-cad_tests.py: Contains various test functions for validating the model performance.
-cad_util.py: Utility functions for the CAD project.
-cad.py: General helper functions.
-registration_util.py: Contains helper functions for image registration.
-registration.py: Handles linear image registration.
-task_5.py: Code for handling k-Nearest Neighbors (k-NN) classification.
-2.5_CAD_project.ipynb: Main Jupyter notebook for the project
-task4.ipynb: Jupyter notebook for neural network classification


#Notebooks
-2.5_CAD_project.ipynb: This is the main notebook for the project, where different Python files are integrated and executed.
    -Linear regression for nuclei area measurement uses cad_project.py and the function nuclei_measurement().
    -Logistic regression for nuclei classification uses the nuclei_classification() function from cad_project.py. IPython.display's display() and clear_output() are also used in this notebook.
    -k-NN classification is performed using task_5.py, which is imported and executed using the main() function.
-task4.ipynb: This notebook is used for experimenting with a neural network for nuclei classification.
    -The script imports modules such as cad_project.py, pandas, matplotlib.pyplot, and cad_tests.py.
    -The notebook explores different combinations of learning rates and batch sizes, and you can visualize results directly in the notebook after execution.

##Usage
To run the project, open the 2.5_CAD_project.ipynb notebook in Jupyter and execute the cells in sequence. The notebook integrates several components from different Python files and provides visual output for the classification tasks.
For neural network experiments, use task4.ipynb, which contains different test cases for adjusting the learning rate and batch size. Results will display directly within the notebook upon execution.