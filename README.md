# Salary prediction project

The Python model is trained with scikit-learn. To train it with WEKA instead,
export the dataset as ARFF and use WEKA Explorer:

1. From this project folder, run
   `.venv\Scripts\python.exe export_to_weka.py`
   in PowerShell. This creates `salarydata.arff` from `salarydata.csv`.
2. Open WEKA and choose **Explorer**.
3. In **Preprocess**, select **Open file...** and open `salarydata.arff`.
4. In **Classify**, set the class attribute to `salary`.
5. Select a regressor such as **trees > RandomForest** or
   **functions > LinearRegression**. Choose **10-fold cross-validation** and
   click **Start** to train and evaluate it.
6. Review the evaluation metrics in the output, or use **Visualize classifier
   errors** to inspect predictions.

The exporter marks blank CSV cells as missing values (`?`) and writes `salary`
as the final attribute, as expected for a prediction target. You can also choose
**percentage split** in WEKA if you prefer a train/test split.
