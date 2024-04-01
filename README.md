# Penguins Classification

### **Project Motive:**
Our goal is to create a model that can help predict a species of a penguin based on physical attributes, then we can use that model to help researchers classify penguins in the field, instead of needing an experienced biologist.

### **Columns of DataSet:**
<pre>
<b>species</b> : penguin species (Chinstrap, Adélie, or Gentoo)
<b>culmen_length_mm</b> : culmen length (mm)
<b>culmen_depth_mm</b> : culmen depth (mm)
<b>flipper_length_mm</b> : flipper length (mm)
<b>body_mass_g</b> : body mass (g)
<b>island</b> : island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
<b>sex</b> : penguin sex
</pre>
---
**Sample DataSet**

<img width="789" alt="Screenshot 2024-04-01 at 11 02 14 AM" src="https://github.com/TRGanesh/Penguins_Classification_Project/assets/117368449/083e1889-35f7-4630-bd14-2af542b32a0a">

---
**Dependencies for Exploratory Data Analysis**
<pre>
<b>Pandas</b>
<b>Numpy</b>
<b>Matplotlib</b>
<b>Seaborn</b>
<b>Plotly</b>
<b>Statistics</b>
</pre>
---
**Data Analysis:**
- Plotted CountPlots for Categorical Columns such as Species, Island
- Plotted a HeatMap which shows Correlation between the Numerical Columns

<img width="579" alt="Screenshot 2024-04-01 at 11 08 51 AM" src="https://github.com/TRGanesh/Penguins_Classification_Project/assets/117368449/7f524538-e835-4752-86ae-a0c4b7f47ea4">

- Scatter plots

<img width="821" alt="Screenshot 2024-04-01 at 11 11 38 AM" src="https://github.com/TRGanesh/Penguins_Classification_Project/assets/117368449/bf971598-4d78-4f76-a719-f0e5de2f8dae">

---
**Dependencies for Machine learning from Scikit-Learn**
1. **Train-Test Split :** Data is split into a Training Set (for Model Training) and a Test Set (for Model Evaluation), with a Common Split Ratio such as 70/30 or 80/20.
2. **GridSearchCV :** Automates the Parameter Selection and Cross-Validation, simplifying the Optimization Process
3. **Randomized Search CV :** It Randomly selects Combinations of Hyperparameters from Predefined Ranges and Evaluates their Performance using Cross-Validation.  
4. **One-Hot Encoder :** Used for Categorical Variable Encoding. Transforms Categorical Features into a Binary Array, with each column representing a Unique category.
5. **StandardScaler :** Standardizes Features by making Mean to 0 & Standard Deviation to 1. Making them Comparable across Different Scales.
6. **Pickle :** Used to save the Machine Learning files, such as StandardScaler, OneHotEncoder..
7. **Plot Tree :** Generates a Graphical Representation of the Decision Tree Structure.
---
**Models used are** 
<pre>
<b>Support Vector Classifier</b>
- It works by finding the Hyperplane that best separates different classes in a high-dimensional space.
- SVC aims to Maximize the Margin between the classes, making it robust to Outliers.
- It's effective for both Linearly Separable and Non-Linearly Separable data through the use of Kernel Functions.
  
<b>Decision Tree Classifier</b>
- It partitions the feature space into distinct regions and assigns a class label to each region.
- It's interpretable and easy to visualize, making it useful for understanding feature importance.
- Decision trees can handle both numerical and categorical data.
- However, they are prone to overfitting, which can be mitigated through techniques like pruning and limiting tree depth.  
</pre>
---
**Classification Metrics used are** 
<pre>
<b>Accuracy Score</b>
- It measures the Proportion of Correctly Predicted Instances out of the Total Instances.
- It is calculated by Dividing the Number of Correct Predictions by the Total Number of Predictions made.
- Accuracy may Not be suitable for ImBalanced Datasets as it does not consider Class Distribution.
  
<b>Confusion Matrix</b>
- It displays the Counts of True Positive, False Positive, True Negative, and False Negative Predictions.
- It helps in understanding the Model's Accuracy, Precision, Recall, and F1-Score.
- The Diagonal Elements represent Correct Predictions, while Off-Diagonal Elements indicate Misclassifications.
</pre>
---
**Data Preprocessing:**
- Splitted the Dataset into Training & Testing parts
- Encoding the Categorical Features
- Scaling both Training & Testing data
---
### Modelling
- Created Instances of the Machine Learning Models using Scikit-Learn functions
- Fitting the Models(having default Parameters) with Scaled & Transformed X_train & Y_train data
- Created Parameter Grids for Machine Learning Models to pass in GridSearchCV & RandomizedSearchCV(For Large Search Space) 
- Also Fitted those Models with Training Data
- Obtained Accuarcy & Confusion Matrix for both Models on Test Data

<img width="891" alt="Screenshot 2024-04-01 at 12 01 44 PM" src="https://github.com/TRGanesh/Penguins_Classification_Project/assets/117368449/6779e824-111c-492d-8d80-d2c389951bb5">

### Trained Decision Tree Structure

<img width="863" alt="Screenshot 2024-04-01 at 12 00 24 PM" src="https://github.com/TRGanesh/Penguins_Classification_Project/assets/117368449/53680db4-8be9-4c8d-a80d-d9d2380f1b73">

- Files such as Scaler, One Hot Encoder, Final Model are Saved using Pickle Module
---
### Streamlit WebApp
