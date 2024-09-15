This Colab notebook analyzes the Titanic dataset to predict passenger survival using a neural network.

**Exploratory data analysis (EDA)** to understand the dataset. This includes:

Loading the dataset
Checking the shape of the dataset
Looking at the first few rows of the dataset
Identifying the columns in the dataset
Checking for unique values
Getting descriptive statistics
Checking data types and missing values
Next, the notebook focuses on data preparation:

**Handling missing values** by 
filling in missing ages with the mean age and missing embarked values with 'C'.
Removing the 'Cabin' column due to a large number of missing values.

Creating an age group feature.
Encoding categorical features (Sex, Pclass, Embarked, and age_group) using LabelEncoder.
The notebook then explores feature understanding through visualizations:

**Visualizing the Data**
Histograms and bar plots are used to visualize the distribution of passengers based on sex, class, embarked town, and survival rate.
Count plots show the number of deaths and survivors based on class and sex.
Pie charts illustrate the survival ratio based on sex and class.
Bar plots show the survival rate by age.

**Model training:**

Features ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked') and 
target variable ('Survived') are selected.

The dataset is split into training and testing sets using train_test_split.
Numerical features are normalized using StandardScaler.
A neural network model is built using TensorFlow with dense layers, dropout, and ReLU activation function.
The model is compiled using the Adam optimizer, CategoricalCrossentropy loss, and CategoricalAccuracy metric.
Early stopping is implemented to prevent overfitting.
The model is trained and evaluated, and the training history is visualized.

The notebook concludes with visualizing the training history to analyze the model's performance over epochs.
