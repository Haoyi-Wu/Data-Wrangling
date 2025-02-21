# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, roc_auc_score, r2_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns



# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    data = data.copy()
    missing_data = data.isnull().sum()
    print(missing_data)

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
        
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            if strategy == 'mean':
                data[col].fillna(data[col].mean(), inplace=True)
            elif strategy == 'median':
                data[col].fillna(data[col].median(), inplace=True)
            elif strategy == 'mode':
                data[col].fillna(data[col].mode()[0], inplace=True)
            
                
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    data.dropna(inplace=True)
    return data
    

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    return data.drop_duplicates()
    pass

# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    data = data.copy()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Use 'minmax' or 'standard'.")
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data
    # TODO: Normalize numerical data using Min-Max or Standard scaling
    pass

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    numeric_data = data.select_dtypes(include=['number'])
    corr_matrix = numeric_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data = data.drop(columns=to_drop)
    return data

    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    pass

# 5. Convert Categorical Data 
def convert_categorical(data):
    """
    Convert categorical variables into numerical format using One-Hot Encoding.
    :param data: pandas DataFrame
    :return: transformed pandas DataFrame
    """
    data = data.copy()
    categorical_cols = data.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # One-hot encoding

    return data
    pass


# ----------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False, threshold = None):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    if threshold is not None:
        # Convert continuous target to binary (0/1)
        target = (target > threshold).astype(int)
    else:
        # Ensure target is categorical
        if target.dtypes != 'int' and target.dtypes != 'object':
            raise ValueError(
                "Target variable must be categorical or specify a threshold to binarize it."
            )
        
    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train, method='standard')
        X_test = normalize_data(X_test, method='standard')
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None


# ----------------------------------------------------
def linear_regression(input_data, target_col, test_size=0.2, scale_data=False, random_state=42):
    """
    Train and evaluate a linear regression model.
    
    Parameters:
    - input_data (pd.DataFrame): The dataset.
    - target_col (str): The column name of the target variable.
    - test_size (float): Proportion of dataset to use for testing.
    - random_state (int): Seed for reproducibility.
    - scale_data (bool): Whether to normalize numerical features.
    Returns:
    - lin_reg: Trained LinearRegression model
    - heat map of R² values for ElasticNet hyperparameters
    - Best ElasticNet Model based on R² score
    """

    # Drop rows with missing target values
    input_data.dropna(inplace=True)

    # Define features and target variables
    x = input_data.drop(columns=[target_col])
    y = input_data[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if scale_data:
     # scale the data if specified
        X_train = normalize_data(X_train, method='standard')
        X_test = normalize_data(X_test, method='standard')

    # Train the model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred_lin = lin_reg.predict(X_test)


    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lin, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression Predictions")
    plt.show()


    # Evaluate the model
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test, y_pred_lin)
    print(f"Root Mean Squared Error: {rmse_lin}")
    print(f"R^2 Score: {r2_lin}")

    # Hyperparameter tuning for ElasticNet
    alphas = np.logspace(-3, 1, 10) 
    l1_ratios = np.linspace(0.1, 1.0, 10)

    results = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
            enet.fit(X_train, y_train)
            y_pred_enet = enet.predict(X_test)
            mse_enet = mean_squared_error(y_test, y_pred_enet)
            rmse_enet = np.sqrt(mse_enet)
            r2_enet = r2_score(y_test, y_pred_enet)
            results.append((alpha, l1_ratio, r2_enet, rmse_enet))
    
    # Convert results to DataFrame for visualization
    results_df = pd.DataFrame(results, columns=["alpha", "l1_ratio", "R2", "RMSE"])

    # Heatmap of R² values for ElasticNet hyperparameters
    pivot_table_r2 = results_df.pivot_table(values="R2", index="alpha", columns="l1_ratio")

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table_r2, annot=True, cmap=plt.cm.Reds)
    plt.title("Heatmap of R² for ElasticNet")
    plt.xlabel("L1 Ratio")
    plt.ylabel("Alpha")
    plt.show()

    # Best ElasticNet Model based on rmse score
    pivot_table_rmse = results_df.pivot_table(values="RMSE", index="alpha", columns="l1_ratio")

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table_rmse, annot=True, cmap=plt.cm.Reds)
    plt.title("Heatmap of RMSE for ElasticNet")
    plt.xlabel("L1 Ratio")
    plt.ylabel("Alpha")
    plt.show()

    # Best ElasticNet Model based on R² and RMSE score
    top_config_r2 = results_df.loc[results_df["R2"].idxmax()]
    print(f"Best ElasticNet Model based on R2: alpha={top_config_r2['alpha']}, l1_ratio={top_config_r2['l1_ratio']}, R²={top_config_r2['R2']}, RMSE={top_config_r2['RMSE']}")

    top_config_rmse = results_df.loc[results_df["RMSE"].idxmin()]
    print(f"Best ElasticNet Model based on RMSE: alpha={top_config_rmse['alpha']}, l1_ratio={top_config_rmse['l1_ratio']}, R²={top_config_rmse['R2']}, RMSE={top_config_rmse['RMSE']}")
    return None


# ----------------------------------------------------
def logistic_regression(input_data, target_col, split_data=True, test_size=0.2, scale_data=False, print_report=True, threshold=0.5):
    """
    Train and evaluate multiple logistic regression models with different solvers and penalties.

    Parameters:
    - input_data (pd.DataFrame): The dataset.
    - target_col (str): The target column for classification.
    - split_data (bool): Whether to split into train/test sets.
    - scale_data (bool): Whether to standardize numerical features.
    - print_report (bool): Whether to print evaluation metrics.
    - threshold (float): Custom probability threshold for classification.

    Returns:
    - logreg_results_df (pd.DataFrame): DataFrame with evaluation metrics for all models.
    - models_dict (dict): Dictionary of trained models with their corresponding solver/penalty.
    """

    # Drop rows with missing target values
    input_data.dropna(inplace=True)

    # Define features and target variables
    X = input_data.drop(columns=[target_col])
    y = input_data[target_col].apply(lambda x: 1 if x > 0 else 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Scale the data if specified
    if scale_data:
        X_train = normalize_data(X_train, method='standard')
        X_test = normalize_data(X_test, method='standard')

    solvers = ["liblinear", "saga"]
    penalties = ["l1", "l2"]
    logreg_results = []
    models_dict = {}
    best_auc = 0
    best_y_prob = None
    best_logreg = None

    for solver in solvers:
        for penalty in penalties:
            if solver == "liblinear" and penalty == "l1":
                continue  

            log_reg = LogisticRegression(penalty=penalty, solver=solver, random_state=42, max_iter=500)
            log_reg.fit(X_train, y_train)

            y_prob = log_reg.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)  

            # Evaluate the model
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            logreg_results.append((solver, penalty, acc, f1, auc_score, auprc))

            if auc_score > best_auc:
                best_auc = auc_score
                best_y_prob = y_prob
                best_logreg = {"Solver": solver, "Penalty": penalty, "Accuracy": acc, "F1 Score": f1, "AUROC": auc_score, "AUPRC": auprc}

            logreg_df = pd.DataFrame(logreg_results, columns=["Solver", "Penalty", "Accuracy", "F1 Score", "AUROC", "AUPRC"])
            logreg_df = logreg_df.sort_values(by="AUROC", ascending=False)
           

            # Store model and performance
            logreg_results.append((solver, penalty, acc, f1, auc_score, auprc))
            models_dict[(solver, penalty)] = log_reg

            # Display coefficients
            print(f"\nLogistic Regression (Solver={solver}, Penalty={penalty})")
            print("Coefficients:", log_reg.coef_)

            if print_report:
                print(f"Accuracy: {acc:.2f}, F1 Score: {f1:.2f}, AUROC: {auc_score:.2f}, AUPRC: {auprc:.2f}")

    # Convert results to DataFrame and sort by AUROC
    logreg_results_df = pd.DataFrame(logreg_results, columns=["Solver", "Penalty", "Accuracy", "F1 Score", "AUROC", "AUPRC"])
    logreg_results_df = logreg_results_df.sort_values(by="AUROC", ascending=False)

    # Display all model performances
    print("\nAll Logistic Regression Model Performances:")
    print(logreg_results_df)

    # Compute ROC and Precision-Recall curves for best model
    fpr, tpr, _ = roc_curve(y_test, best_y_prob)
    precision, recall, _ = precision_recall_curve(y_test, best_y_prob)

    # Plot AUROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUROC={best_logreg['AUROC']:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curve - Logistic Regression")
    plt.legend()
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Logistic Regression (AUPRC={best_logreg['AUPRC']:.2f})", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("AUPRC Curve - Logistic Regression")
    plt.legend()
    plt.show()

    return logreg_results_df, models_dict


# ----------------------------------------------------
def knn_classifier(input_data, target_col, split_data=True, test_size=0.2, scale_data=False, print_report=False, threshold=0.5):
    """
    Train and evaluate a k-NN classifier model with different values of n_neighbors.
    
    Parameters:
    - input_data (pd.DataFrame): The dataset.
    - target_col (str): The target column for classification.
    - split_data (bool): Whether to split into train/test sets.
    - scale_data (bool): Whether to standardize numerical features.
    - print_report (bool): Whether to print evaluation metrics.
    - threshold (float): Custom probability threshold for classification.
    - neighbors_list (list): List of n_neighbors values to evaluate ([1, 5, 10]).

    Returns:
    - knn_results_df (pd.DataFrame): DataFrame containing evaluation metrics for different values of k.
    """

    # Drop rows with missing target values
    input_data.dropna(inplace=True)

    # Define features and target variables
    x = input_data.drop(columns=[target_col])
    y = input_data[target_col].apply(lambda x: 1 if x > 0 else 0)  

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    if scale_data:
     # scale the data if specified
        X_train = normalize_data(X_train, method='standard')
        X_test = normalize_data(X_test, method='standard')
    
    neighbors_list = [1, 5, 10]

    knn_results = []   
    for k in neighbors_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Predict probabilities and apply threshold
        y_knn_prob = knn.predict_proba(X_test)[:, 1]
        y_knn_pred = (y_knn_prob >= threshold).astype(int)

        # Evaluate performance
        acc_knn = accuracy_score(y_test, y_knn_pred)
        f1_knn = f1_score(y_test, y_knn_pred)
        auc_knn = roc_auc_score(y_test, y_knn_prob)
        auprc_knn = average_precision_score(y_test, y_knn_prob)

        # Store results
        knn_results.append((k, acc_knn, f1_knn, auc_knn, auprc_knn))
        print(f"\nk-NN (k={k}) Performance:")
        print(f"Accuracy: {acc_knn:.2f}, F1 Score: {f1_knn:.2f}, AUROC: {auc_knn:.2f}, AUPRC: {auprc_knn:.2f}")

    # Convert results to DataFrame for comparison
    knn_results_df = pd.DataFrame(knn_results, columns=["n_neighbors", "Accuracy", "F1 Score", "AUROC", "AUPRC"])

    # Plot performance metrics
    plt.figure(figsize=(10, 6))
    for metric in ["Accuracy", "F1 Score", "AUROC", "AUPRC"]:
        plt.plot(knn_results_df["n_neighbors"], knn_results_df[metric], marker="o", label=metric)

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Score")
    plt.title("Impact of n_neighbors on k-NN Performance")
    plt.legend()
    plt.grid()
    plt.show()

    return knn_results_df
