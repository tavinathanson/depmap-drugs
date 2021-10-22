import pandas as pd
from os import path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

DATA_DIR = "data"


def get_processed_data(drugs_as_features=False, impute=False):
    df_collapsed = pd.read_csv(
        path.join(DATA_DIR, "primary-screen-replicate-collapsed-logfold-change.csv")
    )
    df_info = pd.read_csv(
        path.join(DATA_DIR, "primary-screen-replicate-collapsed-treatment-info.csv")
    ).set_index("column_name")

    if not drugs_as_features:
        # Transpose so that each column is a feature (cell line), each row is a sample (drug)
        df_collapsed = (
            df_collapsed.rename(columns={"Unnamed: 0": "drug"}).set_index("drug").T
        )
    else:
        df_collapsed = df_collapsed.rename(
            columns={"Unnamed: 0": "cell_line"}
        ).set_index("cell_line")

    if impute:
        df_collapsed = df_collapsed.fillna(df_collapsed.mean())
    return df_collapsed, df_info


def get_cell_line_info():
    return pd.read_csv(
        path.join(DATA_DIR, "primary-screen-cell-line-info.csv")
    ).set_index("row_name")


def logistic_regression_auc(x_values, y_values):
    """
    Keeping it simple. Not currently worrying about hyperparameter optimization,
    cross-fold validation, etc.
    """
    x_values = StandardScaler().fit_transform(x_values)
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=0.40, random_state=0, stratify=y_values
    )
    model = LogisticRegression(solver="liblinear", penalty="l2")
    print("Fitting model...")
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_true=y_test, y_pred=model.predict(x_test))
    balanced_accuracy = balanced_accuracy_score(
        y_true=y_test, y_pred=model.predict(x_test)
    )
    if y_values.dtype == "bool":
        auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    else:
        auc = roc_auc_score(
            y_true=y_test, y_score=model.predict_proba(x_test), multi_class="ovr"
        )
    print(
        "Model achieved an accuracy of {:.2f}, balanced accuracy of {:.2f}, AUC of {:.2f}".format(
            accuracy, balanced_accuracy, auc
        )
    )
    return model
