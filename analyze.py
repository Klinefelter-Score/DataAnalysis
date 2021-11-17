from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, recall_score
from classifiers import get_classifier
import pickle
import matplotlib.pyplot as plt


# Features as defined in human reprod. publication
features = [
    "Age",
    "BMI",
    "Oestradiol",
    "FSH",
    "Height",
    "LH",
    "pH",
    "Prolaktin",
    "Testosteron",
    "Vol",
    "HVol",
]
# How to treat missing values. If not dropped use the mean.
drop_na = True
classifier = "Ada" # Ada, SVM, CatBoost, MLP, XGBoost, KNN, Ensemble, RandomForest
class_threshold = 0.5


def run():
    X, y = prepare_data()
    model = pickle.load(open(f"model_SVM.pickle", "rb"))
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5432)
    result = cross_validate(
        model,
        X,
        y,
        cv=outer_cv,
        scoring={
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "Specificity": make_scorer(specificity_score)
        }
    )
    print(f"Baseline: {sum(y)/len(y):.3f}")
    for name, key in [("Accuracy", "test_Accuracy"), ("Recall", "test_Recall"), ("Precision", "test_Precision"), ("Specificity", "test_Specificity")]:
        print(f"{name}: {np.mean(result[key])} +- {np.std(result[key]):.3f}")


def run_proba():
    X, y = prepare_data()
    # model = pickle.load(open(f"model_Ensemble.pickle", "rb"))
    model = get_classifier(classifier)
    # return
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5432)
    recall, specificity, accuracy = ([], [], [])
    for train, test in outer_cv.split(X, y):
        model.fit(X[train], y[train])
        y_pred = model.predict_proba(X[test])
        if len(y_pred.shape) == 1:
            y_pred = [1 if next_pred > class_threshold else 0 for next_pred in y_pred]
        else:
            healthy, kf = [[], []]
            for next_x, next_y in zip(y_pred[:, 1].flatten(), y[test]):
                if next_y == 1:
                    kf.append(next_x)
                else:
                    healthy.append(next_x)
            healthy.sort()
            kf.sort()
            plt.scatter(healthy, np.cumsum(healthy) / sum(healthy), color=["b" for _ in healthy])
            plt.scatter(kf, np.cumsum(kf) / sum(kf), color=["r" for _ in kf])
            y_pred = [1 if next_pred[1] > class_threshold else 0 for next_pred in y_pred]
        recall.append(recall_score(y[test], y_pred))
        specificity.append(specificity_score(y[test], y_pred))
        accuracy.append(accuracy_score(y[test], y_pred))
    print(f"Accuracy: {np.mean(accuracy)} ({np.std(accuracy)})")
    print(f"Recall: {np.mean(recall)} ({np.std(recall)})")
    print(f"Specif.: {np.mean(specificity)} ({np.std(specificity)})")
    plt.vlines(class_threshold, 0, 1)
    plt.show()


def save_model():
    X, y = prepare_data()
    model = get_classifier(classifier)
    model.fit(X, y)
    pickle.dump(model, open(f"model_{classifier}.pickle", "wb"))


def test_model():
    X, y = prepare_data()
    model = pickle.load(open(f"model_{classifier}.pickle", "rb"))
    y_pred = model.predict_proba(X)
    y_pred = [1 if next_pred[1] > class_threshold else 0 for next_pred in y_pred]
    print(f"Accuracy: {accuracy_score(y, y_pred)}")
    print(f"Recall: {recall_score(y, y_pred)}")
    print(f"Specif.: {specificity_score(y, y_pred)}")


def specificity_score(y_true, y_pred, **kwargs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def load_patients():
    # Studie ML Azoo KS 2020-06-30_pseudo.csv
    # Patienten prospektiv ab 1.7.19 2020-06-07_pseudo.csv
    retrospective = "retrospectiveData.csv"
    prospective = "prospectiveDataAzoo.csv"
    crypto = "prospectiveDataCrypto.csv"
    file_name = retrospective
    # file_name = "Patienten prospektiv ab 1.7.19 2020-06-07_pseudo.csv"
    return pd.read_csv(f'../data/{file_name}', delimiter=',')


def prepare_data() -> Tuple[np.ndarray, list]:
    # Load everything from the given CSV-File
    all_data = load_patients()
    all_data.replace('kl', None, inplace=True)
    all_data.replace('nd', None, inplace=True)

    # Reduce to the relevant Feature/Target Columns
    relevant_data = all_data[[*features, "Karyotyp"]]
    if "Oestradiol" in relevant_data:
        relevant_data["Oestradiol"] = pd.to_numeric(relevant_data["Oestradiol"], errors='coerce')
    relevant_data[relevant_data.columns.difference(["Karyotyp"])] = relevant_data[relevant_data.columns.difference(["Karyotyp"])].astype(np.float)
    # Drop or Replace NaNs
    if drop_na:
        relevant_data.dropna(axis=0, inplace=True)
    else:
        relevant_data.fillna(relevant_data.mean(), inplace=True)

    # Prepare the Samples/Targets
    X, y = [[], []]
    for _, next_row in relevant_data.iterrows():
        next_sample = next_row[features]
        if str(next_row["Karyotyp"]).startswith("46"):
            X.append(next_sample.to_list())
            y.append(0)
        elif str(next_row["Karyotyp"]).startswith("47"):
            X.append(next_sample.to_list())
            y.append(1)

    # X = [transform_sample(next_sample) for next_sample in X]
    X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.int)
    print(f"{X.shape} Samples of which {sum(y)} with KF Syndrome")

    return X, y


def descriptiveStatistic():
    global features
    features = ["Weight"]
    X, y = prepare_data()
    neg, pos = [[], []]
    for next_x, next_y in zip(X, y):
        if next_y == 0:
            neg.append(next_x[0])
        else:
            pos.append(next_x[0])
    print(f"{features[0]}: {np.mean(X)} [{min(X)} - {max(X)}]")
    print(f"\tnon-KS: {np.mean(neg)} [{min(neg)} - {max(neg)}]")
    from scipy.stats import mannwhitneyu, ttest_ind, pointbiserialr, kstest
    import researchpy
    import random, pandas
    random.shuffle(pos)
    random.shuffle(neg)
    as_matrix = list(zip(X, y))
    random.shuffle(as_matrix)
    X, y = list(zip(*as_matrix))
    total = [*pos, *neg]
    random.shuffle(total)
    df_neg = pandas.DataFrame(neg, columns=["h"])
    df_pos = pandas.DataFrame(pos, columns=["nh"])
    mwu = mannwhitneyu(pos, neg) # just to be sure
    ttest = researchpy.ttest(df_pos["nh"], df_neg["h"], group1_name="KS", group2_name="non-KS", equal_variances=False)
    pbr = pointbiserialr(y, [next_x[0] for next_x in X])
    ks = kstest(neg, pos)
    print(f"MannWhitneyU: {mwu} - {mwu[1] < 0.01}")
    print(f"Point Biserial: {pbr[0]} - {pbr[1]}")
    print(f"Kolmo. Smi.: {ks}")


def transform_sample(sample):
    out = list(sample)
    left_right_coef = max(sample[-2:]) - min(*sample[-2:])
    # left_right_coef = min(*sample[-2:]) / max(sample[-2:])
    max_vol = sum(sample[-2:])
    out[-2] = left_right_coef
    out[-1] = max_vol
    return out


if __name__ == "__main__":
    pass
    # run()
    # run_proba()
    # save_model()
    # test_model()
    descriptiveStatistic()
