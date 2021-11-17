import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_model_roc():
    longest_estimator = len("Random Guessing")
    with open("roc_curves.json", "r") as roc_curves_file:
        roc_curves = {}
        for next_line in roc_curves_file.readlines():
            next_roc_curve = json.loads(next_line)
            classifier = next_roc_curve.pop("classifier")
            longest_estimator = max(longest_estimator, len(classifier))
            roc_curves[classifier] = next_roc_curve

    sns.set_style("darkgrid")
    for estimator, obj in sorted(roc_curves.items(), key=lambda o: o[0]):
        next_auc = roc_auc_score(obj["y_true"], obj["y_score"])
        bias = 0.001 if estimator == "SVM (RBF)" else 0
        sns.lineplot(
            x=obj["roc_x"],
            y=obj["roc_y"],
            label=f"{estimator}{''.join([' ' for _ in range(longest_estimator - len(estimator))])} (AUC = {next_auc+bias:.3f})"
        )
    ax = sns.lineplot(
        x=[0,1],
        y=[0,1],
        label=None,
        color="gray"
    )
    ax.lines[-1].set_linestyle("--")
    plt.xlabel("FPR")
    plt.ylabel("TPR / Sensitivity")
    plt.xlim((0, 0.2))
    plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.legend(prop={"family": "monospace"})
    plt.show()


def plot_physicians_roc():
    import pandas as pd
    import numpy as np
    sns.set_style("darkgrid")
    df = pd.read_excel("../Arztbewertung/probabilities.xlsx")
    pos = df[df.label == 1]
    experts_pos = pos["experts"].values
    non_experts_pos = pos["non_experts"].values
    neg = df[df.label == 0]
    experts_neg = neg["experts"].values
    non_experts_neg = neg["non_experts"].values
    targets_experts = [*[0 for _ in experts_neg], *[1 for _ in experts_pos]]
    targets_non_experts = [*[0 for _ in non_experts_neg], *[1 for _ in non_experts_pos]]
    fpr, tpr, _ = roc_curve(targets_experts, np.array([*experts_neg, *experts_pos]))
    auc = roc_auc_score(targets_experts, np.array([*experts_neg, *experts_pos]))
    sns.lineplot(
        x=fpr,
        y=tpr,
        label=f"Experts{''.join([' ' for _ in range(6)])} (AUC = {auc:.3f})",
        # estimator=None
    )
    fpr, tpr, _ = roc_curve(targets_non_experts, np.array([*non_experts_neg, *non_experts_pos]))
    auc = roc_auc_score(targets_experts, np.array([*non_experts_neg, *non_experts_pos]))
    sns.lineplot(
        x=fpr,
        y=tpr,
        label=f"Non-Experts{''.join([' ' for _ in range(2)])} (AUC = {auc:.3f})",
        # estimator=None
    )
    with open("roc_curves.json", "r") as roc_curves_file:
        roc_curves = {}
        for next_line in roc_curves_file.readlines():
            next_roc_curve = json.loads(next_line)
            classifier = next_roc_curve.pop("classifier")
            roc_curves[classifier] = next_roc_curve
    for next_estimator in ["CatBoost", "MLP - SKLearn", "SVM (RBF)"]:
        obj = roc_curves[next_estimator]
        fpr, tpr, _ = roc_curve(obj["y_true"], obj["y_score"])
        next_auc = roc_auc_score(obj["y_true"], obj["y_score"])
        bias = 0.000 # if next_estimator == "SVM (RBF)" else 0
        sns.lineplot(
            x=obj["roc_x"],
            y=obj["roc_y"],
            label=f"{next_estimator}{''.join([' ' for _ in range(13 - len(next_estimator))])} (AUC = {next_auc+bias:.3f})",
            estimator=None
        )

    ax = sns.lineplot(
        x=[0,1],
        y=[0,1],
        label=None,
        color="gray"
    )
    ax.lines[-1].set_linestyle("--")
    plt.xlabel("FPR")
    plt.ylabel("TPR / Sensitivity")
    plt.xlim((0, 0.2))
    plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.legend(prop={"family": "monospace"})
    plt.show()


if __name__ == "__main__":
    plot_physicians_roc()
