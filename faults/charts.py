from matplotlib.figure import Figure
import seaborn as sns


def horizontal(values, title):
    figure = Figure(figsize=(5, 3.2), layout="constrained")
    ax = figure.subplots()
    values = values.sort_values(ascending=False)
    bars = ax.barh(values.index, values, color=["#D75B13"] + ["#776B63"] * (len(values) - 1))
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%g", padding=4)
    ax.margins(x=.2)
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(length=0, labelsize=8)
    ax.xaxis.set_visible(False)
    sns.despine(ax=ax, left=True, bottom=True)
    return figure


def plot_class_counts(counts, title):
    return horizontal(counts, title)


def plot_drift_scores(scores):
    return horizontal(scores.head(5).round(3), "Расстояние Колмогорова–Смирнова")


def plot_learning_curves(curves):
    figure = Figure(figsize=(6, 3.6), layout="constrained")
    ax = figure.subplots()
    names = {"active": "Активный отбор", "random": "Случайный отбор"}
    sns.lineplot(data=curves.assign(strategy=curves.strategy.map(names)), x="budget", y="macro_f1",
                 hue="strategy", palette={names["active"]: "#D75B13", names["random"]: "#776B63"},
                 marker="o", errorbar=None, ax=ax)
    ax.set(xlabel="Число ручных разметок", ylabel="Macro-F1")
    ax.legend(title=None)
    sns.despine(ax=ax)
    return figure
