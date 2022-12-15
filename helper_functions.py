from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def plot_density_graph(
    df:pd.DataFrame,
    var: str,
    hue_: str = None,
    quantile_upper: float = 1.0
):
    tmp = df[df[var].le(df[var].quantile(quantile_upper))]
    sns.kdeplot(data=tmp, x=var, hue=hue_)
    plt.title(f"Density function for {var}")
    plt.show()