#!/usr/bin/env python3
"""
EDA_functions.py

A collection of random functions that I used for EDA analysis while a Masters student at JHU.

Contains:
  - single_variable_eda: Univariate EDA (categorical/numerical)
  - pairwise_eda: Bivariate EDA (cat-cat, cat-num, num-num)
  - correlation_analysis: Correlation heatmaps and target-based analysis
  - statistical_tests: Suite of statistical tests for a target variable
  - chord_diagram: Holoviews-based chord diagram of associations
"""

# Standard Library Imports
import os
import warnings
import logging
from pathlib import Path
from math import pi
import itertools

# Third-Party Imports
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
import holoviews as hv
from scipy.stats import chi2_contingency, f_oneway, pointbiserialr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import brier_score_loss
from IPython.display import display, HTML
from pandas.plotting import register_matplotlib_converters

# Initialize Holoviews and configure environment
hv.extension('bokeh')
register_matplotlib_converters()
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Configure Logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global Style
sns.set_palette('colorblind')

# Utility Functions
def _get_aggregate_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    stats_df = df[column].describe().to_frame().T
    stats_df['missing'] = df[column].isnull().sum()
    return stats_df

def _freedman_diaconis_bins(series: pd.Series) -> int:
    data = series.dropna()
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    width = 2 * iqr * len(data)**(-1/3)
    return max(1, int((data.max() - data.min()) / width))

def _identify_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] < lb) | (df[column] > ub)]

def _determine_type(series: pd.Series) -> str:
    return 'categorical' if series.dtype == 'object' or series.nunique() < 20 else 'numerical'

def cramers_v(confusion: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.values.sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(r-1, k-1))

def correlation_ratio(cats: pd.Series, nums: pd.Series) -> float:
    fcat, _ = pd.factorize(cats)
    n_cat = fcat.max() + 1
    y_mean = nums.mean()
    sizes = []
    means = []
    for i in range(n_cat):
        grp = nums[fcat == i]
        sizes.append(len(grp))
        means.append(grp.mean())
    num = sum(s*(m-y_mean)**2 for s,m in zip(sizes, means))
    den = ((nums - y_mean)**2).sum()
    return np.sqrt(num/den) if den else 0.0

# Plotting Helpers
def _plot_categorical(df: pd.DataFrame, column: str, output_dir: Path | None = None, **kwargs) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    vals = df[column].value_counts()
    sns.countplot(x=column, data=df, ax=axes[0], **kwargs)
    axes[0].set(title=f'Count of {column}', xlabel=column, ylabel='Frequency')
    squarify.plot(sizes=vals.values, label=vals.index, ax=axes[1], **kwargs)
    axes[1].set(title=f'Treemap of {column}'); axes[1].axis('off')
    if len(vals)>=3:
        angles = np.linspace(0, 2*pi, len(vals), endpoint=False).tolist(); angles += angles[:1]
        v = vals.values.tolist(); v += v[:1]
        ax = plt.subplot(1,3,3,polar=True)
        ax.fill(angles, v, alpha=0.25); ax.plot(angles, v)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(vals.index)
        ax.set(title=f'Radar of {column}')
    else:
        axes[2].text(0.5,0.5,'Insufficient categories',ha='center'); axes[2].axis('off')
    plt.tight_layout()
    if output_dir: plt.savefig(Path(output_dir)/f'{column}_cat.png')
    plt.show()
    vc = vals.to_frame().rename(columns={column:'count'})
    vc['%'] = (vc['count']/len(df)*100).round(2).astype(str)+'%'
    display(HTML(vc.reset_index().rename(columns={'index':column}).to_html(index=False)))

def _plot_numerical(df: pd.DataFrame, column: str, output_dir: Path | None = None, bins: int | None = None, **kwargs) -> None:
    ser = pd.to_numeric(df[column], errors='coerce')
    bins = bins or _freedman_diaconis_bins(ser)
    fig, axes = plt.subplots(1,4,figsize=(24,6))
    sns.boxplot(x=ser, ax=axes[0], **kwargs); axes[0].set(title=f'Boxplot of {column}')
    sns.histplot(ser, kde=True, bins=bins, ax=axes[1], **kwargs); axes[1].set(title=f'Hist of {column}')
    sns.violinplot(x=ser, ax=axes[2], **kwargs); axes[2].set(title=f'Violin of {column}')
    sns.kdeplot(ser, ax=axes[3], **kwargs); axes[3].set(title=f'KDE of {column}')
    plt.tight_layout()
    if output_dir: plt.savefig(Path(output_dir)/f'{column}_num.png')
    plt.show()

# Main EDA Functions
def single_variable_eda(df: pd.DataFrame, column: str, analysis_type: str | None = None, bins: int | None = None, output_dir: Path | None = None, **kwargs) -> None:
    if column not in df: raise KeyError(f"Column '{column}' not found.")
    atype = analysis_type or _determine_type(df[column])
    logger.info("Single var EDA: %s (%s)", column, atype)
    if atype=='categorical': _plot_categorical(df,column,output_dir,**kwargs)
    else: _plot_numerical(df,column,output_dir,bins,**kwargs)
    display(HTML(_get_aggregate_stats(df,column).to_html(index=False)))
    if atype=='numerical':
        outs=_identify_outliers(df,column)
        if len(outs): display(HTML(outs.head(10).to_html(index=False)))

def pairwise_eda(df: pd.DataFrame, col1: str, col2: str, analysis_type: str | None = None, bins1: int | None = None, bins2: int | None = None, output_dir: Path | None = None, **kwargs) -> None:
    atype = analysis_type or f"{_determine_type(df[col1])}_to_{_determine_type(df[col2])}"
    logger.info("Pairwise EDA: %s vs %s (%s)", col1, col2, atype)
    if 'cat_to_cat' in atype:
        plt.figure(figsize=(12,5))
        ct=pd.crosstab(df[col1],df[col2])
        sns.heatmap(ct,annot=True,cmap='Blues'); plt.title(f'{col1} vs {col2}')
        sns.countplot(x=col1,hue=col2,data=df); plt.show()
    elif 'cat_to_num' in atype:
        plt.figure(figsize=(12,5))
        sns.boxplot(x=col1,y=col2,data=df); sns.violinplot(x=col1,y=col2,data=df); plt.show()
    elif 'num_to_num' in atype:
        plt.figure(figsize=(12,5))
        sns.scatterplot(x=col1,y=col2,data=df)
        low=sm.nonparametric.lowess(df[col2],df[col1]); plt.plot(low[:,0],low[:,1]); plt.show()
    cors=[]
    if 'num_to_num' in atype:
        cors=[('pearson',df[col1].corr(df[col2])),('spearman',df[col1].corr(df[col2],method='spearman'))]
    elif 'cat_to_cat' in atype:
        cors=[('cramers',cramers_v(pd.crosstab(df[col1],df[col2])))]
    else:
        cors=[('pbs',pointbiserialr(df[col1].astype(int),df[col2])[0])]
    display(HTML(pd.DataFrame(cors,columns=['Measure','Value']).to_html(index=False)))

def correlation_analysis(df: pd.DataFrame, target: str | None = None) -> None:
    nums=df.select_dtypes(include='number')
    cats=df.select_dtypes(include='object')
    if not nums.empty:
        plt.figure(figsize=(8,6)); sns.heatmap(nums.corr(),annot=True); plt.show()
    if not cats.empty:
        m=pd.DataFrame(index=cats.columns,columns=cats.columns)
        for i,j in itertools.combinations(cats.columns,2): m.loc[i,j]=m.loc[j,i]=cramers_v(pd.crosstab(df[i],df[j]))
        plt.figure(figsize=(8,6)); sns.heatmap(m.astype(float),annot=True); plt.show()
    if target and target in df:
        # simple bar of correlations
        if target in nums:
            corrs=nums.corrwith(df[target]).abs().sort_values()
            display(HTML(corrs.to_frame('corr').to_html()))

def statistical_tests(df: pd.DataFrame, target: str, p_value_threshold: float = 0.05) -> None:
    nums=df.select_dtypes(include='number'); cats=df.select_dtypes(include='object')
    if target in cats:
        for num in nums:
            p=f_oneway(*[df[df[target]==lvl][num] for lvl in df[target].unique()])[1]
            print(f'{num}: p={p:.3f}')
    if target in nums:
        for cat in cats:
            _,p=chi2_contingency(pd.crosstab(df[cat],pd.qcut(df[target],3)))[1:3]
            print(f'{cat}: p={p:.3f}')

def chord_diagram(df: pd.DataFrame, threshold: float = 0.3, strong_threshold: float = 0.5, output_file: Path | None = None) -> hv.Chord:
    cols=df.columns.tolist(); size=len(cols)
    mat=pd.DataFrame(np.zeros((size,size)),cols,cols)
    for i,j in itertools.combinations(range(size),2):
        c1,c2=cols[i],cols[j]
        if df[c1].dtype==object and df[c2].dtype==object:
            val=cramers_v(pd.crosstab(df[c1],df[c2]))
        elif df[c1].dtype==object or df[c2].dtype==object:
            val=correlation_ratio(df[c1] if df[c1].dtype==object else df[c2], df[c2] if df[c1].dtype==object else df[c1])
        else:
            val=df[c1].corr(df[c2])
        mat.iat[i,j]=mat.iat[j,i]=val
    edges=[]
    for i,j in itertools.combinations(range(size),2):
        v=mat.iat[i,j]
        if abs(v)>=threshold: edges.append((cols[i],cols[j],v))
    hv_edges=pd.DataFrame(edges,columns=['from','to','value'])
    chord=hv.Chord(hv.Dataset(hv_edges,['from','to'],['value']))
    chord.opts(cmap='Category20',edge_color='value',labels='index')
    if output_file: hv.save(chord,str(output_file))
    return chord