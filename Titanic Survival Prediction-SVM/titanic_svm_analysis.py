import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    '#f9f9f9',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#e0e0e0',
    'grid.linewidth':    0.8,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
})

BLUE   = '#2979FF'
PURPLE = '#7C4DFF'
GREEN  = '#00C853'
RED    = '#FF5252'
AMBER  = '#FFB300'
GRAY   = '#90A4AE'

DATA_PATH = "/home/claude/titanic.csv"

print("=" * 55)
print("  TITANIC SVM CLASSIFICATION")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

miss     = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df  = pd.DataFrame({'Count': miss, 'Pct': miss_pct})[miss > 0]
print(f"\n Missing values:\n{miss_df}")


# ── Figure 1: Missing Values ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
fig.subplots_adjust(left=0.12, right=0.78, top=0.88, bottom=0.18)

cols_m = ['Age', 'Cabin', 'Embarked']
pcts_m = [19.87, 77.10, 0.22]
clrs_m = [BLUE, RED, GREEN]
notes  = ['→ Imputed: median by Pclass', '→ Dropped: too many missing', '→ Imputed: mode (S)']

bars = ax.barh(cols_m, pcts_m, color=clrs_m, height=0.4,
               edgecolor='white', linewidth=1.2)

for bar, pct in zip(bars, pcts_m):
    ax.text(bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f'{pct}%', va='center', ha='left', fontsize=11, fontweight='bold')

ax.axvline(50, color=GRAY, linestyle='--', linewidth=1.2, label='50% threshold')
ax.set_xlim(0, 100)
ax.set_xlabel('Missing %', labelpad=8)
ax.set_title('Figure 1 — Missing Values per Column', pad=12)
ax.legend(loc='lower right')

for i, (bar, note) in enumerate(zip(bars, notes)):
    ax.annotate(note,
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(82, bar.get_y() + bar.get_height() / 2),
                fontsize=8.5, color='#444444', va='center',
                xycoords='data', textcoords='data',
                arrowprops=None)

plt.savefig('fig1_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print(" [Fig 1 saved]")


# ── Preprocessing ──────────────────────────────────────────────────────────
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
df.drop(columns=['SibSp', 'Parch', 'Name', 'Ticket', 'PassengerId'], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df        = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

print(f"\n Features after engineering: {list(df.columns)}")

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scale_cols = ['Age', 'Fare', 'FamilySize']
scaler     = StandardScaler()
X_train_s  = X_train.copy()
X_test_s   = X_test.copy()
X_train_s[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_s[scale_cols]  = scaler.transform(X_test[scale_cols])


# ── Linear SVM ────────────────────────────────────────────────────────────
svm_lin       = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_lin.fit(X_train_s, y_train)
lin_train_acc = accuracy_score(y_train, svm_lin.predict(X_train_s))
lin_test_acc  = accuracy_score(y_test,  svm_lin.predict(X_test_s))
lin_roc       = roc_auc_score(y_test, svm_lin.predict_proba(X_test_s)[:, 1])
print(f"\n Linear SVM -> Train: {lin_train_acc:.4f} | Test: {lin_test_acc:.4f} | AUC: {lin_roc:.4f}")


# ── RBF Grid Search ───────────────────────────────────────────────────────
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1]}
rbf_gs = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, verbose=1)
rbf_gs.fit(X_train_s, y_train)

best_params = rbf_gs.best_params_
best_cv_auc = rbf_gs.best_score_
best_rbf    = rbf_gs.best_estimator_
print(f"\n Best RBF -> C={best_params['C']}, gamma={best_params['gamma']}, CV AUC={best_cv_auc:.4f}")

cv_res   = pd.DataFrame(rbf_gs.cv_results_)
cv_table = cv_res[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score']].copy()
cv_table.columns = ['C', 'Gamma', 'Mean AUC', 'Std AUC']
cv_table = cv_table.sort_values('Mean AUC', ascending=False).reset_index(drop=True)


# ── Figure 2: CV Heatmap + Rankings ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.38)

pivot = cv_table.copy()
pivot['Gamma'] = pivot['Gamma'].astype(str)
hmap  = pivot.pivot(index='C', columns='Gamma', values='Mean AUC')
hmap  = hmap.reindex(columns=['scale', 'auto', '0.1'])

sns.heatmap(hmap, annot=True, fmt='.4f', cmap='Blues',
            linewidths=1.5, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=axes[0], cbar_kws={'shrink': 0.75, 'label': 'Mean AUC', 'pad': 0.02})
axes[0].set_title('Figure 2a — CV ROC-AUC Heatmap (C × Gamma)', pad=14)
axes[0].set_xlabel('Gamma', fontsize=11, labelpad=8)
axes[0].set_ylabel('C', fontsize=11, labelpad=8)
axes[0].tick_params(axis='x', rotation=0)
axes[0].tick_params(axis='y', rotation=0)

labels_cv  = [f"C={r['C']},  gamma={r['Gamma']}" for _, r in cv_table.iterrows()]
bar_colors = [PURPLE if i == 0 else '#BBDEFB' for i in range(len(cv_table))]
axes[1].barh(labels_cv[::-1], cv_table['Mean AUC'].values[::-1],
             color=bar_colors[::-1], edgecolor='white', height=0.55)


x_min = cv_table['Mean AUC'].min() - 0.008
x_max = cv_table['Mean AUC'].max() + 0.018
axes[1].set_xlim(x_min, x_max)
axes[1].set_xlabel('Mean CV ROC-AUC', labelpad=8)
axes[1].set_title('Figure 2b — Grid Search Rankings', pad=14)
axes[1].axvline(best_cv_auc, color=PURPLE, linestyle='--', linewidth=1.5,
                label=f'Best = {best_cv_auc:.4f}')
axes[1].legend(loc='lower right')

for i, val in enumerate(cv_table['Mean AUC'].values[::-1]):
    axes[1].text(val + 0.0005, i, f'{val:.4f}',
                 va='center', ha='left', fontsize=9, color='#333333')

plt.savefig('fig2_grid_search.png', dpi=150, bbox_inches='tight')
plt.show()
print(" [Fig 2 saved]")


# ── Test Set Metrics ───────────────────────────────────────
y_pred  = best_rbf.predict(X_test_s)
y_proba = best_rbf.predict_proba(X_test_s)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
mcc  = matthews_corrcoef(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

lin_pred_test = svm_lin.predict(X_test_s)
lin_prec = precision_score(y_test, lin_pred_test)
lin_rec  = recall_score(y_test, lin_pred_test)
lin_f1   = f1_score(y_test, lin_pred_test)
lin_mcc  = matthews_corrcoef(y_test, lin_pred_test)
improvement = auc - lin_roc


# ── Figure 3: Metrics Comparison ─────────────────────────
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'ROC-AUC']
lin_vals     = [lin_test_acc, lin_prec, lin_rec, lin_f1, lin_mcc, lin_roc]
rbf_vals     = [acc, prec, rec, f1, mcc, auc]
x_pos        = np.arange(len(metric_names))
w            = 0.32

fig, ax = plt.subplots(figsize=(13, 6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.18)

b1 = ax.bar(x_pos - w/2, lin_vals, w, label='Linear SVM (C=1)',
            color=BLUE, alpha=0.88, edgecolor='white', linewidth=1.2)
b2 = ax.bar(x_pos + w/2, rbf_vals, w,
            label=f'RBF SVM  (C={best_params["C"]}, γ={best_params["gamma"]})',
            color=PURPLE, alpha=0.88, edgecolor='white', linewidth=1.2)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=8.5, color=BLUE, fontweight='bold')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=8.5, color=PURPLE, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0, 1.18)   
ax.set_ylabel('Score', labelpad=8)
ax.set_title('Figure 3 — Linear vs RBF SVM: All Metrics', pad=14)
ax.axhline(0.8, color=GRAY, linestyle=':', linewidth=1, alpha=0.7, label='0.8 reference')
ax.legend(loc='upper left', framealpha=0.9)

delta_lbl = (f"AUC Δ = {improvement:+.4f}  —  "
             f"{'Meaningful improvement (>0.02)' if improvement > 0.02 else 'Not meaningful (Δ ≤ 0.02)'}")
fig.text(0.52, 0.04, delta_lbl, ha='center', fontsize=10, color='#333333',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#F9A825'))

plt.savefig('fig3_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(" [Fig 3 saved]")


# ── Figure 4: ROC Curves ──────────────────────────
fpr_lin, tpr_lin, _ = roc_curve(y_test, svm_lin.predict_proba(X_test_s)[:, 1])
fpr_rbf, tpr_rbf, _ = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 7))
fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)

ax.plot(fpr_lin, tpr_lin, color=BLUE, lw=2.5, linestyle='--',
        label=f'Linear SVM  (AUC = {lin_roc:.4f})')
ax.plot(fpr_rbf, tpr_rbf, color=PURPLE, lw=2.5,
        label=f'RBF SVM     (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, linestyle=':', label='Random classifier')
ax.fill_between(fpr_rbf, tpr_rbf, alpha=0.08, color=PURPLE)
ax.fill_between(fpr_lin, tpr_lin, alpha=0.08, color=BLUE)

ax.set_xlabel('False Positive Rate', labelpad=8)
ax.set_ylabel('True Positive Rate', labelpad=8)
ax.set_title('Figure 4 — ROC Curves: Linear vs RBF SVM', pad=14)
ax.legend(loc='lower right', framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.04])

# Delta annotation 
ax.annotate(f'Δ AUC = {improvement:+.4f}',
            xy=(0.18, 0.62),
            fontsize=12, color=PURPLE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#EDE7F6', edgecolor=PURPLE, alpha=0.9))

plt.savefig('fig4_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(" [Fig 4 saved]")


# ── Figure 5: Confusion Matrix ────────────────────────────────
cm             = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr_val        = fp / (fp + tn)
recall_val     = tp / (tp + fn)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.12, wspace=0.40)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Not Survived', 'Survived'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(
    f'Figure 5a — Confusion Matrix\n(C={best_params["C"]}, γ={best_params["gamma"]})',
    pad=12)
axes[0].set_xlabel('Predicted Label', fontsize=11, labelpad=8)
axes[0].set_ylabel('True Label', fontsize=11, labelpad=8)
for text in axes[0].texts:
    text.set_fontsize(15)
    text.set_fontweight('bold')

cats   = ['True Neg\n(correct)', 'False Pos\n(→survived)', 'False Neg\n(→not surv)', 'True Pos\n(correct)']
counts = [tn, fp, fn, tp]
clrs   = [GREEN, AMBER, RED, PURPLE]

bars2 = axes[1].bar(cats, counts, color=clrs, edgecolor='white',
                    linewidth=1.5, width=0.5)
for bar, cnt in zip(bars2, counts):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.6,
                 str(cnt), ha='center', va='bottom', fontsize=13, fontweight='bold')

axes[1].set_ylabel('Count', labelpad=8)
axes[1].set_title('Figure 5b — Confusion Matrix Breakdown', pad=12)
axes[1].set_ylim(0, max(counts) * 1.30) 
axes[1].tick_params(axis='x', labelsize=9.5)

# Info box
info_txt = (f"False Positive Rate : {fpr_val*100:.1f}%\n"
            f"(non-survivors → survived)\n\n"
            f"Recall (TPR)        : {recall_val*100:.1f}%\n"
            f"(survivors correctly id.)")
axes[1].text(0.97, 0.97, info_txt,
             transform=axes[1].transAxes,
             va='top', ha='right', fontsize=9.5,
             linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.6',
                       facecolor='#F3E5F5', edgecolor=PURPLE, alpha=0.9))

plt.savefig('fig5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print(" [Fig 5 saved]")


# ── Figure 6: Summary Card ─────────────────────
fig = plt.figure(figsize=(11, 9))
fig.patch.set_facecolor('#FAFAFA')
ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])   # [left, bottom, width, height]
ax.axis('off')

summary_rows = [
    ("DATASET",               ""),
    ("  Records",             "891 passengers"),
    ("  Train / Test",        "712 / 179  (80/20 stratified)"),
    ("",                      ""),
    ("LINEAR SVM  (C=1)",     ""),
    ("  Train Accuracy",      f"{lin_train_acc:.4f}"),
    ("  Test  Accuracy",      f"{lin_test_acc:.4f}"),
    ("  ROC-AUC",             f"{lin_roc:.4f}"),
    ("",                      ""),
    ("BEST RBF SVM",          f"C={best_params['C']},  gamma={best_params['gamma']}"),
    ("  CV AUC (5-fold)",     f"{best_cv_auc:.4f}"),
    ("  Accuracy",            f"{acc:.4f}"),
    ("  Precision",           f"{prec:.4f}"),
    ("  Recall",              f"{rec:.4f}"),
    ("  F1 Score",            f"{f1:.4f}"),
    ("  MCC",                 f"{mcc:.4f}"),
    ("  ROC-AUC",             f"{auc:.4f}"),
    ("",                      ""),
    ("COMPARISON",            ""),
    ("  AUC delta",           f"{improvement:+.4f}  ({'meaningful' if improvement > 0.02 else 'not meaningful (<0.02)'})"),
    ("",                      ""),
    ("CONFUSION MATRIX",      ""),
    ("  False Positive Rate", f"{fpr_val*100:.1f}%  (non-survivors predicted as survived)"),
    ("  Recall (TPR)",        f"{recall_val*100:.1f}%  (survivors correctly identified)"),
]

ax.set_title('Figure 6 — Final Summary Card', fontsize=14, fontweight='bold',
             pad=14, loc='center')

ROW_H  = 0.038
y_pos  = 0.94
LEFT_X = 0.02
VAL_X  = 0.52

for label, value in summary_rows:
    if label == "":
        y_pos -= ROW_H * 0.6
        continue

    is_header = not label.startswith("  ")

    if is_header:
        ax.add_patch(plt.Rectangle((0, y_pos - 0.005), 1.0, ROW_H + 0.002,
                                   transform=ax.transAxes, clip_on=True,
                                   facecolor='#E8EAF6', edgecolor='none', zorder=0))

    ax.text(LEFT_X, y_pos, label,
            transform=ax.transAxes,
            fontsize=11 if is_header else 10.5,
            color='#1A237E' if is_header else '#212121',
            fontweight='bold' if is_header else 'normal',
            va='top', fontfamily='monospace')

    if value:
        ax.text(VAL_X, y_pos, value,
                transform=ax.transAxes,
                fontsize=10.5, color='#37474F',
                va='top', fontfamily='monospace')

    y_pos -= ROW_H

# Border
ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0,
             transform=ax.transAxes, fill=False,
             edgecolor=PURPLE, linewidth=2.5, clip_on=False))

plt.savefig('fig6_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print(" [Fig 6 saved]")


# ── Terminal Report ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL RESULTS")
print("=" * 55)
print(f"  Linear SVM  ->  AUC: {lin_roc:.4f}  Acc: {lin_test_acc:.4f}")
print(f"  RBF SVM     ->  AUC: {auc:.4f}  Acc: {acc:.4f}")
print(f"  Best params ->  C={best_params['C']}, gamma={best_params['gamma']}")
print(f"  Delta AUC   ->  {improvement:+.4f}  "
      f"({'meaningful' if improvement > 0.02 else 'NOT meaningful (<= 0.02)'})")
print(f"  FPR         ->  {fpr_val*100:.1f}%  |  Recall: {recall_val*100:.1f}%")
print("=" * 55)
print("  6 figures saved in your project folder.")
print("=" * 55)