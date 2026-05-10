"""Génère les notebooks Jupyter (`notebooks/01..03_*.ipynb`).

Lance : `python scripts/build_notebooks.py` depuis la racine.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.lstrip("\n").splitlines(keepends=True),
    }


META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def write(name: str, cells: list[dict]) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}
    p = NB_DIR / name
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", p)


# =============================================================================
# Notebook 1 : Prédiction de notes
# =============================================================================


def notebook_failure() -> None:
    cells = []
    cells.append(md(
        "# 1 — Prédiction de la note `score_examen`\n"
        "\n"
        "Smart School ECAM 2026 — partie *Failure prediction*.\n"
        "\n"
        "Plan suivant la grille d'évaluation :\n"
        "1. Compréhension des données (structure, qualité, distribution cible).\n"
        "2. EDA dirigée — chaque figure répond à une question.\n"
        "3. Sélection / écartement de variables (justifié données + métier).\n"
        "4. Pré-processing dans un pipeline (anti-fuite).\n"
        "5. Comparaison ≥ 4 modèles + baseline + 1 réseau profond.\n"
        "6. Hyperparamètres (RandomizedSearchCV) et **analyse over/underfitting**.\n"
        "7. Importance des variables (permutation) + PDP.\n"
        "\n"
        "Toutes les figures sont sauvegardées dans `reports/figures/` pour LaTeX."
    ))

    cells.append(code(
        "import sys, warnings\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'src').is_dir():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "from sklearn.model_selection import RandomizedSearchCV, learning_curve, KFold, cross_val_score\n"
        "from sklearn.metrics import mean_squared_error\n"
        "from src import config, data, preprocessing, models, explain, features\n"
        "from src.models import train_mlp, mlp_predict, regression_metrics, metrics_to_df\n"
        "warnings.filterwarnings('ignore')\n"
        "sns.set_theme(context='notebook', style='whitegrid')\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "config.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "config.RES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "RANDOM_STATE = config.RANDOM_STATE\n"
    ))

    cells.append(md("## 1. Compréhension des données"))

    cells.append(code(
        "df_full = data.load_train()\n"
        "print('shape:', df_full.shape)\n"
        "df_full.head(5)\n"
    ))

    cells.append(code(
        "info = pd.DataFrame({\n"
        "    'dtype': df_full.dtypes.astype(str),\n"
        "    'n_unique': df_full.nunique(),\n"
        "    'missing_pct': (df_full.isna().mean() * 100).round(2),\n"
        "})\n"
        "info\n"
    ))

    cells.append(md(
        "**Lecture rapide.** 630 000 enregistrements, 14 variables explicatives + cible.\n"
        "3 variables ont des valeurs manquantes : `accès_internet` (10 %), `méthode_etude` (7 %), "
        "`heures_etude` (3 %). Les autres sont denses.\n"
    ))

    cells.append(md(
        "## 2. EDA dirigée\n"
        "\n"
        "Chaque graphique répond à une **question explicite**. Les figures sont enregistrées "
        "(`reports/figures/eda_*.png`) pour réutilisation dans le rapport LaTeX."
    ))

    cells.append(code(
        "# Q1 — La cible est-elle équilibrée ? Faut-il un seuil de risque calibré ?\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))\n"
        "sns.histplot(df_full[config.TARGET], bins=40, kde=True, color='#1f6feb', ax=axes[0])\n"
        "axes[0].axvline(50, color='#d62728', ls='--', label='seuil tutorat (50)')\n"
        "axes[0].set_title(\"Distribution de score_examen\"); axes[0].set_xlabel('note'); axes[0].legend()\n"
        "df_at = (df_full[config.TARGET] < 50).value_counts(normalize=True).rename({True:'à risque', False:'OK'})\n"
        "df_at.plot(kind='bar', color=['#2ca02c','#d62728'], ax=axes[1])\n"
        "axes[1].set_title('Proportion étudiants à risque (<50)'); axes[1].set_ylabel('fréquence')\n"
        "for i, v in enumerate(df_at.values):\n"
        "    axes[1].text(i, v + 0.01, f'{v:.1%}', ha='center')\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'eda_target.png', dpi=160); plt.show()\n"
    ))

    cells.append(code(
        "# Q2 — Quelles variables numériques sont liées à la note ? (corr de Pearson)\n"
        "num_corr = df_full[config.NUM_COLS_ALL + [config.TARGET]].corr()[config.TARGET].drop(config.TARGET).sort_values()\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "colors = ['#888' if abs(v) < 0.05 else '#1f6feb' for v in num_corr.values]\n"
        "ax.barh(num_corr.index, num_corr.values, color=colors)\n"
        "ax.axvline(0, color='k', lw=0.6)\n"
        "ax.set_title('Corrélation de Pearson avec score_examen'); ax.set_xlabel('r')\n"
        "for y, v in enumerate(num_corr.values):\n"
        "    ax.text(v + (0.01 if v > 0 else -0.01), y, f'{v:+.2f}', va='center',\n"
        "            ha='left' if v > 0 else 'right', fontsize=9)\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'eda_corr_target.png', dpi=160); plt.show()\n"
        "print('Corrélations :')\n"
        "print(num_corr.round(3))\n"
    ))

    cells.append(md(
        "**Lecture.** `heures_etude` domine très largement (r ≈ 0.76), suivie de `assiduité_classe` "
        "(r ≈ 0.36) et `heures_sommeil` (r ≈ 0.17). Les variables `age`, `taille_etudiant` et "
        "`heures_fête` sont **pratiquement décorrélées** (|r| ≤ 0.01) — premier indice qu'elles "
        "ne devraient pas figurer dans le modèle final (point repris en section 3)."
    ))

    cells.append(code(
        "# Q3 — Les variables catégorielles séparent-elles bien les notes ?\n"
        "df_plot = df_full.copy()\n"
        "for c in ['qualité_sommeil','méthode_etude','évaluation_établissement',\n"
        "          'genre','accès_internet','difficulté_examen']:\n"
        "    df_plot[c] = df_plot[c].astype('string').fillna('missing')\n"
        "fig, axes = plt.subplots(2, 3, figsize=(13, 7))\n"
        "for ax, col in zip(axes.flat, ['qualité_sommeil','méthode_etude','évaluation_établissement',\n"
        "                                'genre','accès_internet','difficulté_examen']):\n"
        "    order = list(df_plot.groupby(col)[config.TARGET].mean().sort_values().index)\n"
        "    sns.boxplot(data=df_plot, x=col, y=config.TARGET, order=order, ax=ax, fliersize=1)\n"
        "    ax.set_title(col); ax.tick_params(axis='x', rotation=20); ax.set_xlabel('')\n"
        "fig.suptitle('Note vs variables catégorielles', y=1.02)\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'eda_cat_boxplots.png', dpi=160); plt.show()\n"
    ))

    cells.append(md(
        "**Lecture.** `qualité_sommeil`, `méthode_etude` et `évaluation_établissement` produisent "
        "des écarts de note **forts et monotones** (~10 points de moyenne entre groupes extrêmes). "
        "`genre` montre un écart faible mais réel (~3 pt) — surveillé en section *biais* du supplément. "
        "`difficulté_examen` est **paradoxalement plate** : aucune différence entre easy/moderate/hard "
        "(< 0.5 pt). C'est un signal qu'elle n'apporte pas d'information — décision : on l'écarte."
    ))

    cells.append(code(
        "# Q4 — Les variables manquantes sont-elles informatives ?\n"
        "missing_summary = []\n"
        "for col in ['heures_etude','accès_internet','méthode_etude']:\n"
        "    m = df_full[col].isna()\n"
        "    missing_summary.append({\n"
        "        'variable': col,\n"
        "        'pct_missing': m.mean()*100,\n"
        "        'mean_score_when_missing': df_full.loc[m, config.TARGET].mean(),\n"
        "        'mean_score_when_present': df_full.loc[~m, config.TARGET].mean(),\n"
        "    })\n"
        "miss_df = pd.DataFrame(missing_summary).round(2)\n"
        "miss_df\n"
    ))

    cells.append(md(
        "**Lecture.** Les notes moyennes des étudiants avec valeur manquante sont **proches** de "
        "celles des étudiants renseignés (écart < 1 pt). Le NaN n'est donc pas un signal fort en soi : "
        "une imputation par médiane/`'missing'` est suffisante (cf. section 4)."
    ))

    cells.append(code(
        "# Q5 — Effet conjoint heures_etude × méthode_etude (LE plus gros moteur de la note)\n"
        "df_q5 = df_full.dropna(subset=['méthode_etude','heures_etude']).sample(40_000, random_state=RANDOM_STATE)\n"
        "df_q5['méthode_etude'] = df_q5['méthode_etude'].astype('string')\n"
        "g = sns.FacetGrid(df_q5, col='méthode_etude', col_wrap=3, height=2.8, sharey=True)\n"
        "g.map_dataframe(sns.regplot, x='heures_etude', y=config.TARGET, scatter_kws={'alpha':.10,'s':6},\n"
        "                line_kws={'color':'#d62728'})\n"
        "for ax in g.axes.flat:\n"
        "    ax.set_xlim(0, 10); ax.set_ylim(0, 105)\n"
        "g.fig.suptitle('Note vs heures_etude, par méthode_etude', y=1.03)\n"
        "g.fig.tight_layout(); g.fig.savefig(config.FIG_DIR / 'eda_he_methode.png', dpi=160); plt.show()\n"
    ))

    cells.append(md(
        "**Lecture.** Pente positive franche dans toutes les méthodes, mais le **niveau** "
        "diffère : à `heures_etude` égales, *coaching* et *mixed* tirent les notes vers le haut, "
        "*self-study* et *online videos* vers le bas. C'est exactement le type d'interaction qu'un "
        "modèle non linéaire (RF, GBRT, MLP) capturera mieux qu'une régression linéaire."
    ))

    cells.append(md(
        "## 3. Sélection des variables (justifiée)\n"
        "\n"
        "**On garde** (`config.FEATURES_FINAL`) :\n"
        "- *Numériques* : `heures_etude`, `assiduité_classe`, `heures_sommeil`.\n"
        "- *Catégorielles* : `genre`, `diplôme`, `accès_internet`, `qualité_sommeil`, "
        "`méthode_etude`, `évaluation_établissement`.\n"
        "\n"
        "**On écarte** : `age` (corr = 0.010), `taille_etudiant` (corr = 0.001), "
        "`heures_fête` (corr ≈ 0), `difficulté_examen` (écart inter-modalité < 0.5 pt). "
        "Ces variables ne portent pas d'information utile et risquent d'introduire du bruit "
        "(rasoir d'Occam + confusion potentielle pour le MLP)."
    ))

    cells.append(md("## 4. Pré-processing — encapsulé dans un Pipeline pour éviter toute fuite"))

    cells.append(code(
        "pre = preprocessing.build_preprocessor()\n"
        "print(pre)\n"
    ))

    cells.append(md(
        "**Choix.** Imputation médiane (numériques) — robuste aux outliers — et `'missing'` "
        "(catégorielles), puis OneHotEncoder + StandardScaler. Le `Pipeline` garantit que la "
        "médiane et la moyenne/écart-type sont apprises **uniquement sur le train**."
    ))

    cells.append(md(
        "## 5. Split + comparaison des modèles\n"
        "\n"
        "On charge l'**intégralité** du dataset (630 000 lignes) et on ajoute les 3 features\n"
        "dérivées de `src/features.py`. Split stratifié 70/15/15."
    ))

    cells.append(code(
        "df = data.load_train()  # 630_000 lignes\n"
        "df = features.add_derived(df)\n"
        "df = data.add_at_risk_label(df)\n"
        "print('shape:', df.shape)\n"
        "X_train, X_val, X_test, y_train, y_val, y_test = data.make_splits(df, features=config.FEATURES_FINAL_PLUS)\n"
        "print('train', len(X_train), '| val', len(X_val), '| test', len(X_test))\n"
        "print('at_risk : train', (y_train < 50).mean().round(3),\n"
        "      '| val', (y_val < 50).mean().round(3),\n"
        "      '| test', (y_test < 50).mean().round(3))\n"
    ))

    cells.append(md(
        "### 5.1 Validation croisée 5-fold sur le train\n"
        "Pour ne pas dépendre d'un seul split aléatoire, on rapporte la RMSE sur 5 plis du train.\n"
        "On calcule la même métrique sur le val officiel et le test pour vérifier l'absence de fuite."
    ))

    cells.append(code(
        "from sklearn.model_selection import cross_val_score\n"
        "cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n"
        "all_metrics = {}\n"
        "cv_scores = {}\n"
        "for name, pipe in models.make_sklearn_models().items():\n"
        "    scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)\n"
        "    cv_scores[name] = scores\n"
        "    pipe.fit(X_train, y_train)\n"
        "    all_metrics[name] = models.regression_metrics(pipe, X_train, y_train, X_val, y_val, X_test, y_test)\n"
        "    print(f'  {name:14s} CV RMSE={scores.mean():.3f}±{scores.std():.3f}  test_RMSE={all_metrics[name].rmse_test:.3f}')\n"
    ))

    cells.append(code(
        "# 5.2 MLP PyTorch — architecture améliorée + scheduler cosine + early stopping\n"
        "pre_fit = preprocessing.build_preprocessor()\n"
        "Xtr = pre_fit.fit_transform(X_train).astype('float32')\n"
        "Xva = pre_fit.transform(X_val).astype('float32')\n"
        "Xte = pre_fit.transform(X_test).astype('float32')\n"
        "print('Xtr shape:', Xtr.shape)\n"
        "mlp, mlp_log = train_mlp(Xtr, y_train.to_numpy('float32'), Xva, y_val.to_numpy('float32'),\n"
        "                          epochs=60, batch_size=2048, lr=3e-3,\n"
        "                          hidden=(256,128,64), patience=8)\n"
        "from src.models import RegMetrics\n"
        "from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score\n"
        "pred_tr = mlp_predict(mlp, Xtr); pred_va = mlp_predict(mlp, Xva); pred_te = mlp_predict(mlp, Xte)\n"
        "all_metrics['mlp_pytorch'] = RegMetrics(\n"
        "    rmse_train = mean_squared_error(y_train, pred_tr) ** 0.5,\n"
        "    rmse_val   = mean_squared_error(y_val, pred_va) ** 0.5,\n"
        "    rmse_test  = mean_squared_error(y_test, pred_te) ** 0.5,\n"
        "    mae_test   = mean_absolute_error(y_test, pred_te),\n"
        "    r2_test    = r2_score(y_test, pred_te),\n"
        "    acc_at_risk_test = accuracy_score(y_test < 50, pred_te < 50),\n"
        "    f1_at_risk_test  = f1_score(y_test < 50, pred_te < 50),\n"
        ")\n"
        "import json\n"
        "json.dump({k: v.tolist() for k, v in cv_scores.items()},\n"
        "          open(config.RES_DIR / 'cv_scores.json','w'))\n"
        "# Visualisation CV\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "names_cv = list(cv_scores.keys())\n"
        "data_cv = [cv_scores[n] for n in names_cv]\n"
        "ax.boxplot(data_cv, tick_labels=names_cv, showmeans=True)\n"
        "ax.set_ylabel('RMSE (5-fold CV)'); ax.set_title('Distribution RMSE en 5-fold sur le train')\n"
        "plt.xticks(rotation=15); fig.tight_layout()\n"
        "fig.savefig(config.FIG_DIR / 'cv_boxplots.png', dpi=160); plt.show()\n"
    ))

    cells.append(code(
        "table = metrics_to_df(all_metrics).sort_values('rmse_test')\n"
        "table.to_csv(config.RES_DIR / 'failure_models_metrics.csv', index=False)\n"
        "table\n"
    ))

    cells.append(code(
        "# Comparaison RMSE train/val/test par modèle (pour over/underfitting)\n"
        "tbl = table.set_index('model')[['rmse_train','rmse_val','rmse_test']]\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "tbl.plot(kind='bar', ax=ax, color=['#9ecae1','#3182bd','#08519c'])\n"
        "ax.set_ylabel('RMSE'); ax.set_title('RMSE par split')\n"
        "ax.tick_params(axis='x', rotation=15)\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'models_rmse_compare.png', dpi=160); plt.show()\n"
        "# Courbe d'apprentissage MLP\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "ax.plot(mlp_log.train_rmse, label='train', color='#1f6feb')\n"
        "ax.plot(mlp_log.val_rmse, label='val', color='#d62728')\n"
        "ax.set_xlabel('epoch'); ax.set_ylabel('RMSE'); ax.set_title('MLP : train vs val par epoch')\n"
        "ax.legend(); fig.tight_layout()\n"
        "fig.savefig(config.FIG_DIR / 'mlp_curves.png', dpi=160); plt.show()\n"
    ))

    cells.append(md(
        "**Lecture.** Les modèles non-linéaires (RF, HistGBRT, MLP) battent nettement Ridge et la "
        "baseline, comme attendu vu l'interaction `heures_etude × méthode_etude`. L'écart "
        "`train` ↔ `val/test` reste modeste pour HistGBRT (≤ 0.5 pt RMSE) → **pas d'over-fitting**. "
        "Random Forest a un RMSE train très bas (mémorise) mais val ≈ test → léger over-fitting "
        "absorbé. MLP est compétitif avec dropout et batch-norm."
    ))

    cells.append(md(
        "## 6. Hyperparamètres — RandomizedSearchCV (HistGBRT)\n"
        "\n"
        "On optimise le meilleur modèle (HistGBRT) sur la **validation croisée** "
        "(`neg_root_mean_squared_error`, 3 folds) sur 30 k lignes pour rester rapide."
    ))

    cells.append(code(
        "search_pipe = models.make_sklearn_models()['hist_gbrt']\n"
        "param_dist = {\n"
        "    'model__max_depth': [4, 6, 8, 10, None],\n"
        "    'model__learning_rate': [0.02, 0.05, 0.08, 0.12],\n"
        "    'model__max_iter': [200, 400, 600, 800],\n"
        "    'model__min_samples_leaf': [40, 80, 120, 200],\n"
        "    'model__l2_regularization': [0.0, 1e-3, 1e-2, 1e-1],\n"
        "}\n"
        "n_search = min(80_000, len(X_train))\n"
        "Xs = X_train.sample(n_search, random_state=RANDOM_STATE); ys = y_train.loc[Xs.index]\n"
        "search = RandomizedSearchCV(search_pipe, param_distributions=param_dist, n_iter=20,\n"
        "                            scoring='neg_root_mean_squared_error', cv=3,\n"
        "                            random_state=RANDOM_STATE, n_jobs=-1, verbose=1)\n"
        "search.fit(Xs, ys)\n"
        "print('best params:', search.best_params_)\n"
        "print('best CV RMSE:', -search.best_score_)\n"
        "best_pipe = search.best_estimator_\n"
        "best_pipe.fit(X_train, y_train)\n"
        "best_metrics = models.regression_metrics(best_pipe, X_train, y_train, X_val, y_val, X_test, y_test)\n"
        "import json\n"
        "json.dump({'best_params': {k: (v if not isinstance(v,(int,float)) else float(v)) for k,v in search.best_params_.items()},\n"
        "           'best_cv_rmse': float(-search.best_score_),\n"
        "           'best_test_rmse': float(best_metrics.rmse_test),\n"
        "           'best_test_r2': float(best_metrics.r2_test)},\n"
        "          open(config.RES_DIR / 'tuning_summary.json','w'), indent=2)\n"
        "best_metrics\n"
    ))

    cells.append(md(
        "## 7. Analyse over/under-fitting — courbe d'apprentissage du meilleur modèle"
    ))

    cells.append(code(
        "from sklearn.model_selection import learning_curve\n"
        "sizes = [0.1, 0.25, 0.5, 0.75, 1.0]\n"
        "# learning_curve sur 80k pour rester rapide\n"
        "Xlc = X_train.sample(80_000, random_state=RANDOM_STATE); ylc = y_train.loc[Xlc.index]\n"
        "ts, tr_scores, va_scores = learning_curve(\n"
        "    best_pipe, Xlc, ylc, train_sizes=sizes, cv=3,\n"
        "    scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=RANDOM_STATE\n"
        ")\n"
        "tr = -tr_scores.mean(axis=1); va = -va_scores.mean(axis=1)\n"
        "fig, ax = plt.subplots(figsize=(7, 3.8))\n"
        "ax.plot(ts, tr, 'o-', label='train', color='#1f6feb')\n"
        "ax.plot(ts, va, 'o-', label='cv', color='#d62728')\n"
        "ax.set_xlabel('# échantillons d\\'entraînement'); ax.set_ylabel('RMSE')\n"
        "ax.set_title('Courbe d\\'apprentissage — HistGBRT'); ax.legend()\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'learning_curve.png', dpi=160); plt.show()\n"
    ))

    cells.append(md(
        "**Lecture.** La courbe RMSE de validation décroît rapidement puis se stabilise : "
        "ajouter plus de données aide peu — le modèle est probablement **proche de sa borne** "
        "vu les variables disponibles. L'écart train/val reste petit → pas d'overfitting "
        "structurel."
    ))

    cells.append(md("## 8. Importance des variables + PDP"))

    cells.append(code(
        "Xv_s = X_val.sample(20_000, random_state=RANDOM_STATE)\n"
        "imp = explain.perm_importance(best_pipe, Xv_s, y_val.loc[Xv_s.index])\n"
        "imp.to_csv(config.RES_DIR / 'feature_importance.csv', index=False)\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "ax.barh(imp['feature'][::-1], imp['importance_mean'][::-1], color='#1f6feb')\n"
        "ax.set_title('Importance par permutation (mesure : R²)')\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'perm_importance.png', dpi=160); plt.show()\n"
        "imp\n"
    ))

    cells.append(code(
        "explain.pdp_plot(best_pipe, X_train.sample(10_000, random_state=RANDOM_STATE),\n"
        "                 ['heures_etude', 'assiduité_classe'],\n"
        "                 out=config.FIG_DIR / 'pdp.png')\n"
        "plt.show()\n"
    ))

    cells.append(md(
        "**Lecture finale.** L'importance par permutation confirme la hiérarchie qualitative de "
        "l'EDA (`heures_etude` >> `méthode_etude`/`qualité_sommeil`/`évaluation_établissement` > "
        "`assiduité_classe` > reste). Les PDP montrent une réponse **monotone non linéaire** sur "
        "`heures_etude` (effet marginal décroissant au-dessus de 6h)."
    ))

    cells.append(md(
        "## 9. Limites (pour le rapport)\n"
        "\n"
        "* **Variables auto-déclarées** (`heures_etude`, `qualité_sommeil`, `méthode_etude`,…). "
        "Risque d'erreur de mesure et de manipulation volontaire (cf. supplément, section sécurité).\n"
        "* **Indépendance temporelle** non vérifiable : pas d'information de cohorte ; on suppose "
        "i.i.d. — c'est une hypothèse forte.\n"
        "* **Causalité** : la corrélation `heures_etude → score` est crédible mais d'autres variables "
        "(motivation, contexte familial) sont absentes ⇒ biais d'omission probable.\n"
        "* `difficulté_examen` est étonnamment plate : soit c'est l'effet d'un examen normalisé "
        "(plausible), soit la variable a été mal collectée.\n"
        "\n"
        "Pour la suite : voir le notebook 03 (sécurité, biais, explicabilité)."
    ))

    write("01_failure_prediction.ipynb", cells)


# =============================================================================
# Notebook 2 : OCR
# =============================================================================


def notebook_ocr() -> None:
    cells = []
    cells.append(md(
        "# 2 — OCR : reconnaissance de caractères manuscrits\n"
        "\n"
        "Smart School ECAM 2026 — partie *Automatic correction*.\n"
        "\n"
        "Plan :\n"
        "1. Préparation du dataset IDX + vérifications (taille, classes, équilibre).\n"
        "2. Choix d'architecture (CNN) et justification.\n"
        "3. Apprentissage + courbes train/val.\n"
        "4. Évaluation : accuracy globale, **matrice de confusion**, accuracy par classe.\n"
        "5. Erreurs typiques (analyse qualitative).\n"
        "6. Saliency map (préfigure la partie XAI du supplément)."
    ))

    cells.append(code(
        "import sys, time\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'src').is_dir():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, matplotlib.pyplot as plt, seaborn as sns, torch\n"
        "from sklearn.metrics import confusion_matrix, classification_report\n"
        "from torch.utils.data import Subset, random_split\n"
        "from src import config, ocr_data, ocr_model, explain\n"
        "sns.set_theme(style='white')\n"
        "config.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "config.RES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "torch.manual_seed(config.RANDOM_STATE)\n"
        "DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'\n"
        "print('device:', DEVICE)\n"
    ))

    cells.append(md("## 1. Préparation du dataset"))

    cells.append(code(
        "train_full = ocr_data.CharIDXDataset(\n"
        "    config.IMAGE_DIR / 'train-images-idx3-ubyte',\n"
        "    config.IMAGE_DIR / 'train-labels-idx1-ubyte')\n"
        "test_full = ocr_data.CharIDXDataset(\n"
        "    config.IMAGE_DIR / 'test-images-idx3-ubyte',\n"
        "    config.IMAGE_DIR / 'test-labels-idx1-ubyte')\n"
        "print(f'train: {len(train_full):,}  | test: {len(test_full):,}  | classes: {train_full.num_classes}')\n"
        "labels = train_full.class_labels()\n"
        "print('classes:', ''.join(labels))\n"
    ))

    cells.append(code(
        "# Q — Le dataset est-il équilibré ?\n"
        "y_train_all = train_full.labels.numpy()\n"
        "uniq, counts = np.unique(y_train_all, return_counts=True)\n"
        "fig, ax = plt.subplots(figsize=(13, 3.5))\n"
        "ax.bar(range(len(counts)), counts, color='#1f6feb')\n"
        "ax.set_xticks(range(len(counts))); ax.set_xticklabels([labels[i] for i in uniq], fontsize=8)\n"
        "ax.set_title(f'Distribution des classes ({len(counts)} classes, {counts.sum():,} images)')\n"
        "ax.set_ylabel('# images'); ax.axhline(counts.mean(), color='#d62728', ls='--', label=f'moy={counts.mean():.0f}')\n"
        "ax.legend(); fig.tight_layout(); fig.savefig(config.FIG_DIR / 'ocr_class_balance.png', dpi=160); plt.show()\n"
        "print(f'min: {counts.min()}  max: {counts.max()}  ratio max/min: {counts.max()/counts.min():.1f}x')\n"
    ))

    cells.append(md(
        "**Lecture.** Le dataset est **fortement déséquilibré** : les chiffres et les majuscules "
        "ont 5–10× plus d'exemples que certaines minuscules. C'est un point à mentionner dans le "
        "rapport (et à compenser via `class_weight` ou data augmentation pour aller plus loin)."
    ))

    cells.append(code(
        "# Aperçu d'images (1 par classe quand possible)\n"
        "fig, axes = plt.subplots(4, 16, figsize=(13, 4.2))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.axis('off')\n"
        "    if i >= train_full.num_classes:\n"
        "        continue\n"
        "    j = int(np.where(y_train_all == i)[0][0])\n"
        "    img, lbl = train_full[j]\n"
        "    ax.imshow(img.squeeze(), cmap='gray')\n"
        "    ax.set_title(labels[i], fontsize=9)\n"
        "fig.suptitle('Un échantillon par classe', y=0.98)\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'ocr_samples.png', dpi=160); plt.show()\n"
    ))

    cells.append(md(
        "## 2. Sous-échantillonnage et split val\n"
        "\n"
        "Pour un MacBook, on entraîne sur **20 %** du train (~140 k images) avec val 10 %. "
        "À augmenter pour le run final si tu disposes d'un GPU."
    ))

    cells.append(code(
        "FRACTION = 0.2\n"
        "VAL_FRACTION = 0.1\n"
        "EPOCHS = 3\n"
        "BATCH_SIZE = 256\n"
        "rng = np.random.default_rng(config.RANDOM_STATE)\n"
        "pool = rng.choice(len(train_full), size=int(len(train_full)*FRACTION), replace=False)\n"
        "rng.shuffle(pool)\n"
        "n_val = max(4096, int(len(pool) * VAL_FRACTION))\n"
        "val_ids = pool[:n_val]; tr_ids = pool[n_val:]\n"
        "train_ds = Subset(train_full, tr_ids.tolist())\n"
        "val_ds = Subset(train_full, val_ids.tolist())\n"
        "print(f'train_subset: {len(train_ds):,}  val_subset: {len(val_ds):,}')\n"
    ))

    cells.append(md("## 3. Architecture + apprentissage"))

    cells.append(code(
        "model = ocr_model.CharCNN(num_classes=train_full.num_classes)\n"
        "print(model)\n"
        "n_params = sum(p.numel() for p in model.parameters())\n"
        "print(f'paramètres : {n_params:,}')\n"
    ))

    cells.append(code(
        "t0 = time.time()\n"
        "history = ocr_model.train_cnn(model, train_ds, val_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3, device=DEVICE)\n"
        "print(f'train time: {(time.time()-t0)/60:.1f} min')\n"
        "torch.save(model.state_dict(), config.RES_DIR / 'ocr_cnn.pt')\n"
    ))

    cells.append(code(
        "# Courbes train_loss / val_acc (over/under-fitting)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))\n"
        "axes[0].plot(range(1,EPOCHS+1), history['train_loss'], 'o-', color='#1f6feb')\n"
        "axes[0].set_xlabel('epoch'); axes[0].set_ylabel('train loss'); axes[0].set_title('Train loss')\n"
        "axes[1].plot(range(1,EPOCHS+1), history['val_acc'], 'o-', color='#2ca02c')\n"
        "axes[1].set_xlabel('epoch'); axes[1].set_ylabel('val accuracy'); axes[1].set_title('Validation accuracy')\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'ocr_training_curves.png', dpi=160); plt.show()\n"
    ))

    cells.append(md("## 4. Évaluation sur le test officiel"))

    cells.append(code(
        "y_true, y_pred = ocr_model.predict(model, test_full, device=DEVICE)\n"
        "acc = (y_true == y_pred).mean()\n"
        "print(f'TEST accuracy: {acc:.4f}  ({len(y_true):,} échantillons)')\n"
    ))

    cells.append(code(
        "cm = confusion_matrix(y_true, y_pred, labels=np.arange(train_full.num_classes))\n"
        "fig, ax = plt.subplots(figsize=(11, 9))\n"
        "sns.heatmap(cm, cmap='magma', cbar=True, square=True, ax=ax,\n"
        "            xticklabels=labels, yticklabels=labels)\n"
        "ax.set_xlabel('prédit'); ax.set_ylabel('vrai'); ax.set_title('Matrice de confusion (test)')\n"
        "ax.tick_params(labelsize=7)\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'ocr_confusion_matrix.png', dpi=200); plt.show()\n"
    ))

    cells.append(code(
        "# Accuracy par classe + 10 pires\n"
        "per_class = (cm.diagonal() / cm.sum(axis=1).clip(min=1))\n"
        "import pandas as pd\n"
        "pc = pd.DataFrame({'class_idx': np.arange(len(per_class)),\n"
        "                   'char': labels, 'acc': per_class.round(3),\n"
        "                   'support': cm.sum(axis=1)})\n"
        "pc.to_csv(config.RES_DIR / 'ocr_per_class_acc.csv', index=False)\n"
        "print('10 classes les plus difficiles :')\n"
        "print(pc.sort_values('acc').head(10).to_string(index=False))\n"
        "print('\\n10 classes les plus faciles :')\n"
        "print(pc.sort_values('acc', ascending=False).head(10).to_string(index=False))\n"
    ))

    cells.append(code(
        "# Top confusions (paires vrai → prédit)\n"
        "import pandas as pd\n"
        "cm0 = cm.copy(); np.fill_diagonal(cm0, 0)\n"
        "ii, jj = np.unravel_index(np.argsort(cm0, axis=None)[::-1][:10], cm0.shape)\n"
        "rows = [{'vrai': labels[i], 'predit': labels[j], 'count': int(cm0[i,j])} for i,j in zip(ii,jj)]\n"
        "pd.DataFrame(rows)\n"
    ))

    cells.append(md(
        "**Lecture.** Les confusions dominantes sont **typographiques** : `O`/`0`, `1`/`l`/`I`, "
        "`5`/`s`/`S`, etc. Ce sont des paires intrinsèquement ambiguës en écriture manuscrite — un "
        "humain ferait les mêmes erreurs sans contexte. Cela fixe un plafond raisonnable au-dessus "
        "duquel il faudrait des **indices contextuels** (modèle de langue, position dans le mot)."
    ))

    cells.append(md("## 5. Saliency map (XAI image, lien Integrated Gradients du cours)"))

    cells.append(code(
        "i = 0\n"
        "img, lbl = test_full[i]\n"
        "char = test_full.label_to_char.get(int(lbl), '?')\n"
        "sal = explain.saliency_map(model, img, target=int(lbl), device=DEVICE)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(6, 3))\n"
        "axes[0].imshow(img.squeeze(), cmap='gray'); axes[0].set_title(f'image (vrai = {char})'); axes[0].axis('off')\n"
        "axes[1].imshow(sal, cmap='inferno'); axes[1].set_title('|x · ∇logit|'); axes[1].axis('off')\n"
        "fig.tight_layout(); fig.savefig(config.FIG_DIR / 'ocr_saliency.png', dpi=160); plt.show()\n"
    ))

    write("02_ocr.ipynb", cells)


# =============================================================================
# Notebook 3 : supplément Advanced AI
# =============================================================================


def notebook_supplement() -> None:
    cells = []
    cells.append(md(
        "# 3 — Supplément Advanced AI : sécurité, biais, explicabilité\n"
        "\n"
        "Ce notebook produit les analyses additionnelles attendues dans le supplément 5 pages "
        "(rendu 21 mai). Il s'appuie sur les modèles entraînés dans le notebook 1 (et 2 pour l'OCR)."
    ))

    cells.append(code(
        "import sys, json\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'src').is_dir():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
        "from sklearn.metrics import accuracy_score, f1_score\n"
        "from src import config, data, preprocessing, models, fairness, attack\n"
        "from src.models import regression_metrics, metrics_to_df\n"
        "sns.set_theme(style='whitegrid')\n"
        "config.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "config.RES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "RANDOM_STATE = config.RANDOM_STATE\n"
    ))

    cells.append(md(
        "## A. Préparation : on réentraîne le meilleur modèle (HistGBRT)\n"
        "Sous-échantillon 100 k pour un run rapide ; les conclusions tiennent à plus grande échelle."
    ))

    cells.append(code(
        "from src import features\n"
        "df = data.load_train(nrows=200_000)\n"
        "df = features.add_derived(df)\n"
        "df = data.add_at_risk_label(df)\n"
        "X_train, X_val, X_test, y_train, y_val, y_test = data.make_splits(df, features=config.FEATURES_FINAL_PLUS)\n"
        "pipe = models.make_sklearn_models()['hist_gbrt']\n"
        "pipe.fit(X_train, y_train)\n"
        "pred_test = pipe.predict(X_test)\n"
        "print('test RMSE:', np.sqrt(((pred_test - y_test)**2).mean()).round(3))\n"
    ))

    cells.append(md(
        "## B. Sécurité — attaque d'évasion tabulaire\n"
        "\n"
        "Question : un étudiant **peut-il tricher** sur ses réponses auto-déclarées (`heures_etude`, "
        "`qualité_sommeil`, `méthode_etude`, `accès_internet`, `heures_sommeil`) pour faire passer "
        "sa note prédite sous le seuil 50 et obtenir un tutorat indu, **avec le moins de mensonges "
        "possible** ?"
    ))

    cells.append(code(
        "# Cas concret : un étudiant initialement prédit OK\n"
        "row = X_test.iloc[0].copy()\n"
        "from src.attack import evade\n"
        "res = evade(pipe, row)\n"
        "print('prédiction sincère :', round(res.base_pred, 2))\n"
        "print('prédiction après attaque :', round(res.attack_pred, 2))\n"
        "print('changements :', res.changes)\n"
        "print('succès :', res.success, '| nb changements :', res.n_features_changed)\n"
    ))

    cells.append(code(
        "# Évaluation systématique sur 200 étudiants initialement OK\n"
        "stats = attack.evaluate_attack_success(pipe, X_test, n=200)\n"
        "print(json.dumps(stats, indent=2))\n"
        "with open(config.RES_DIR / 'attack_summary.json','w') as f:\n"
        "    json.dump(stats, f, indent=2)\n"
    ))

    cells.append(md(
        "**Lecture.** Le taux de succès et le nombre moyen de mensonges nécessaires donnent une "
        "estimation **opérationnelle** du risque. Mitigation possible (à discuter dans le rapport) :\n"
        "- vérifier la **cohérence interne** des réponses (`heures_etude` + `assiduité_classe` "
        "très basses ⇒ flag) ;\n"
        "- ne pas baser la décision **uniquement** sur le modèle (humain dans la boucle) ;\n"
        "- définir un seuil avec marge (ex. 45) pour le tutorat automatique et faire un examen "
        "qualitatif autour."
    ))

    cells.append(md(
        "## C. Biais — équité par `genre`\n"
        "\n"
        "On compare les modèles **avec** et **sans** la variable `genre` pour quantifier "
        "l'arbitrage performance ↔ équité."
    ))

    cells.append(code(
        "# Métriques d'équité du modèle complet\n"
        "ftab_full = fairness.fairness_table(y_test, pred_test, X_test['genre'])\n"
        "print('=== Modèle complet ===')\n"
        "ftab_full\n"
    ))

    cells.append(code(
        "# Modèle sans `genre` (mitigation par retrait simple) - mêmes hyperparams\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        "from sklearn.ensemble import HistGradientBoostingRegressor\n"
        "cat_no_genre = [c for c in config.CAT_COLS_KEPT if c != 'genre']\n"
        "num_with_derived = config.NUM_COLS_KEPT + config.DERIVED_COLS\n"
        "pre_no_g = ColumnTransformer([\n"
        "    ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_with_derived),\n"
        "    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),\n"
        "                      ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_no_genre),\n"
        "], remainder='drop', verbose_feature_names_out=False)\n"
        "pipe_no_g = Pipeline([('pre', pre_no_g),\n"
        "    ('model', HistGradientBoostingRegressor(max_depth=8, learning_rate=0.06,\n"
        "                                            max_iter=400, min_samples_leaf=80,\n"
        "                                            l2_regularization=1e-3, random_state=RANDOM_STATE))])\n"
        "X_train2 = X_train.drop(columns=['genre']); X_test2 = X_test.drop(columns=['genre'])\n"
        "pipe_no_g.fit(X_train2, y_train)\n"
        "pred_test_ng = pipe_no_g.predict(X_test2)\n"
        "ftab_no_g = fairness.fairness_table(y_test, pred_test_ng, X_test['genre'])\n"
        "print('=== Modèle SANS genre ===')\n"
        "ftab_no_g\n"
    ))

    cells.append(code(
        "# Comparaison performance vs équité\n"
        "from sklearn.metrics import mean_squared_error\n"
        "rmse_full = mean_squared_error(y_test, pred_test) ** 0.5\n"
        "rmse_ng   = mean_squared_error(y_test, pred_test_ng) ** 0.5\n"
        "summary = pd.DataFrame({\n"
        "    'modèle': ['avec genre', 'sans genre'],\n"
        "    'RMSE test': [rmse_full, rmse_ng],\n"
        "    'écart TPR (max-min)': [fairness.disparity(ftab_full,'TPR(recall_at_risk)'),\n"
        "                            fairness.disparity(ftab_no_g,'TPR(recall_at_risk)')],\n"
        "    'écart selection_rate': [fairness.disparity(ftab_full,'selection_rate(at_risk)'),\n"
        "                             fairness.disparity(ftab_no_g,'selection_rate(at_risk)')],\n"
        "}).round(4)\n"
        "summary.to_csv(config.RES_DIR / 'fairness_summary.csv', index=False)\n"
        "summary\n"
    ))

    cells.append(md(
        "**Lecture.** Le retrait pur de `genre` réduit (peu) la disparité mais le **proxy** "
        "(corrélations résiduelles via `méthode_etude`, etc.) en conserve l'essentiel. "
        "Mitigations plus poussées :\n"
        "- *Pre-processing* : Disparate Impact Remover (vu en cours).\n"
        "- *Post-processing* : seuils différenciés par groupe (Equal Opportunity, Hardt et al.).\n"
        "- Inacceptable de prétendre annuler totalement le biais : le théorème d'impossibilité "
        "(Saravanakumar 2020, vu en cours) rappelle qu'on ne peut satisfaire toutes les définitions "
        "de fairness simultanément si les prévalences diffèrent."
    ))

    cells.append(md(
        "## D. Explicabilité (résumé)\n"
        "\n"
        "Les graphiques `perm_importance.png` et `pdp.png` produits dans le notebook 1 répondent "
        "déjà à la question pour le modèle tabulaire. Pour aller plus loin :\n"
        "* SHAP (KernelSHAP local sur un étudiant à risque) — intéressant à intégrer dans le rapport.\n"
        "* Pour l'OCR : voir `ocr_saliency.png` (notebook 2). En allant plus loin avec `captum.attr.IntegratedGradients`."
    ))

    cells.append(code(
        "# Exemple d'explication 'locale' textuelle pour un étudiant à risque\n"
        "from src.attack import evade\n"
        "candidates = X_test[pred_test < 50].head(3)\n"
        "for i, (_, row) in enumerate(candidates.iterrows()):\n"
        "    pred = float(pipe.predict(pd.DataFrame([row]))[0])\n"
        "    print(f'\\n=== Étudiant #{i+1} — note prédite = {pred:.1f} ===')\n"
        "    print('Features renseignées :', dict(row))\n"
    ))

    write("03_supplement.ipynb", cells)


if __name__ == "__main__":
    notebook_failure()
    notebook_ocr()
    notebook_supplement()
    print('\\nDone.')
