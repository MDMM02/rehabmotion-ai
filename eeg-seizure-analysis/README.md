# EEG Seizure Lab

Projet portfolio de data science consacré à l'exploration de signaux EEG et à la classification binaire **crise / hors crise**, à partir du dataset Kaggle [Epileptic Seizure Recognition](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition).

Le projet fournit :

- une EDA reproductible et des graphiques exportés ;
- 28 caractéristiques temporelles et fréquentielles interprétables ;
- quatre modèles comparés en validation croisée groupée, puis évalués sur un holdout sans fuite ;
- une application Streamlit interactive ;
- des rapports EDA et ML générés automatiquement ;
- des tests unitaires du pipeline.

> **Avertissement médical** — démonstrateur éducatif et R&D uniquement. Il ne s'agit pas d'un dispositif médical et il ne doit pas être utilisé pour le diagnostic, la surveillance, le traitement ou une décision clinique.

## Pourquoi le split par acquisition compte

Le CSV contient 11 500 fenêtres de 178 points, mais elles proviennent de seulement 500 acquisitions originales, chacune découpée en 23 morceaux. Un split aléatoire par ligne placerait très probablement des morceaux du même enregistrement dans le train et le test et produirait un score trop optimiste.

Ce projet reconstruit donc l'identifiant de l'acquisition parente et effectue un split stratifié 80/20 **au niveau des 500 acquisitions**.

## Structure

```text
eeg-seizure-analysis/
├── analysis/
│   ├── 01_eda.py                 # fichier à cellules EDA
│   └── 02_ml.py                  # fichier à cellules ML
├── eeg_seizure/
│   ├── data.py                   # chargement, validation et groupes
│   ├── features.py               # variables temps/fréquence
│   └── modeling.py               # split, modèles et métriques
├── scripts/
│   ├── download_data.py
│   ├── run_eda.py
│   └── train_model.py
├── reports/                      # rapports, tableaux et figures
├── artifacts/                    # modèle joblib local
├── tests/
├── app.py
└── requirements.txt
```

Les fichiers `analysis/*.py` utilisent les marqueurs `# %%` et peuvent être ouverts comme notebooks dans VS Code ou Jupyter. Les scripts de `scripts/` sont les versions entièrement reproductibles qui exportent les résultats.

## Installation

Depuis ce dossier :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
```

Le dataset a aussi déjà été téléchargé localement dans `data/raw/` dans cet espace de travail. Les données brutes sont ignorées par Git afin de ne pas redistribuer le fichier Kaggle.

## Reproduire l'analyse

```bash
python scripts/run_eda.py
python scripts/train_model.py
```

Ces commandes créent notamment :

- `reports/EDA_REPORT.md` et `reports/MODEL_REPORT.md` ;
- `reports/model_comparison.csv` et `reports/test_predictions.csv` ;
- les figures dans `reports/figures/` ;
- `artifacts/seizure_model.joblib`.

## Lancer Streamlit

```bash
streamlit run app.py
```

L'application permet d'explorer un segment, son spectre, les distributions de variables, les métriques du meilleur modèle, son importance des variables et des prédictions d'exemple.

## Méthode ML

La cible binaire est `y == 1` (crise) contre `y ∈ {2, 3, 4, 5}` (hors crise). Les modèles comparés sont :

- Dummy selon la prévalence ;
- régression logistique pondérée ;
- Random Forest pondérée ;
- Extra Trees pondéré.

Le critère principal est l'**Average Precision**, adapté à la cible binaire déséquilibrée (20 % de crise). Le modèle est choisi par validation croisée groupée à quatre plis sur le train, puis mesuré une seule fois sur le holdout de 20 %. Le rapport présente aussi ROC AUC, sensibilité, spécificité, précision, F1 et balanced accuracy.

## Limites

- Les enregistrements sont monocanaux, courts, anciens et pré-nettoyés.
- Les modalités et populations diffèrent entre les classes.
- Le CSV ne fournit pas une identité patient exploitable ; le split par acquisition n'est donc pas une preuve de généralisation à de nouveaux patients.
- Les bandes fréquentielles sont des résumés exploratoires sur des fenêtres d'environ une seconde, pas une analyse EEG clinique complète.
- Toute validation médicale exigerait des données externes, continues, multi-canaux, représentatives et annotées cliniquement.

## Tests

```bash
pytest -q
```

## Source

Le dataset Kaggle est une version restructurée du jeu de données EEG de Bonn : les signaux originaux de 4 097 points ont été découpés en 23 fenêtres de 178 points. Le projet conserve la fréquence d'échantillonnage documentée de 173,61 Hz pour les variables spectrales.
