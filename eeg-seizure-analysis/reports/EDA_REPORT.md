# Rapport EDA — EEG et crises épileptiques

## Qualité et structure

- **11,500 segments** de 178 points, sans valeur manquante.
- **500 acquisitions parentes** reconstruites, avec 23 fenêtres par acquisition.
- Cinq classes équilibrées à 2 300 segments chacune.
- La cible binaire « crise » représente **20%** des segments.
- 0 signaux dupliqués exactement.

## Premiers constats

La classe 1 (activité ictale) se distingue surtout par l'amplitude, l'énergie, la longueur de ligne et plusieurs caractéristiques spectrales. La PCA montre la structure globale mais aussi un recouvrement entre classes : une frontière non linéaire est donc pertinente. Les graphiques associés sont dans `reports/figures/` et le tableau des variables dans `reports/engineered_features.csv`.

## Point méthodologique critique

Les lignes ne sont pas indépendantes : chacune des 500 acquisitions originales a été découpée en 23 fenêtres. Toute évaluation ML doit conserver les 23 fenêtres d'une acquisition dans un seul split. Le pipeline de ce projet applique cette séparation par groupe.

## Limites

Ce benchmark monocanal, ancien, très nettoyé et composé de fenêtres courtes ne représente pas la diversité d'un EEG clinique continu. Il sert à démontrer une démarche data science, pas à valider un dispositif de diagnostic.
