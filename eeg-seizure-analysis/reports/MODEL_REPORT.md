# Rapport ML — détection binaire de crise

## Protocole

- Cible : classe 1 contre classes 2 à 5.
- Variables : 28 caractéristiques temporelles et spectrales interprétables.
- Split : 80/20 stratifié au niveau des 500 acquisitions parentes (aucune fenêtre sœur entre train et test).
- Comparaison : Dummy, régression logistique, Random Forest et Extra Trees.
- Sélection : meilleure Average Precision en validation croisée groupée à 4 plis sur le train.
- Évaluation finale : holdout de 20 % jamais utilisé pour choisir le modèle, seuil fixé à 0,50.

## Meilleur modèle : `extra_trees`

- Average Precision en validation croisée : **0.990**
- ROC AUC : **0.998**
- Average Precision : **0.992**
- Sensibilité / rappel : **0.998**
- Spécificité : **0.960**
- Précision : **0.861**
- F1 : **0.924**
- Balanced accuracy : **0.979**

## Lecture responsable

Ces scores mesurent la discrimination sur un benchmark nettoyé, pas la performance clinique sur de nouveaux patients. Les fenêtres partagent une provenance limitée à 500 acquisitions, et l'identité patient n'est pas fournie explicitement. Une validation externe, prospective et multi-centres serait nécessaire avant tout usage médical.
