# Guide pour l'équipe : refaire ce qu'Aziz a fait (heart, D1-D4)

Le pipeline est commun : chacun **ajoute seulement ses datasets**. On ne touche pas à
`src/config.py`, `src/models.py` ni `src/evaluate.py` sans accord du groupe.
L'exemple complet à suivre : branche `feature/heart-aziz`.

## Étapes (CRISP-DM)

1. **Branche** : `git checkout main && git pull && git checkout -b feature/<maladie>-<prénom>`
2. **Déclarer ses datasets** : copier `src/datasets/_template.py` vers `src/datasets/<maladie>.py`.
   Pour chaque dataset, renseigner `files`, `target`, `expected` (= tableau 2 de l'article) et l'URL Kaggle/UCI.
   Si la cible est du texte, la convertir en 0/1 dans `transform` (1 = malade).
   Puis l'ajouter dans `src/datasets/__init__.py` (comme `heart`).
3. **Données** : `python download_data.py D5 D6 D7`, ou mettre le CSV à la main dans `data/raw/<maladie>/D5.csv`.
   La commande affiche `OK` ou `DIFFÉRENT` par rapport au tableau 2. Un `DIFFÉRENT` doit être expliqué dans le rapport.
4. **Data Understanding** : un notebook par dataset dans `notebooks/<maladie>/`, en copiant `notebooks/heart/01_eda_D1.ipynb`.
   À vérifier : doublons, valeurs manquantes ou impossibles (ex. 0 en cholestérol), déséquilibre des classes,
   corrélations, et **fuite de données** (une colonne connue seulement après le diagnostic, comme `time` sur D4).
5. **Test rapide** : `python run_experiments.py --datasets D5 D6 D7 --quick`
6. **Protocole complet** : `python run_experiments.py --datasets D5 D6 D7 --seeds 0 1 2 3 4 42`,
   puis `python seed_analysis.py <maladie>` et `python compare_article.py <maladie>`.
7. **Se rapprocher de l'article** (l'article ne donne ni le split ni la seed) :
   ```
   python calibrate_protocol.py --datasets D5 D6 D7 --seeds 30
   python run_experiments.py --datasets D5 --test-size <ts> --seeds <s1> <s2> <s3> --tag _calibrated
   python compare_article.py <maladie> --suffix _calibrated --best
   ```
   Règle : **un seul** couple (test_size, seed) par dataset, commun aux 15 modèles. Jamais une seed par modèle.
8. **Analyse** : écrire `report/<maladie>_analysis.md` sur le modèle de `report/heart_analysis.md`
   (données, résultats, écarts avec l'article, critique).
9. **Push** : `git add -f results/<maladie>_* report/ notebooks/<maladie>/ src/datasets/<maladie>.py`,
   commit, `git push -u origin <branche>`, puis Pull Request vers `main`.

## Points importants découverts sur heart

- **Moyenne des métriques** : pour la famille bagging et le classical boosting, l'article calcule
  precision/recall/F1 sur la classe positive. Pour les autres boostings et les stackings, il utilise le
  « weighted avg ». C'est déjà géré automatiquement (`ARTICLE_WEIGHTED_MODELS`, colonnes `*_a`).
- **Doublons** : ils gonflent les scores (D3 = D2 copié 3 fois, accuracy 98,6 % → 76,9 % sans doublons).
  Tester avec `--dedup`.
- **Fuites** : tester avec `--drop <colonne>`.
- L'article contient des valeurs incohérentes (ex. D4 classical stacking 99 %). Ne pas chercher à les
  « atteindre » : les signaler dans la partie critique.

## Par membre

| Membre | Datasets | Attention |
|---|---|---|
| Fedi (foie) | D5-D7 | Cible ILPD souvent codée 1/2 : convertir en 1 = malade dans `transform` |
| Omar (diabète) | D8-D11 | Pima : les 0 en glucose, tension, BMI et insuline sont des valeurs manquantes |
| Rami (rein) | D12-D14 | CKD : texte avec espaces/tabulations (`"ckd\t"`), beaucoup de manquants. L'exemple est dans `_template.py` |
| Khalil, Elyesse (peau) | D15-D16 | Multiclasse : `expected = (lignes, None, None)`. Métriques pondérées, AUC un-contre-tous (déjà géré) |

À la fin : `python integrate.py` fusionne tous les résultats dans `results/summary/` (RPI, best count) pour le rapport commun.
