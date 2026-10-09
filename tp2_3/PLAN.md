# TP 2.3 – Classification automatique de textes (activité notée)

> Document de cadrage : choix du cas d'usage, justification, puis plan de réalisation du notebook.

---

## 1. Cas d'usage retenu

### Routage automatique des réclamations clients dans le secteur bancaire / financier

**Problème métier.** Une banque ou un organisme de crédit reçoit chaque jour des milliers de réclamations
rédigées librement par ses clients (formulaire web, e-mail). Chaque réclamation doit être transmise au bon
service (cartes de crédit, prêts immobiliers, recouvrement de dettes, comptes bancaires, rapports de
crédit…). Aujourd'hui ce tri est souvent manuel : il est lent, coûteux et source d'erreurs, ce qui retarde
la réponse au client et peut entraîner des sanctions réglementaires.

**Tâche NLP.** Classification **multi-classes** supervisée : à partir du texte libre de la réclamation
(`consumer_complaint_narrative`), prédire la **catégorie de produit** concernée (`product`).

**Jeu de données.** *Consumer Complaint Database* du **CFPB** (Consumer Financial Protection Bureau, agence
fédérale américaine).

| Élément | Détail |
|---|---|
| **Version retenue** | Kaggle `cfpb/us-consumer-finance-complaints` (publiée par le CFPB, fichier `consumer_complaints.csv`, 2011–2016) : https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints |
| Source d'origine | https://www.consumerfinance.gov/data-research/consumer-complaints/ |
| Licence | Données publiques du gouvernement américain (domaine public) |
| Variable d'entrée | `consumer_complaint_narrative` (texte libre, anglais, anonymisé : `XXXX`) |
| Variable cible | `product` (11 produits, regroupés en 6 classes métier) |
| Volume | ≈ 555 000 réclamations dont ≈ 66 000 avec texte → échantillon stratifié de 50 000 lignes |

### Pourquoi ce cas d'usage ? (justification)

1. **Valeur métier claire et mesurable.** Le routage automatique réduit directement le délai de traitement
   et la charge des équipes. On peut traduire les métriques (précision par classe, matrice de confusion)
   en conséquences concrètes (« X % des réclamations seraient mal aiguillées »). Le domaine **financier** est
   explicitement cité dans la consigne.
2. **Données réelles, publiques et volumineuses.** Il s'agit de vraies plaintes de consommateurs, pas d'un
   jeu « jouet ». Le volume permet un entraînement robuste et un jeu de test fiable, et la licence
   (domaine public) ne pose aucun problème de réutilisation ni de livraison du jeu de données.
3. **Différent des exemples du TP 2.1 et 2.2.** TP 2.1 = classification binaire de sentiment, TP 2.2 =
   classification de cépage. Ici on traite une classification **thématique multi-classes** sur des textes
   **longs** et **bruités**, ce qui permet d'aller plus loin plutôt que de reproduire les exemples.
4. **Riche pour la partie « exploration / nettoyage / préparation » (60 % de la note).** Le jeu présente de
   vrais défis de préparation qui donnent matière à des choix argumentés :
   - beaucoup de lignes sans texte (à filtrer) ;
   - jetons d'anonymisation `XXXX`, montants `{$100.00}`, dates masquées ;
   - libellés de produits qui ont changé au fil des années (ex. « Credit reporting » vs
     « Credit reporting, credit repair services, or other personal consumer reports ») → **regroupement
     de classes** à justifier ;
   - **déséquilibre de classes** important → stratification, pondération, métrique macro ;
   - doublons et textes très courts / très longs.
5. **Bien adapté aux modèles vus en cours, avec une comparaison naturelle (bonus).** Les textes sont
   longs et riches en vocabulaire spécifique (« foreclosure », « APR », « collection agency »…), ce qui se
   prête très bien à la vectorisation **TF-IDF** combinée à un modèle linéaire (**SVM linéaire**, référence
   reconnue en classification de texte), avec **Naive Bayes** comme baseline. Une comparaison avec un
   modèle à base d'**embeddings** (Word2Vec/GloVe ou DistilBERT) est possible en bonus.
6. **Interprétabilité.** Avec un modèle linéaire sur TF-IDF, on peut extraire les mots/n-grammes les plus
   discriminants par classe : cela permet d'**interpréter** les résultats, exigence explicite de la consigne.

### Alternatives envisagées et écartées

| Cas d'usage | Raison de l'écarter |
|---|---|
| Détection de spam SMS (UCI) | Trop simple (≈ 98 % d'accuracy immédiate), peu de matière à interprétation. |
| Analyse de sentiment (IMDB, Amazon, Financial PhraseBank) | Trop proche du TP 2.1 (classification binaire / sentiment). |
| Spécialité médicale à partir de comptes rendus (MTSamples) | Intéressant mais petit (≈ 5 000 textes) et labels très ambigus (« Surgery » chevauche les autres spécialités), résultats difficiles à interpréter. |
| Détection de fake news (LIAR) | Tâche très difficile en NLP « classique » (énoncés courts, ~25 % d'accuracy sur 6 classes), risque d'un modèle peu exploitable. |

---

## 2. Choix des techniques et du modèle (et pourquoi)

| Étape | Choix | Justification |
|---|---|---|
| Nettoyage | minuscules, suppression `XXXX`/montants/dates masqués, ponctuation, chiffres, espaces multiples | Ces éléments sont du bruit d'anonymisation, non porteurs de sens pour le produit. |
| Stop words | liste NLTK anglaise, **en conservant** les négations (`not`, `no`) | Mots très fréquents peu discriminants ; les négations peuvent porter du sens. |
| Tokenisation | au niveau des mots (NLTK `word_tokenize` / regex) | Les textes sont en anglais standard ; la tokenisation mot est adaptée à TF-IDF. |
| Normalisation | **lemmatisation** (WordNet) plutôt que stemming | Conserve des mots lisibles pour l'interprétation (« charged » → « charge »). |
| Vectorisation | **TF-IDF** unigrammes + bigrammes (`min_df`, `max_df`, `sublinear_tf`) | Pondère les termes spécifiques à une catégorie ; les bigrammes capturent « credit report », « late fee »… |
| Baseline | **Multinomial Naive Bayes** | Rapide, classique en classification de texte (vu en TP 2.1), sert de point de référence. |
| Modèle principal | **SVM linéaire (LinearSVC)** avec `class_weight="balanced"` | Excellent en haute dimension et sur données creuses (TF-IDF), robuste, rapide, interprétable via ses coefficients. Gère le déséquilibre grâce à la pondération. |
| Bonus (2e famille) | **Régression logistique** (+ éventuellement **DistilBERT** fine-tuné ou embeddings moyens Word2Vec/GloVe) | Comparer un modèle « sac de mots » à un modèle exploitant le contexte et justifier lequel est le meilleur ici. |

> Note sur le Naive Bayes **gaussien** utilisé dans le TP 2.1 : il suppose des variables continues
> gaussiennes, hypothèse inadaptée à des comptes de mots ; le Naive Bayes **multinomial** est la variante
> théoriquement adaptée à du texte. Ce point sera expliqué dans le notebook.

---

## 3. Plan du notebook (livrable)

Le notebook `tp2_3/TP2_3_classification_reclamations.ipynb` suivra la structure ci-dessous. **Chaque cellule de
code sera précédée d'une cellule Markdown d'explication et suivie d'une interprétation des sorties**
(exigence obligatoire de la consigne).

### 0. Introduction
- Contexte, cas d'usage, objectif métier, présentation du jeu de données (source, licence, colonnes utiles).
- Démarche globale (pipeline NLP) et critères de succès (macro-F1 cible ≥ 0,80).

### 1. Revue de littérature (brève)
- Prétraitement, tokenisation, stop words, stemming vs lemmatisation.
- Représentations : Bag-of-Words, TF-IDF, word embeddings (Word2Vec, GloVe), embeddings contextuels (BERT).
- Modèles : Naive Bayes, SVM, régression logistique, RNN/CNN, Transformers ; leurs forces/limites en classification de texte.

### 2. Chargement et exploration des données — *partie 60 %*
- Chargement du CSV (colonnes `product`, `sub_product`, `issue`, `consumer_complaint_narrative`, `date_received`).
- Dimensions, types, valeurs manquantes (part de réclamations sans texte).
- Distribution des classes (graphique en barres) → mise en évidence du déséquilibre.
- Longueur des textes (nb de mots) : histogramme, boxplot par classe.
- Mots les plus fréquents par classe / nuages de mots.
- Interprétation : quelles classes semblent faciles / difficiles à séparer ?

### 3. Nettoyage et préparation — *partie 60 %*
- Filtrage des lignes sans narratif, suppression des doublons.
- **Regroupement des libellés** de produits historiques en ~6 classes cohérentes (table de correspondance justifiée), suppression des classes trop rares.
- Échantillonnage stratifié (~50 000 réclamations) pour garder des temps de calcul raisonnables.
- Fonction de nettoyage : minuscules, suppression `XXXX`, montants, URL, ponctuation, chiffres.
- Tokenisation, suppression des stop words, lemmatisation ; exemples avant / après.
- Statistiques après nettoyage (taille du vocabulaire, longueur moyenne).
- Séparation **train / test stratifiée (80 / 20)**, `random_state` fixé pour la reproductibilité.

### 4. Vectorisation — *partie 25 %*
- TF-IDF (uni + bigrammes), choix des hyperparamètres (`max_features`, `min_df`, `max_df`, `sublinear_tf`).
- Ajustement **uniquement sur le train** (éviter la fuite de données), transformation du test.
- Dimensions de la matrice, sparsité, exemples de termes au poids élevé.

### 5. Modélisation — *partie 25 %*
- 5.1 Baseline : Multinomial Naive Bayes.
- 5.2 Modèle principal : LinearSVC (`class_weight="balanced"`), recherche d'hyperparamètres (`C`) par **GridSearchCV** (validation croisée stratifiée 5 plis, score macro-F1) dans un `Pipeline` scikit-learn.
- 5.3 Bonus : régression logistique et/ou DistilBERT / embeddings GloVe moyennés.
- Fonction de prédiction sur une nouvelle réclamation rédigée à la main (démonstration du routage).

### 6. Évaluation — *partie 15 %*
- Métriques : accuracy, **précision / rappel / F1 par classe**, **macro-F1** (métrique principale car classes déséquilibrées), weighted-F1.
- **Matrice de confusion** normalisée (heatmap) et analyse des confusions (ex. « Debt collection » ↔ « Credit reporting »).
- Tableau comparatif des modèles (scores + temps d'entraînement).
- Interprétabilité : top n-grammes par classe d'après les coefficients du SVM.
- Analyse d'erreurs : exemples de réclamations mal classées et explication.

### 7. Conclusion
- Synthèse des résultats et du meilleur modèle (et **pourquoi** il est le meilleur dans ce cas).
- Limites (labels bruités, chevauchement thématique, anglais uniquement, textes anonymisés).
- Pistes d'amélioration (Transformers, prise en compte de `Sub-product`/`Issue`, seuil de confiance pour renvoyer les cas incertains à un humain).

---

## 4. Organisation du dépôt

```
tp2_3/
├── PLAN.md                                   # ce document
├── TP2_3_classification_reclamations.ipynb   # notebook livrable
├── requirements.txt                          # dépendances (pandas, scikit-learn, nltk, matplotlib, seaborn, kagglehub)
└── data/
    ├── consumer_complaints.csv               # fichier brut Kaggle (non versionné, téléchargé par le notebook)
    └── complaints_prepared.csv.gz            # jeu préparé livré (généré par le notebook)
```

## 5. Étapes de réalisation

1. Télécharger le jeu de données Kaggle `cfpb/us-consumer-finance-complaints` (automatique via `kagglehub`) ; le notebook produit `data/complaints_prepared.csv.gz`.
2. Rédiger les sections 0–1 (introduction, revue de littérature).
3. Implémenter et commenter l'exploration (2) puis le nettoyage / la préparation (3).
4. Implémenter la vectorisation (4) et les modèles (5).
5. Évaluer, comparer, interpréter (6) ; rédiger la conclusion (7).
6. Relecture : vérifier que **chaque étape** contient code + explication + interprétation, exécuter le notebook de bout en bout (« Restart & Run All ») avant le dépôt sur BOOSTCAMP.

## 6. Risques identifiés et parades

| Risque | Parade |
|---|---|
| Fichier source très volumineux (> 1 Go) | Lecture par morceaux (`chunksize`) / colonnes utiles seulement, échantillon stratifié sauvegardé. |
| Déséquilibre de classes | Stratification, `class_weight="balanced"`, macro-F1 comme métrique principale. |
| Libellés incohérents dans le temps | Table de regroupement explicite et justifiée. |
| Fuite de données | Vectoriseur ajusté sur le train uniquement, dans un `Pipeline`. |
| Temps de calcul (BERT) | Bonus limité à un sous-échantillon et exécuté sur GPU (Google Colab). |
