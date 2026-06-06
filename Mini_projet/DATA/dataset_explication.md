# Dataset Synthétique — Structure et Explication

## Informations générales

| Propriété | Valeur |
|-----------|--------|
| Nombre de lignes | 1 000 observations |
| Nombre de colonnes | 49 variables |
| Valeurs manquantes | 0 |
| Groupes de variables | 4 |

## Les lignes

Chaque **ligne** représente une **observation d'entreprise** (une société à un instant t). Elle regroupe l'ensemble des caractéristiques financières internes de l'entreprise, ses scores synthétiques, le contexte macroéconomique de son environnement et ses performances de l'année précédente.

Ce dataset est typique d'un modèle de **scoring ou de prédiction financière** (risque de crédit, prédiction de défaut, notation interne), combinant données micro-financières, scores comportementaux et contexte macroéconomique.

---

## Les colonnes — 4 groupes

### Groupe 1 — Ratios financiers internes (`Ratio_1` à `Ratio_24`)

> **Plage de valeurs :** −0.50 à +3.00 | **Moyenne :** ≈ 1.25 | **Écart-type :** ≈ 1.00

24 ratios financiers calculés à partir des états financiers de l'entreprise (bilan, compte de résultat). La distribution est asymétrique à droite, typique de données financières normalisées.

| Colonnes | Interprétation probable |
|----------|------------------------|
| `Ratio_1` à `Ratio_8` | Ratios de **liquidité, solvabilité et structure du bilan** (ex : current ratio, quick ratio, levier financier) |
| `Ratio_9` à `Ratio_16` | Ratios de **rentabilité et d'efficacité opérationnelle** (ex : ROE intermédiaire, rotation des actifs) |
| `Ratio_17` à `Ratio_24` | Ratios de **couverture, de croissance et de flux de trésorerie** (ex : DSCR, croissance du chiffre d'affaires) |

---

### Groupe 2 — Scores synthétiques internes (9 variables)

> **Plage de valeurs :** 0 à 1 | **Moyenne :** ≈ 0.50 | **Écart-type :** ≈ 0.29

Indicateurs composites agrégés et normalisés, probablement issus d'un modèle de scoring interne. Chaque score synthétise plusieurs dimensions de la performance ou du risque.

| Colonne | Signification | Statistiques |
|---------|--------------|-------------|
| `NCG` | **Besoin en Capitaux de Gestion** — besoin de financement du cycle d'exploitation | moy : 0.49 · std : 0.28 |
| `OCG` | **Origine des Capitaux de Gestion** — source de financement du cycle opérationnel | moy : 0.50 · std : 0.28 |
| `CLCC` | **Capacité de remboursement / Couverture des charges** — solidité face aux engagements financiers | moy : 0.50 · std : 0.28 |
| `OCS` | **Organisation et Contrôle de la Structure** — efficacité de la structure organisationnelle | moy : 0.50 · std : 0.30 |
| `QPT` | **Qualité des Processus et Transactions** — fiabilité opérationnelle interne | moy : 0.49 · std : 0.28 |
| `QOFF` | **Qualité de l'Offre** — compétitivité et attractivité des produits/services | moy : 0.48 · std : 0.28 |
| `LYCA` | **Loyauté Client / Ancrage** — fidélité de la clientèle, risque de churn | moy : 0.50 · std : 0.29 |
| `IAICOC` | **Indice d'Adéquation Infrastructure / Coût d'Opportunité** — efficience des investissements | moy : 0.50 · std : 0.29 |
| `ROA2bond` | **Écart ROA vs taux obligataire** — prime de performance par rapport au taux sans risque | moy : 0.51 · std : 0.29 |

---

### Groupe 3 — Indicateurs macroéconomiques (10 variables)

> **Plage de valeurs :** −5% à +10% | **Moyenne :** ≈ 2–3% | **Écart-type :** ≈ 4.3%

Variables décrivant l'environnement économique **externe** de l'entreprise. Elles capturent le contexte conjoncturel national ou global au moment de l'observation.

| Colonne | Signification | Statistiques |
|---------|--------------|-------------|
| `Unemployment` | **Taux de chômage** — contexte du marché du travail | moy : 2.25% · std : 4.47% |
| `GDP_growth` | **Croissance du PIB** — dynamisme de l'économie nationale | moy : 2.23% · std : 4.33% |
| `Inflation` | **Taux d'inflation** — pression sur les coûts et le pouvoir d'achat | moy : 2.75% · std : 4.22% |
| `Bond10Y` | **Rendement obligataire à 10 ans** — taux sans risque de référence | moy : 2.59% · std : 4.36% |
| `Interest_rate` | **Taux directeur de la banque centrale** — coût du refinancement | moy : 2.23% · std : 4.34% |
| `Exchange_rate` | **Taux de change** — exposition aux risques de change | moy : 2.86% · std : 4.34% |
| `Commodity_index` | **Indice des matières premières** — pression sur les intrants | moy : 2.62% · std : 4.29% |
| `Consumer_sentiment` | **Indice de confiance des consommateurs** — demande future anticipée | moy : 2.25% · std : 4.29% |
| `Fiscal_balance` | **Solde budgétaire** — stabilité des finances publiques | moy : 2.50% · std : 4.32% |
| `Trade_balance` | **Balance commerciale** — compétitivité externe du pays | moy : 2.51% · std : 4.31% |

---

### Groupe 4 — Performances passées à t−1 (6 variables)

> **Plage de valeurs :** −20% à +30% | **Moyenne :** ≈ 4–5% | **Écart-type :** ≈ 14.5%

Indicateurs de performance de la **période précédente (t−1)**, utilisés comme variables laggées (décalées d'un an) pour la prédiction. Ils permettent au modèle de tenir compte de la trajectoire historique de l'entreprise.

| Colonne | Signification | Statistiques |
|---------|--------------|-------------|
| `ROA_t1` | **Return on Assets à t−1** — rentabilité des actifs l'année précédente | moy : 4.3% · std : 14.6% |
| `ROE_t1` | **Return on Equity à t−1** — rentabilité des capitaux propres l'année précédente | moy : 4.5% · std : 14.6% |
| `NetMargin_t1` | **Marge nette à t−1** — part du chiffre d'affaires restant après toutes charges | moy : 5.1% · std : 14.6% |
| `OperatingMargin_t1` | **Marge opérationnelle à t−1** — efficacité de l'activité principale avant charges financières | moy : 4.9% · std : 14.4% |
| `CashRatio_t1` | **Ratio de liquidité immédiate à t−1** — capacité à honorer les dettes à très court terme | moy : 5.2% · std : 14.7% |
| `OCG_t1` | **Origine des Capitaux de Gestion à t−1** — version laggée de la variable `OCG` (groupe 2) | moy : 4.5% · std : 14.5% |

---

## Récapitulatif des 4 groupes

| Groupe | Variables | Plage | Rôle dans le modèle |
|--------|-----------|-------|---------------------|
| Ratios financiers | `Ratio_1` → `Ratio_24` | −0.5 à 3.0 | Santé financière interne (liquidité, rentabilité, levier) |
| Scores synthétiques | `NCG`, `OCG`, `CLCC`, `OCS`, `QPT`, `QOFF`, `LYCA`, `IAICOC`, `ROA2bond` | 0 à 1 | Indicateurs composites normalisés (qualité, efficience, fidélité) |
| Indicateurs macro | `Unemployment` → `Trade_balance` | −5% à +10% | Environnement économique externe (chômage, PIB, taux) |
| Performances passées | `ROA_t1` → `OCG_t1` | −20% à +30% | Résultats de l'année t−1, variables laggées |

---

*Source : dataset_synthetique.csv — 1 000 observations · 49 variables · 0 valeur manquante*
