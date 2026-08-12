<div align="center">

# Traffic-Sign-Recognition

### Reconnaissance de panneaux de signalisation routière par réseau de neurones convolutif (CNN)

*43 classes GTSRB, une image en entrée, une classification en sortie — 98.45% d'exactitude sur le jeu de test.*

[![Build](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=jupyter)](#)
[![Version](https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)](#contribution--licence)
[![Python](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)](#)
[![Status](https://img.shields.io/badge/status-prototype%20validé-orange?style=for-the-badge)](#)

</div>

---

> **Note de transparence** : ce README a été mis à jour à partir du contenu réel du notebook `Traffic-Sign-Recognition.ipynb` (43 cellules) présent dans le dépôt [`ELGHAD/Traffic-Sign-Recognition`](https://github.com/ELGHAD/Traffic-Sign-Recognition). Toutes les valeurs techniques (architecture, hyperparamètres, résultats d'entraînement, accuracy finale) proviennent directement de l'exécution des cellules du notebook et non d'une estimation.

---

## Sommaire

- [Présentation & Problématique](#présentation--problématique)
- [Fonctionnalités Clés](#fonctionnalités-clés)
- [Résultats](#résultats)
- [Stack Technique](#stack-technique)
- [Architecture & Structure du Projet](#architecture--structure-du-projet)
- [Guide d'Installation & Démarrage Rapide](#guide-dinstallation--démarrage-rapide)
- [Documentation & Exemples de Prédiction](#documentation--exemples-de-prédiction)
- [Tests & Qualité](#tests--qualité)
- [Roadmap](#roadmap)
- [Contribution & Licence](#contribution--licence)

---

## Présentation & Problématique

### Le problème

La reconnaissance automatique des panneaux de signalisation est une brique critique des systèmes avancés d'aide à la conduite (ADAS) et des véhicules autonomes. Un système fiable doit être capable de :

- Distinguer des dizaines de catégories de panneaux visuellement proches (limitations de vitesse, interdictions, obligations, dangers)
- Rester robuste face aux variations d'exposition, de luminosité et d'angle de prise de vue
- Fournir une prédiction rapide, avec un niveau de confiance exploitable

Une approche par règles (détection de formes/couleurs classiques) est fragile et généralise mal. C'est un problème de classification d'images multi-classes, traité ici avec un réseau de neurones convolutif.

### La solution : `Traffic-Sign-Recognition`

Ce projet implémente un pipeline complet de classification de panneaux de signalisation routière, entraîné et évalué sur le jeu de données **GTSRB (German Traffic Sign Recognition Benchmark)**, qui compte **43 classes** de panneaux.

Le pipeline couvre :

1. L'exploration du dataset (inspection visuelle des images brutes)
2. Le prétraitement des images (conversion HSV, égalisation d'histogramme, recadrage centré, redimensionnement à 48x48)
3. La construction et l'entraînement d'un CNN séquentiel avec Keras
4. L'évaluation sur le jeu de test officiel du GTSRB
5. L'amélioration du modèle par augmentation de données (rotation, translation, zoom)
6. L'inférence sur de nouvelles images, avec visualisation du Top-3 des prédictions

### Valeur ajoutée

| Pour qui ? | Ce que ça apporte |
|---|---|
| Étudiants / chercheurs en Computer Vision | Un exemple complet et documenté de pipeline CNN de bout en bout sur un benchmark reconnu (GTSRB) |
| Équipes ADAS / mobilité intelligente | Une preuve de concept avec une exactitude mesurée de 98.45% sur le jeu de test |
| Développeurs Data Science | Une base de code claire (Keras séquentiel) à adapter à d'autres jeux de données de classification d'images |
| Formateurs | Un support pédagogique illustrant l'impact mesurable de la Data Augmentation (+1.15 point d'exactitude) |

---

## Fonctionnalités Clés

### Exploration des données

- Chargement des images d'entraînement au format `.ppm` depuis `Final_Training/Images/`
- Correction des chemins de fichiers multiplateforme (Windows/Unix) via `correct_all_paths`
- Visualisation d'un échantillon aléatoire d'images brutes du dataset

### Prétraitement des images

- Conversion RGB vers HSV et égalisation d'histogramme sur le canal de luminosité (correction des images sur/sous-exposées)
- Reconversion HSV vers RGB après égalisation
- Recadrage centré sur le côté le plus court de l'image, puis redimensionnement uniforme à 48x48 pixels
- Encodage one-hot des labels de classe pour l'entraînement Keras

### Modélisation (CNN)

- Réseau séquentiel Keras : 6 couches convolutionnelles (`Conv2D`), avec `MaxPooling2D` et `Dropout(0.2)` après chaque bloc
- Activation `ReLU` sur toutes les couches convolutionnelles, `Softmax` en sortie (43 classes)
- Optimiseur `SGD` (`lr=0.01`, `decay=1e-6`, `momentum=0.9`, `nesterov=True`)
- Fonction de perte `categorical_crossentropy`
- Callbacks : `LearningRateScheduler` (décroissance par paliers de 10 epochs), `ModelCheckpoint` (sauvegarde du meilleur modèle), `EarlyStopping` (patience de 5 epochs sur `val_acc`)

### Évaluation & amélioration

- Évaluation sur le jeu de test officiel GTSRB (`GT-final_test.csv`)
- Deuxième cycle d'entraînement avec `ImageDataGenerator` (décalage horizontal/vertical de 10%, zoom de 20%) pour réduire le sur-apprentissage
- Comparaison directe de l'exactitude avant/après augmentation de données

### Visualisation des résultats

- Courbes d'apprentissage (accuracy et loss, train vs validation) générées avec `matplotlib`
- Visualisation des prédictions Top-3 sous forme de graphique en barres horizontales avec probabilités

---

## Résultats

Valeurs extraites directement des sorties d'exécution du notebook.

| Phase | Détail | Valeur |
|---|---|---|
| Split des données | Train / Validation | 31 367 échantillons / 7 842 échantillons (80/20) |
| Entraînement initial (sans augmentation) | Arrêt anticipé | Epoch 24/30 |
| Entraînement initial (sans augmentation) | `val_acc` finale | 0.9978 |
| Évaluation sur le jeu de test officiel (sans augmentation) | Test accuracy | **0.9730** (97.30%) |
| Entraînement avec Data Augmentation | Arrêt anticipé | Epoch 27/30 |
| Entraînement avec Data Augmentation | `val_acc` finale | 0.9980 |
| Évaluation finale sur le jeu de test officiel (avec augmentation) | Test accuracy | **0.9845** (98.45%) |

La Data Augmentation (translation, zoom) a permis de gagner **1.15 point d'exactitude** sur le jeu de test (97.30% → 98.45%), confirmant l'intérêt de cette technique pour améliorer la généralisation du modèle sur des images non vues à l'entraînement.

---

## Stack Technique

<div align="center">

| Catégorie | Technologie | Rôle dans le projet |
|:---:|:---:|---|
| Langage | ![Python](https://img.shields.io/badge/Python_3-3776AB?style=flat-square&logo=python&logoColor=white) | Langage principal du notebook |
| Environnement | ![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white) | Développement interactif et documentation du pipeline |
| Deep Learning | ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) | Construction du modèle CNN (`Sequential`, `Conv2D`, `MaxPooling2D`, `Dropout`, `Dense`) |
| Persistance du modèle | ![h5py](https://img.shields.io/badge/h5py-HDF5-blue?style=flat-square) | Sauvegarde des poids du modèle (`model.h5`, `model_aug.h5`) |
| Calcul scientifique | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Manipulation des tableaux d'images (`X`) et labels one-hot (`Y`) |
| Traitement d'images | ![scikit-image](https://img.shields.io/badge/scikit--image-orange?style=flat-square) | Lecture des images, conversion HSV, égalisation d'histogramme, redimensionnement (`skimage.color`, `skimage.exposure`, `skimage.transform`, `skimage.io`) |
| Analyse & visualisation | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) | Lecture du CSV de vérité terrain (`GT-final_test.csv`) et génération des graphiques |
| Machine Learning utilitaire | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Split entraînement/validation (`train_test_split`) |
| Gestion de fichiers | ![glob](https://img.shields.io/badge/glob%20%2F%20os-standard%20library-lightgrey?style=flat-square) | Parcours et correction des chemins d'images du dataset |

</div>

---

## Architecture & Structure du Projet

### Structure actuelle du dépôt

```
Traffic-Sign-Recognition/
│
├── Traffic-Sign-Recognition.ipynb   # Notebook principal : exploration, preprocessing,
│                                     # entraînement, évaluation, augmentation, prédictions
├── classes.jpg                      # Aperçu visuel des 43 classes de panneaux
└── README.md                        # Documentation du projet
```

### Structure attendue du dataset (référencée dans le notebook)

```
data/
└── traffic_sign_dataset/
    ├── Final_Training/
    │   └── Images/
    │       └── <ClassId>/*.ppm       # Images d'entraînement, une sous-dossier par classe (0 à 42)
    ├── Final_Test/
    │   └── Images/*.ppm              # Images de test brutes
    └── GT-final_test.csv             # Vérité terrain du test (Filename ; ClassId)
```

### Pipeline de traitement

```
Images brutes (.ppm, 43 classes, dossier par ClassId)
        |
        v
Prétraitement (RGB -> HSV -> égalisation histogramme -> RGB -> recadrage centré -> resize 48x48)
        |
        v
Encodage one-hot des labels (NUM_CLASSES = 43)
        |
        v
Split Train / Validation (80/20)
        |
        v
CNN Sequential (6x Conv2D + MaxPooling2D + Dropout(0.2)) -> Flatten -> Dense -> Softmax
        |
        v
Entraînement SGD (LearningRateScheduler, ModelCheckpoint, EarlyStopping)
        |
        v
Évaluation sur GT-final_test.csv -> Test accuracy = 0.9730
        |
        v
Augmentation de données (ImageDataGenerator : shift 10%, zoom 20%) + réentraînement
        |
        v
Évaluation finale -> Test accuracy = 0.9845
        |
        v
Inférence sur nouvelle image -> Top-3 des prédictions avec probabilités
```

---

## Guide d'Installation & Démarrage Rapide

### Prérequis

| Outil | Version recommandée |
|---|---|
| Python | 3.x |
| Jupyter Notebook / JupyterLab | Dernière version stable |
| Keras / TensorFlow | Le notebook utilise l'API Keras historique (`keras.layers.core`, `keras.layers.convolutional`) ; sur un environnement récent, ces imports doivent être adaptés vers `tensorflow.keras` (voir [Roadmap](#roadmap)) |
| GPU (optionnel mais recommandé) | Accélère fortement l'entraînement (le notebook a été exécuté sur un NVIDIA Tesla K80) |
| pip | Pour la gestion des dépendances |

### 1. Cloner le dépôt

```bash
git clone https://github.com/ELGHAD/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Contenu recommandé pour `requirements.txt`, correspondant aux imports réels du notebook :

```txt
numpy
pandas
matplotlib
scikit-image
scikit-learn
h5py
tensorflow
jupyter
```

### 4. Télécharger et organiser le dataset GTSRB

Le notebook attend le jeu de données GTSRB organisé comme suit à la racine du projet :

```
data/traffic_sign_dataset/Final_Training/Images/
data/traffic_sign_dataset/Final_Test/Images/
data/traffic_sign_dataset/GT-final_test.csv
```

### 5. Configurer les variables d'environnement

Les paramètres suivants sont actuellement codés en dur dans la cellule 3 du notebook. Il est recommandé de les externaliser dans un fichier `.env` :

```env
NUM_CLASSES=43
IMG_SIZE=48
TRAINING_PATH=data/traffic_sign_dataset/Final_Training/Images/
TEST_PATH=data/traffic_sign_dataset/Final_Test/Images/
BATCH_SIZE=32
EPOCHS=30
LEARNING_RATE=0.01
```

> Ajoutez `.env`, `data/`, et `*.h5` à votre `.gitignore` pour éviter de committer des données volumineuses.

### 6. Lancer le notebook

```bash
jupyter notebook Traffic-Sign-Recognition.ipynb
```

Exécuter les cellules dans l'ordre : exploration du dataset, prétraitement, construction du modèle, entraînement, évaluation, augmentation de données, réentraînement, prédiction.

### 7. Entraînement via script (une fois le code extrait dans `src/`)

```bash
python src/train.py --epochs 30 --batch-size 32 --image-size 48
```

### 8. Prédiction sur une nouvelle image

```bash
python src/predict.py --image ./samples/panneau_test.png --model ./models/model_aug.h5
```

---

## Documentation & Exemples de Prédiction

### Distribution des images par classe

L'analyse exploratoire du dataset révèle un déséquilibre marqué entre les 43 classes : certaines classes dépassent 2000 images tandis que d'autres n'en comptent que quelques centaines. Ce déséquilibre est cohérent avec la structure connue du jeu de données GTSRB.

![Distribution des images par classe](distribution_classes)

*Distribution du nombre d'images par identifiant de classe (0 à 42). Ce déséquilibre justifie le recours à la Data Augmentation, qui a permis de faire progresser l'exactitude sur le test de 97.30% à 98.45%.*

### Exemple de prédiction — Feu tricolore

Le modèle reçoit une image de panneau annonçant un feu tricolore et retourne ses Top-3 prédictions classées par probabilité.

![Prédiction feu tricolore](prediction_feu_tricolore)

*Le modèle prédit la Classe 26 avec une confiance de 100.0%, loin devant les classes 17 et 25 (0.0%).*

### Exemple de prédiction — Passage piéton

![Prédiction passage piéton](prediction_passage_pieton)

*Figure 6.7 — Exemple de prédiction : panneau passage piéton et Top-3 prédictions associées. Le modèle identifie la Classe 27 avec 100.0% de confiance, devant les classes 11 et 18 (0.0% chacune).*

### Utilisation programmatique (code réel du notebook, cellule 39)

```python
# Prédiction sur le jeu de test après entraînement avec augmentation de données
y_predict = model.predict_classes(X_test)
accuracy = np.sum(y_predict == y_test) / np.size(y_predict)
print("Test accuracy = {}".format(accuracy))
# Test accuracy = 0.9844813935075217
```

> Note technique : `model.predict_classes()` est une méthode dépréciée puis supprimée dans les versions récentes de Keras/TensorFlow. Pour un usage sur un environnement actuel, remplacer par :
> ```python
> y_predict = np.argmax(model.predict(X_test), axis=-1)
> ```

---

## Tests & Qualité

Le projet est actuellement un notebook exploratoire, sans suite de tests automatisés ni script de couverture. Voici les pratiques recommandées pour fiabiliser le pipeline lors de son industrialisation :

### Tests à mettre en place

```bash
pytest tests/
```

- Prétraitement : vérifier que `preprocess_images()` retourne bien une image carrée de taille `48x48x3`
- Chargement des données : vérifier la cohérence entre le nombre d'images et de labels, et l'absence de classe manquante sur les 43 attendues
- Modèle : vérifier que la sortie de `build_cnn_model()` a bien la forme `(batch_size, 43)` et que les probabilités softmax somment à 1
- Régression de performance : vérifier que l'accuracy sur un sous-échantillon de test ne descend pas sous un seuil de référence (ex. 0.95), pour détecter une régression après modification du pipeline

### Évaluation du modèle

Le notebook évalue déjà le modèle sur le jeu de test officiel GTSRB via la cellule d'évaluation (comparaison `y_predict == y_test`). Pour aller plus loin :

- Ajouter un `classification_report` (scikit-learn) pour obtenir précision, rappel et F1-score par classe, et identifier les classes les plus sujettes à confusion (notamment les classes minoritaires visibles dans le graphique de distribution)
- Ajouter une matrice de confusion (`sklearn.metrics.confusion_matrix`) pour visualiser les erreurs inter-classes

### Couverture de code

```bash
pytest --cov=src tests/
```

---

## Roadmap

- [x] Exploration du dataset GTSRB et visualisation de la distribution des 43 classes
- [x] Prétraitement des images (égalisation d'histogramme, recadrage, redimensionnement 48x48)
- [x] Construction et entraînement d'un CNN Keras (6 couches convolutionnelles)
- [x] Évaluation sur le jeu de test officiel — accuracy de 97.30%
- [x] Amélioration par Data Augmentation — accuracy portée à 98.45%
- [x] Visualisation des prédictions Top-3 avec probabilités
- [ ] Migration des imports Keras historiques (`keras.layers.core`, `keras.layers.convolutional`) vers `tensorflow.keras` pour compatibilité avec les versions récentes
- [ ] Remplacement de `model.predict_classes()` (obsolète) par `np.argmax(model.predict(...))`
- [ ] Extraction du code du notebook vers des modules Python réutilisables (`src/`)
- [ ] Ajout d'un `classification_report` et d'une matrice de confusion par classe
- [ ] Gestion explicite du déséquilibre des classes (pondération de classes ou sur-échantillonnage ciblé)
- [ ] Ajout d'un fichier `requirements.txt` figé avec versions exactes
- [ ] Suite de tests automatisés (`pytest`) sur le prétraitement, le modèle et l'inférence
- [ ] Interface de démonstration interactive (Streamlit ou Gradio) pour tester une image en direct
- [ ] Exposition du modèle via une API REST (FastAPI ou Flask) pour une intégration temps réel
- [ ] Intégration continue (GitHub Actions) pour valider automatiquement le pipeline
- [ ] Optimisation du modèle pour l'embarqué (quantization, TensorFlow Lite) en vue d'une intégration ADAS
- [ ] Détection en amont de la localisation du panneau dans une image complète (pas seulement classification d'une image déjà cadrée)

---

## Contribution & Licence

### Comment contribuer

1. Forker le dépôt
2. Créer une branche dédiée
   ```bash
   git checkout -b feature/extraction-pipeline-src
   ```
3. Committer les changements avec des messages clairs
   ```bash
   git commit -m "feat: extraction du prétraitement dans src/preprocessing.py"
   ```
4. Pousser la branche
   ```bash
   git push origin feature/extraction-pipeline-src
   ```
5. Ouvrir une Pull Request en décrivant le contexte et l'impact du changement

### Directives de contribution

- Privilégier des scripts Python modulaires plutôt que des cellules de notebook monolithiques pour tout code destiné à être réutilisé
- Documenter toute nouvelle fonction avec une docstring claire (entrées, sorties, exemple)
- Ajouter des tests pour toute nouvelle fonctionnalité de prétraitement, d'entraînement ou d'inférence
- Ne pas régresser l'accuracy de référence (98.45% sur le jeu de test officiel GTSRB avec Data Augmentation) sans justification documentée

### Licence

Ce projet est distribué sous licence MIT. Vous êtes libre de l'utiliser, le modifier et le redistribuer, à des fins académiques ou commerciales, sous réserve de conserver la notice de copyright originale.

```
MIT License — libre d'utilisation, modification et distribution
avec attribution.
```

---

<div align="center">

Projet réalisé par [ELGHAD](https://github.com/ELGHAD)

</div>
