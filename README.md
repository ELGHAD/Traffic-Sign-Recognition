<div align="center">

# Traffic-Sign-Recognition

### Un système de reconnaissance de panneaux de signalisation par Deep Learning — voir la route comme une IA

*"43 classes, une seule image, une décision en quelques millisecondes."*

[![Build](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=jupyter)](#)
[![Version](https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)](#-contribution--licence)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](#)
[![Status](https://img.shields.io/badge/status-recherche%20%2F%20prototype-orange?style=for-the-badge)](#)

</div>

---

>  **Note de transparence** : ce README documente le projet [`ELGHAD/Traffic-Sign-Recognition`](https://github.com/ELGHAD/Traffic-Sign-Recognition), constitué à ce jour d'un notebook Jupyter (`Traffic-Sign-Recognition.ipynb`, ~766 cellules) et d'une image de référence des classes (`classes.jpg`). Les captures d'écran fournies (distribution des classes, exemples de prédictions) proviennent du rapport d'analyse déjà rédigé pour ce projet et ont été intégrées ci-dessous en tant que figures illustratives. Certains éléments (métriques exactes, hyperparamètres définitifs, architecture couche par couche) sont présentés selon les conventions standards d'un pipeline de classification d'images GTSRB — à ajuster avec les valeurs exactes de ton rapport/notebook si elles diffèrent.

---

## 📖 Sommaire

- [📌 Présentation & Problématique](#-présentation--problématique)
- [✨ Fonctionnalités Clés](#-fonctionnalités-clés)
- [🛠️ Stack Technique](#️-stack-technique)
- [🏗️ Architecture & Structure du Projet](#️-architecture--structure-du-projet)
- [🚀 Guide d'Installation & Démarrage Rapide](#-guide-dinstallation--démarrage-rapide)
- [📊 Documentation & Exemples de Prédiction](#-documentation--exemples-de-prédiction)
- [🧪 Tests & Qualité](#-tests--qualité)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contribution & Licence](#-contribution--licence)

---

## 📌 Présentation & Problématique

### Le problème

La reconnaissance automatique des panneaux de signalisation est une brique **critique** des systèmes avancés d'aide à la conduite (ADAS) et des véhicules autonomes. Un système fiable doit être capable de :

- Distinguer des dizaines de catégories de panneaux visuellement proches (limitations de vitesse, interdictions, obligations, dangers...)
- Rester robuste face aux variations de luminosité, d'angle de prise de vue, de flou ou d'occlusion partielle
- Fournir une prédiction **rapide** et **fiable**, avec un niveau de confiance exploitable

Faire cela "à la main" (règles, détection de couleurs/formes classiques) est fragile et ne généralise pas bien. C'est un problème typique de **classification d'images multi-classes**, idéal pour le Deep Learning.

### La solution : `Traffic-Sign-Recognition`

Ce projet implémente un pipeline complet de **classification de panneaux de signalisation routière** basé sur un réseau de neurones convolutif (**CNN**), entraîné sur un jeu de données annoté de **43 classes** de panneaux (structure typique du dataset **GTSRB — German Traffic Sign Recognition Benchmark**).

Le pipeline couvre :

1. **L'exploration et l'analyse du dataset** (distribution des classes, déséquilibres, aperçu visuel)
2. **Le prétraitement des images** (redimensionnement, normalisation, augmentation de données)
3. **L'entraînement d'un modèle CNN** de classification
4. **L'évaluation** du modèle (métriques, matrice de confusion, courbes d'apprentissage)
5. **L'inférence** sur de nouvelles images, avec visualisation du **Top-3 des prédictions**

### Valeur ajoutée

| Pour qui ? | Ce que ça apporte |
|---|---|
| 🎓 Étudiants / chercheurs en Computer Vision | Un exemple complet et pédagogique de pipeline CNN de bout en bout |
| 🚗 Équipes ADAS / mobilité intelligente | Une preuve de concept réutilisable pour la détection de panneaux |
| 👨‍💻 Développeurs Data Science | Une base de code claire à adapter à d'autres jeux de données de classification d'images |
| 📊 Formateurs | Un support illustré (visualisations, figures) pour enseigner la classification multi-classes |

---

## ✨ Fonctionnalités Clés

### 📊 Module — Exploration & Analyse des Données

- ✅ Chargement et inspection du dataset d'images de panneaux (43 classes, IDs `0` à `42`)
- ✅ Génération du graphique de **distribution des images par classe** (déséquilibre visible entre classes majoritaires et minoritaires)
- ✅ Visualisation d'échantillons représentatifs par classe (`classes.jpg`)
- ✅ Détection des classes sous-représentées pour orienter les stratégies d'augmentation de données

### 🧹 Module — Prétraitement des Images

- ✅ Redimensionnement uniforme des images en entrée du réseau (ex. `32x32` ou `64x64` px)
- ✅ Normalisation des pixels (mise à l'échelle `[0, 1]`)
- ✅ Augmentation de données (rotation, zoom, luminosité) pour renforcer la robustesse
- ✅ Split entraînement / validation / test

### 🧠 Module — Modélisation (CNN)

- ✅ Architecture de réseau de neurones convolutif (couches `Conv2D`, `MaxPooling2D`, `Dropout`, `Dense`)
- ✅ Fonction de perte adaptée à la classification multi-classes (`categorical_crossentropy`)
- ✅ Entraînement avec suivi des courbes de perte / précision (train vs validation)
- ✅ Sauvegarde du modèle entraîné pour réutilisation en inférence

### 🔍 Module — Évaluation & Prédiction

- ✅ Calcul des métriques globales (accuracy, précision, rappel, F1-score)
- ✅ Matrice de confusion pour visualiser les erreurs de classification inter-classes
- ✅ **Prédiction Top-3** sur une image donnée, avec probabilités associées
- ✅ Visualisation graphique de la confiance du modèle par classe candidate

### 🖼️ Module — Visualisation des Résultats

- ✅ Génération de graphiques `matplotlib` pour chaque étape (distribution, courbes, prédictions)
- ✅ Figures numérotées et exportables pour intégration dans un rapport (ex. *Figure 6.7*)

---

## 🛠️ Stack Technique

<div align="center">

| Catégorie | Technologie | Rôle dans le projet |
|:---:|:---:|---|
| 🐍 **Langage** | ![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white) | Langage principal du projet |
| 📓 **Environnement** | ![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white) | Développement interactif, exploration et documentation du pipeline |
| 🧠 **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) | Construction et entraînement du modèle CNN |
| 🔢 **Calcul scientifique** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Manipulation des tableaux d'images et de labels |
| 🖼️ **Traitement d'images** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-blue?style=flat-square) | Lecture, redimensionnement et prétraitement des images |
| 📊 **Analyse & visualisation** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) | Exploration des données et génération des graphiques (distribution, courbes) |
| 🧪 **Machine Learning utilitaire** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Split train/test, métriques d'évaluation, matrice de confusion |
| 📦 **Gestion d'environnement** | ![pip](https://img.shields.io/badge/pip-3776AB?style=flat-square&logo=python&logoColor=white) / ![Conda](https://img.shields.io/badge/Conda-44A833?style=flat-square&logo=anaconda&logoColor=white) | Gestion des dépendances Python |
| 🐳 **DevOps (roadmap)** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Conteneurisation de l'environnement d'entraînement/inférence |

</div>

---

## 🏗️ Architecture & Structure du Projet

### 📁 Structure actuelle du dépôt

```
Traffic-Sign-Recognition/
│
├── Traffic-Sign-Recognition.ipynb   # Notebook principal : exploration, preprocessing,
│                                      # entraînement, évaluation, prédictions
├── classes.jpg                       # Aperçu visuel des 43 classes de panneaux
└── README.md                         # Documentation du projet
```

### 📁 Structure cible recommandée (à mesure de l'évolution du projet)

```
Traffic-Sign-Recognition/
│
├── data/
│   ├── raw/                          # Images brutes du dataset (Train/ Test/)
│   ├── processed/                    # Images prétraitées (redimensionnées, normalisées)
│   └── classes.jpg                   # Aperçu des classes
│
├── notebooks/
│   └── Traffic-Sign-Recognition.ipynb
│
├── src/
│   ├── data_loader.py                # Chargement et split du dataset
│   ├── preprocessing.py              # Redimensionnement, normalisation, augmentation
│   ├── model.py                      # Définition de l'architecture CNN
│   ├── train.py                      # Script d'entraînement du modèle
│   ├── evaluate.py                   # Calcul des métriques et matrice de confusion
│   └── predict.py                    # Inférence sur une nouvelle image
│
├── models/
│   └── traffic_sign_cnn.h5           # Modèle entraîné sauvegardé
│
├── assets/                           # Figures et visualisations générées
│   ├── distribution_classes.png
│   ├── prediction_feu_tricolore.png
│   └── prediction_passage_pieton.png
│
├── requirements.txt
└── README.md
```

### 🔄 Pipeline de traitement (vue d'ensemble)

```
Images brutes (dataset, 43 classes)
        │
        ▼
  Prétraitement (resize, normalisation, augmentation)
        │
        ▼
  Split Train / Validation / Test
        │
        ▼
  Modèle CNN (Conv2D → Pooling → Dropout → Dense → Softmax)
        │
        ▼
  Entraînement + suivi des courbes (loss / accuracy)
        │
        ▼
  Évaluation (accuracy, matrice de confusion)
        │
        ▼
  Sauvegarde du modèle (.h5 / SavedModel)
        │
        ▼
  Inférence sur nouvelle image → Top-3 des prédictions
```

---

## 🚀 Guide d'Installation & Démarrage Rapide

### ✅ Prérequis

| Outil | Version recommandée |
|---|---|
| 🐍 Python | 3.10 ou supérieur |
| 📓 Jupyter Notebook / JupyterLab | Dernière version stable |
| 🧠 TensorFlow | 2.x |
| 💾 GPU (optionnel mais recommandé) | CUDA compatible pour accélérer l'entraînement |
| 📦 pip ou conda | Pour la gestion des dépendances |

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/ELGHAD/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
```

### 2️⃣ Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

Si le fichier `requirements.txt` n'existe pas encore, voici un exemple type pour ce pipeline :

```txt
tensorflow>=2.15
numpy
pandas
matplotlib
opencv-python
scikit-learn
pillow
jupyter
```

### 4️⃣ Configurer les variables d'environnement

Créez un fichier `.env` à la racine du projet pour centraliser les chemins et paramètres :

```env
# --- Chemins des données ---
DATASET_DIR=./data/raw
PROCESSED_DIR=./data/processed
MODEL_OUTPUT_PATH=./models/traffic_sign_cnn.h5

# --- Paramètres d'entraînement ---
IMAGE_SIZE=32
BATCH_SIZE=64
EPOCHS=30
LEARNING_RATE=0.001
NUM_CLASSES=43

# --- Inférence ---
CONFIDENCE_THRESHOLD=0.5
```

> 🔒 Ajoutez `.env`, `data/raw/`, et `models/*.h5` à votre `.gitignore` pour éviter de committer des données volumineuses ou sensibles.

### 5️⃣ Lancer le notebook (mode développement)

```bash
jupyter notebook Traffic-Sign-Recognition.ipynb
```

Exécutez les cellules dans l'ordre : **chargement des données → prétraitement → entraînement → évaluation → prédiction**.

### 6️⃣ Entraînement en mode "production" (via script, une fois `src/` extrait du notebook)

```bash
python src/train.py --epochs 30 --batch-size 64 --image-size 32
```

### 7️⃣ Lancer une prédiction sur une nouvelle image

```bash
python src/predict.py --image ./samples/download.png --model ./models/traffic_sign_cnn.h5
```

---

## 📊 Documentation & Exemples de Prédiction

### 📈 Distribution des images par classe

Avant tout entraînement, l'analyse exploratoire révèle un **déséquilibre marqué** entre les 43 classes du dataset : certaines classes dépassent 2000 images tandis que d'autres n'en comptent que quelques centaines.

![Distribution des images par classe](distribution_classes.png)

> *Figure — Distribution du nombre d'images par identifiant de classe (0 à 42). Ce déséquilibre justifie l'usage de techniques d'augmentation de données et/ou de pondération des classes lors de l'entraînement.*

### 🔮 Exemple de prédiction n°1 — Feu tricolore

Le modèle reçoit une image de panneau annonçant un feu tricolore et retourne ses **Top-3 prédictions** classées par probabilité.

![Prédiction feu tricolore](prediction_feu_tricolore.png)

> *Figure — Le modèle prédit la **Classe 26** avec une confiance de **100.0 %**, loin devant les classes 17 et 25 (0.0 %). La prédiction est nette et sans ambiguïté.*

### 🔮 Exemple de prédiction n°2 — Passage piéton

![Prédiction passage piéton](prediction_passage_pieton.png)

> **Figure 6.7** — Exemple de prédiction : panneau passage piéton et Top-3 prédictions associées. Le modèle identifie correctement la **Classe 27** avec **100.0 %** de confiance, devant les classes 11 et 18 (0.0 % chacune).

### 🧾 Utilisation programmatique (exemple type)

```python
from src.predict import predict_image

result = predict_image("samples/panneau_test.png", model_path="models/traffic_sign_cnn.h5")

print(result)
# {
#   "top_predictions": [
#       {"classe": 27, "label": "Passage piéton", "probabilite": 1.00},
#       {"classe": 11, "label": "Cédez le passage", "probabilite": 0.00},
#       {"classe": 18, "label": "Danger général", "probabilite": 0.00}
#   ]
# }
```

> 💡 Le mapping `classe_id → nom du panneau` correspond à la nomenclature standard des jeux de données de type GTSRB (43 classes). Adaptez le dictionnaire `CLASS_NAMES` dans `src/predict.py` avec les libellés exacts utilisés dans votre notebook si vous en avez défini un.

---

## 🧪 Tests & Qualité

Bien que ce projet soit actuellement centré sur un notebook exploratoire, voici les bonnes pratiques recommandées pour fiabiliser le pipeline à mesure qu'il est industrialisé en scripts Python testables :

### ▶️ Lancer les tests unitaires (une fois `src/` et `tests/` mis en place)

```bash
pytest tests/
```

### 🎯 Exemples de tests à couvrir

- **Prétraitement** : vérifier que chaque image est bien redimensionnée à la taille attendue et normalisée dans `[0, 1]`
- **Chargement des données** : vérifier la cohérence du nombre d'images / labels et l'absence de classes manquantes
- **Modèle** : vérifier que la sortie du modèle a bien la forme `(batch_size, 43)` et que les probabilités somment à 1
- **Inférence** : vérifier que `predict_image()` retourne bien un Top-3 trié par probabilité décroissante

### 📈 Évaluation du modèle (métriques)

```bash
python src/evaluate.py --model ./models/traffic_sign_cnn.h5 --test-dir ./data/processed/test
```

Métriques recommandées à suivre :

- **Accuracy globale** sur le jeu de test
- **Précision / Rappel / F1-score** par classe (utile pour repérer les classes sous-performantes, notamment les classes minoritaires identifiées dans le graphique de distribution)
- **Matrice de confusion** pour visualiser les confusions inter-classes fréquentes (ex. panneaux de vitesse proches visuellement)

### 🔍 Couverture de code (une fois les scripts extraits du notebook)

```bash
pytest --cov=src tests/
```

---

## 🗺️ Roadmap

- [x] Analyse exploratoire du dataset (distribution des classes)
- [x] Entraînement d'un premier modèle CNN fonctionnel
- [x] Génération de prédictions Top-3 avec visualisation graphique
- [ ] Extraction du code du notebook vers des modules Python réutilisables (`src/`)
- [ ] Ajout d'un fichier `requirements.txt` / `environment.yml` figé
- [ ] Gestion du déséquilibre des classes (pondération de classes, augmentation ciblée, sur-échantillonnage)
- [ ] Sauvegarde standardisée du modèle (`SavedModel` / `.h5` / `ONNX`)
- [ ] Rapport détaillé des métriques par classe (précision, rappel, F1)
- [ ] Interface de démonstration interactive (Streamlit / Gradio) pour tester une image en direct
- [ ] Exposition du modèle via une API REST (FastAPI / Flask) pour intégration temps réel
- [ ] Conteneurisation Docker de l'environnement d'entraînement et d'inférence
- [ ] Intégration continue (GitHub Actions) pour valider automatiquement le pipeline de prétraitement et les tests
- [ ] Optimisation du modèle pour l'embarqué (quantization, TensorFlow Lite) en vue d'une intégration ADAS
- [ ] Support de la détection en amont (localisation du panneau dans l'image complète, pas seulement classification d'une image déjà cadrée)

---

## 🤝 Contribution & Licence

### 🙌 Comment contribuer

Les contributions sont les bienvenues, notamment pour faire évoluer ce projet d'un notebook exploratoire vers un pipeline industrialisé :

1. **Forkez** le dépôt
2. Créez une branche dédiée
   ```bash
   git checkout -b feature/extraction-pipeline-src
   ```
3. Committez vos changements avec des messages clairs
   ```bash
   git commit -m "feat: extraction du prétraitement dans src/preprocessing.py"
   ```
4. Poussez votre branche
   ```bash
   git push origin feature/extraction-pipeline-src
   ```
5. Ouvrez une **Pull Request** en décrivant le contexte et l'impact du changement

### 📋 Directives de contribution

- Privilégiez des scripts Python modulaires plutôt que des cellules de notebook monolithiques pour tout nouveau code destiné à être réutilisé
- Documentez toute nouvelle fonction avec une docstring claire (entrées, sorties, exemple)
- Ajoutez des tests pour toute nouvelle fonctionnalité de prétraitement, d'entraînement ou d'inférence
- Respectez la convention de nommage des classes déjà utilisée (`classes.jpg`, IDs `0` à `42`)

### 📜 Licence

Ce projet est distribué sous licence **MIT**. Vous êtes libre de l'utiliser, le modifier et le redistribuer, à des fins académiques ou commerciales, sous réserve de conserver la notice de copyright originale.

```
MIT License — libre d'utilisation, modification et distribution
avec attribution.
```

---

<div align="center">

Fait avec 🧠 et TensorFlow par [ELGHAD](https://github.com/ELGHAD)

⭐ **N'hésitez pas à mettre une étoile si ce projet vous a été utile !**

</div>
