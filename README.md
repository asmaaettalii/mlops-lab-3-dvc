# Rapport de Lab – Gestion de données et pipeline reproductible avec DVC

## 1. Introduction

L’objectif de ce laboratoire est d’utiliser **DVC (Data Version Control)** afin de gérer des datasets volumineux et de construire un **pipeline reproductible** pour le traitement et l’entraînement de modèles de Machine Learning.  
L’intégration avec **Git** permet de versionner le code et la structure du projet tout en conservant un dépôt léger.

---

## 2. Étape 1 : Initialisation de DVC dans le projet

### Instructions
Installation et initialisation de DVC :

<img width="945" height="431" alt="image" src="https://github.com/user-attachments/assets/5791247c-d7c7-42d7-b3c2-f801a1959efc" />

<img width="940" height="541" alt="image" src="https://github.com/user-attachments/assets/3aaf410c-a395-48ec-9eae-e050c2b3b1d6" />

### Résultat attendu :
Un projet prêt à suivre des fichiers volumineux avec Git + DVC.

## 3. Étape 2 : Versionner les données brutes avec DVC
Instructions réalisées :

Suppression de data/ de .gitignore

Ajout du dataset au suivi DVC :

<img width="945" height="226" alt="image" src="https://github.com/user-attachments/assets/36c5993b-2b81-46b9-986d-edae8ab518e9" />

### Changements effectués par la commande :

* **Création :** `data/raw.csv.dvc`
* **Mise à jour :** `.gitignore`
* **Git :** Ajout des fichiers au versionnement (staging)

<img width="945" height="234" alt="image" src="https://github.com/user-attachments/assets/010687ae-4038-4770-b7f2-6b4325e4703a" />

### Résultat attendu :

* **Gestion des fichiers :** `data/raw.csv` n’est plus suivi par Git seul.
* **Versionnement :** `raw.csv.dvc` est désormais versionné.
* **Optimisation :** Git reste léger, tandis que **DVC** gère les fichiers volumineux.

## 4. Étape 3 : Configuration d’un remote DVC

### Instructions réalisées :

1.  **Création d’un dossier pour le remote :**
   
   <img width="945" height="79" alt="image" src="https://github.com/user-attachments/assets/7a338ca7-a7b7-4854-9b52-227741533017" />

2.  **Déclaration du remote :**
    
    <img width="945" height="73" alt="image" src="https://github.com/user-attachments/assets/e16437ce-c635-4c83-bcfa-c90d101d1271" />


3.  **Versionnement de la configuration :**

    <img width="945" height="68" alt="image" src="https://github.com/user-attachments/assets/f105f1bd-967b-4003-a76e-ab3fe28ff66f" />

    <img width="945" height="89" alt="image" src="https://github.com/user-attachments/assets/96f611d3-df1a-4c5c-a92e-9d5aefacab53" />


---

### Résultat attendu :
Le projet **DVC** est maintenant configuré pour stocker les fichiers dans un emplacement centralisé.

### Screenshot :
*(Insérer ici le screenshot du fichier `.dvc/config` et de l’arborescence)*

## 5. Étape 4 : Push des données dans le remote DVC

### Instructions réalisées :

**Envoi des données :**

<img width="945" height="170" alt="image" src="https://github.com/user-attachments/assets/43ef19c9-6a3a-4bde-a49c-892c34f74317" />

**Vérification du remote :**

<img width="945" height="342" alt="image" src="https://github.com/user-attachments/assets/6e7b2a2b-973b-496a-a98d-eb624a9c82d6" />

### Résultat attendu :

* Les fichiers hashés sont présents dans `dvc_storage/`.
* Les données sont partagées et récupérables par tout collaborateur.

## 6. Étape 5 : Simulation d’une collaboration

### Instructions réalisées :

**Suppression du dataset local :**

<img width="945" height="86" alt="image" src="https://github.com/user-attachments/assets/dfb73bf1-7ab1-43cb-88f1-89c0bb766f4f" />

**Vérification de l’absence de fichier :**

<img width="945" height="420" alt="image" src="https://github.com/user-attachments/assets/5522729d-312b-4461-ad68-a2aea68ca2b9" />

**Récupération du dataset via DVC :**

<img width="945" height="270" alt="image" src="https://github.com/user-attachments/assets/0166b22e-77f5-411c-ae1e-7184935d266e" />


### Résultat attendu :
Le fichier `data/raw.csv` réapparaît identique à l’original, confirmant la reproductibilité du suivi DVC.

## 7. Étape 6 : Création d’un pipeline reproductible

### Instructions réalisées :

**Étape d’entraînement :**

<img width="945" height="375" alt="image" src="https://github.com/user-attachments/assets/3f5106e6-97a6-4ef8-8f17-018eb9e48273" />

**Étape d’évaluation :**

<img width="945" height="260" alt="image" src="https://github.com/user-attachments/assets/6ab7c113-261b-468b-b5c9-669a62abe56c" />

<img width="945" height="58" alt="image" src="https://github.com/user-attachments/assets/9cf800fe-1eed-4d2d-8441-fb5b69b64545" />


### Résultat attendu :

DVC a enregistré dans `dvc.yaml` :
* Les dépendances
* Les sorties
* Les commandes du pipeline

## 8. Étape 7 : Reproduction automatique du pipeline

### Instructions réalisées :

**Modification d’un script (par ex. src/prepare_data.py)**

**Reproduction automatique :**

<img width="945" height="447" alt="image" src="https://github.com/user-attachments/assets/b93761fc-e5ee-44d0-b75f-365678337512" />
<img width="945" height="460" alt="image" src="https://github.com/user-attachments/assets/6e5c50ca-2b3d-4433-bbcf-c9ab07cdec14" />

### Résultat attendu :
Seules les étapes impactées sont réexécutées, démontrant la reproductibilité totale.

## 9. Conclusion

* **DVC** permet de versionner efficacement les datasets volumineux.
* L’utilisation d’un **remote** facilite la collaboration entre plusieurs utilisateurs.
* Les **pipelines DVC** assurent une reproductibilité totale pour la préparation, l’entraînement et l’évaluation des modèles.
