# Projet Data Engineering

Ce projet a été mené dans le cadre du module Data Engineering du Master 2 MLDS (Machine Learning for Data Science) de l’Université Paris Cité. Il simule un projet d'entreprise réel où trois développeurs collaborent sur une tâche commune.

## Objectifs du Projet :

1. **Collaboration avec Git :** Pratique de la collaboration entre les trois développeurs en utilisant Git.
2. **Déploiement Docker :** Déploiement d’une image Docker sur Docker Hub.

## Description du Projet :

Le projet vise à développer un modèle de clustering reposant sur la réduction de la dimensionnalité par l'**ACP**, **t-SNE** et **UMAP** (avec un espace réduit à 20 dimensions). Cette approche tandem ou séquentielle prend en entrée des données textuelles (2000 documents NG20) et combine les méthodes de réduction de dimensionnalité avec les algorithmes de clustering **k-means** et **CAH**.

Le dataset NG20 regroupe des documents de nouvelles en 20 catégories distinctes.

## Réalisateurs du Projet :

- Abdesselam BENAMEUR
- Hakim IGUENI
- Salma TALANTIKITE

## Approche Méthodologique :

Nous avons utilisé trois méthodes de réduction de dimensionnalité (ACP, t-SNE et UMAP), suivies d'un clustering avec k-means et CAH sur les résultats obtenus. De plus, nous avons également appliqué clustering sur les données d'origine pour comparaison.

## Déploiement :

Une fois le développement terminé, place au déploiement. Pour ce faire, nous avons créé une image Docker et mis en place un volume local afin de créer un miroir entre les fichiers locaux et ceux dans le conteneur. Cela nous offre l'avantage d'interagir facilement avec les fichiers du conteneur tout en conservant la cohérence avec les fichiers locaux.
Enfin, nous avons publié notre image Docker sur le **_DockerHub_**.
Pour plus d'informations sur son utilisation: voir la section **Utilisation**.

## Librairies Utilisées

Les librairies utilisées sont mentionnées dans le fichier `requirements.txt`

## Utilisation :

Pour pouvoir utiliser notre projet, assurez-vous d'avoir Docker bien installé. Pour vérifier ça, utilisez la commande suivante:

```bash
docker --version
```

Si vous n'avez pas Docker, suivez les instruction décrites dans ce lien :

[Installer Docker](https://www.docker.com/products/docker-desktop/).

Une fois Docker installé et configuré, téléchargez l'image Docker en éxecutant cette commande :

```bash
docker pull abdesselambenameur/projet_data_eng:v1
```

Une fois l’image Docker téléchargée, vous pouvez lancer le projet en utiliser deux commandes :

**Avec Volume :**

```bash
docker run -v .:/app projet_data_eng:v1
```

Cette commande doit être exécutée à l'intérieur du dossier du projet, elle crée un conteneur avec un volume, établissant une liaison entre le chemin local et le chemin dans le conteneur, permettant ainsi une interaction efficace.

**Sans Volume :**

```bash
docker run projet_data_eng:v1
```

Utilisez cette commande si vous n'avez pas besoin de spécifier un volume. Elle lancera le projet dans un conteneur Docker sans liens particuliers avec des fichiers locaux.

## Évaluation :

Pour chaque méthode, nous avons calculé les métriques **NMI** (Normalized Mutual Information) et **ARI** (Adjusted Rand Index).
