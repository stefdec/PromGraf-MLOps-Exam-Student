# Monitoring MLOps — Bike Sharing

Ce projet met en place une API FastAPI capable de prédire le nombre de vélos partagés (`cnt`), avec une stack complète de monitoring basée sur Prometheus, Grafana, Evidently et Node Exporter.

Le projet contient :

une API pour effectuer les prédictions et évaluer le modèle ;

un générateur de trafic qui appelle continuellement l'API avec des requêtes valides mais aussi des erreurs volontaires ;

des métriques Prometheus pour suivre l'API, le modèle et l'infrastructure ;

trois dashboards Grafana automatiquement provisionnés au démarrage ;

des règles d'alertes dans Prometheus et Grafana ;

un service d'évaluation qui mesure les performances du modèle ainsi que la dérive des données avec Evidently.

## Prérequis

Docker avec Docker Compose ;

`make` ;

`curl`, uniquement utilisé par la commande `make fire-alert`.

## Lancer le projet

À la racine du projet :

```bash
make
```

La cible par défaut du Makefile est `all`, ce qui revient à exécuter :

```bash
make all
```

Cette commande construit les différentes images Docker puis démarre tous les services en arrière-plan.

Les interfaces sont ensuite accessibles ici :

| Service               | Adresse                       |
| --------------------- | ----------------------------- |
| API FastAPI           | http://localhost:8080         |
| Documentation Swagger | http://localhost:8080/docs    |
| Métriques de l'API    | http://localhost:8080/metrics |
| Prometheus            | http://localhost:9090         |
| Grafana               | http://localhost:3000         |
| Node Exporter         | http://localhost:9100/metrics |

Les identifiants Grafana par défaut sont `admin` / `admin`. Grafana peut demander de modifier le mot de passe lors de la première connexion.

Le premier démarrage peut être un peu plus long puisque Docker doit construire les images et que l'API entraîne le modèle avant de commencer à répondre aux requêtes.

Pour suivre le démarrage de l'API :

```bash
docker-compose logs -f bike-api
```

Pour arrêter les services :

```bash
make stop
```

## Commandes du Makefile

| Commande             | Rôle                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `make` ou `make all` | Construit et démarre l'API, Prometheus, Grafana, Node Exporter, le service d'évaluation et le générateur de trafic. |
| `make stop`          | Arrête les services du projet. Les volumes Prometheus et Grafana sont conservés.                                    |
| `make evaluation`    | Reconstruit et lance le conteneur d'évaluation afin de mettre à jour les métriques du modèle et de dérive.          |
| `make predict`       | Reconstruit et démarre le générateur permanent de trafic vers `/predict`.                                           |
| `make fire-alert`    | Force `model_rmse_score` à `1000` afin de tester l'alerte Grafana liée à la RMSE.                                   |

Après avoir lancé :

```bash
make fire-alert
```

il faut attendre au moins 10 secondes pour que l'alerte Grafana passe en état d'alerte.

Pour remettre ensuite la RMSE à sa vraie valeur, il suffit de relancer une évaluation :

```bash
make evaluation
```

## Simulation permanente du trafic avec `predict`

Le service Docker `predict` exécute le script `src/predict/run_predict.py`.

Son rôle est uniquement de générer du trafic vers l'API. Il n'entraîne pas le modèle et ne réalise aucune évaluation.

Il simule des utilisateurs qui appellent continuellement :

```text
http://bike-api:8080/predict
```

Par défaut, le script fonctionne sans limite tant que le conteneur est lancé.

Il envoie une requête toutes les 0,5 seconde avec environ 80 % de requêtes valides et 20 % de requêtes volontairement incorrectes.

Il continue également de fonctionner si l'API est temporairement indisponible.

Les erreurs sont choisies aléatoirement parmi trois scénarios :

un champ obligatoire manquant ;

un type de donnée invalide ;

une requête envoyée avec la méthode HTTP `GET` à la place de `POST`.

Cela permet notamment de générer des réponses HTTP `422` ou `405`.

Ce trafic permet d'alimenter les métriques utilisées dans le dashboard API, notamment le nombre de requêtes, les temps de réponse, les codes HTTP et le taux d'erreur.

Le service est automatiquement démarré avec :

```bash
make all
```

Il est aussi possible de le reconstruire et de le relancer indépendamment avec :

```bash
make predict
```

Les logs permettent de suivre les requêtes envoyées, leur type, leur code de retour ainsi que les compteurs cumulés :

```bash
docker-compose logs -f predict
```

Le comportement du générateur peut être configuré avec les variables d'environnement suivantes :

| Variable                   | Valeur par défaut              | Description                                             |
| -------------------------- | ------------------------------ | ------------------------------------------------------- |
| `PREDICT_URL`              | `http://bike-api:8080/predict` | Endpoint appelé.                                        |
| `REQUEST_INTERVAL_SECONDS` | `0.5`                          | Temps d'attente entre deux requêtes.                    |
| `ERROR_RATE`               | `0.2`                          | Proportion d'erreurs simulées, entre `0` et `1`.        |
| `REQUEST_COUNT`            | `0`                            | Nombre de requêtes à envoyer. `0` signifie sans limite. |
| `REQUEST_TIMEOUT_SECONDS`  | `5`                            | Timeout HTTP pour une requête.                          |

## Entraînement et prédiction

Au démarrage du conteneur `bike-api`, un modèle `RandomForestRegressor` est entraîné puis conservé en mémoire pour effectuer les prédictions.

Les données utilisées comme référence correspondent uniquement au mois de janvier 2011, du 1er au 31 janvier inclus.

L'endpoint :

```text
POST /predict
```

accepte les onze variables utilisées par le modèle.

Exemple :

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temp": 0.24,
    "atemp": 0.2879,
    "hum": 0.81,
    "windspeed": 0.0,
    "mnth": 1,
    "hr": 8,
    "weekday": 6,
    "season": 1,
    "holiday": 0,
    "workingday": 0,
    "weathersit": 1
  }'
```

## Évaluation du modèle

Le service `evaluation` télécharge le dataset Bike Sharing puis sélectionne la période courante définie dans `run_evaluation.py`.

Il envoie ensuite jusqu'à 1 000 observations à l'endpoint :

```text
POST /evaluate
```

avec leurs features ainsi que leur valeur réelle `cnt`.

Lors de cet appel, l'API effectue plusieurs opérations :

1. elle calcule les prédictions du modèle sur les observations reçues ;

2. elle compare les prédictions avec les vraies valeurs `cnt` ;

3. elle utilise Evidently pour comparer la distribution des données courantes avec les données de référence de janvier ;

4. elle met à jour les métriques Prometheus concernant les performances du modèle et la dérive des données.

Les métriques RMSE, MAE, R² ainsi que les métriques de drift sont donc mises à jour uniquement lorsqu'une nouvelle évaluation est effectuée.

Lors du lancement de la stack, Docker Compose peut démarrer le service `evaluation` alors que l'API n'est pas encore complètement disponible.

Pour gérer ce cas, `run_evaluation.py` peut tenter l'appel jusqu'à 10 fois avec un intervalle de 10 secondes entre chaque tentative.

Si les 10 appels échouent, le conteneur termine avec un code de sortie différent de zéro.

Le service `predict` fonctionne indépendamment et peut continuer à générer du trafic pendant ce temps.

## Métriques Prometheus

L'API expose notamment les métriques suivantes :

`api_requests_total{endpoint,method,status_code}` : nombre total de requêtes HTTP ;

`api_request_duration_seconds{endpoint,method,status_code}` : histogramme permettant de mesurer les temps de réponse ;

`model_rmse_score` : RMSE du modèle ;

`model_mae_score` : MAE du modèle ;

`model_r2_score` : coefficient de détermination R² ;

`evidently_data_drift_detected_status` : état global du data drift détecté par Evidently.

## Choix de la métrique personnalisée

La métrique personnalisée utilisée pour le monitoring du drift est :

```text
evidently_data_drift_detected_status
```

Il s'agit d'une `Gauge` Prometheus qui prend deux valeurs possibles :

`0` lorsqu'aucune dérive globale n'est détectée ;

`1` lorsqu'Evidently considère qu'une dérive est présente entre les données courantes et les données de référence de janvier.

Cette métrique complète les métriques classiques de performance comme RMSE, MAE et R².

Ces dernières permettent de mesurer directement la qualité des prédictions mais nécessitent de connaître la vraie valeur `cnt`.

Le data drift permet de surveiller autre chose : l'évolution de la distribution des données reçues par le modèle.

Une dérive peut donc apparaître avant même qu'une baisse importante des performances ne soit visible.

Le format binaire de cette métrique permet aussi de l'utiliser très simplement dans Grafana ou dans une règle d'alerte Prometheus.

Cette gauge est mise à jour uniquement lors d'un appel réussi à `/evaluate`.

Les appels à `/predict` ne modifient pas cette métrique.

## Dashboards et alertes Grafana

Au démarrage, Grafana charge automatiquement la datasource Prometheus, les dashboards JSON ainsi que les règles d'alerte présentes dans :

```text
deployment/grafana/provisioning
```

Il n'est donc pas nécessaire d'importer manuellement les dashboards.

Dans le dossier Grafana `MLOps`, trois dashboards sont disponibles.

### Bike API Observability

Ce dashboard permet de suivre le comportement de l'API :

nombre et débit des requêtes ;

latence P95 ;

codes HTTP ;

nombre et taux d'erreurs.

### Bike Model Observability

Ce dashboard permet de suivre les performances et l'état du modèle :

RMSE ;

MAE ;

R² ;

état du data drift.

### Node Exporter Full

Ce dashboard permet de suivre les ressources de la machine :

CPU ;

mémoire ;

disque ;

autres métriques système exposées par Node Exporter.

Ces dashboards sont basés sur des dashboards disponibles dans le hub Grafana puis modifiés pour correspondre aux besoins du projet.

Les métriques du modèle restent à zéro tant qu'aucune évaluation n'a été exécutée avec succès.

Pour les initialiser ou les mettre à jour :

```bash
make evaluation
```

Une alerte Grafana nommée :

```text
Bike model RMSE too high
```

est configurée pour se déclencher lorsque :

```text
model_rmse_score > 100
```

pendant au moins 10 secondes.

La commande :

```bash
make fire-alert
```

appelle l'endpoint de test `/false_rmse`, qui force temporairement la RMSE à `1000`.

Cela permet de vérifier rapidement que l'alerte fonctionne correctement.

Prometheus charge également deux règles depuis :

```text
deployment/prometheus/rules/alert_rules.yml
```

`BikeApiDown` est déclenchée si l'API reste indisponible pendant une minute.

`DataDriftDetected` est déclenchée si `evidently_data_drift_detected_status` reste à `1` pendant cinq minutes.

## Structure utile du projet

```text
.
├── deployment/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── prometheus/
│       ├── prometheus.yml
│       └── rules/
├── src/
│   ├── api/
│   ├── evaluation/
│   └── predict/
├── docker-compose.yml
├── Makefile
└── README.md
```

## Diagnostic rapide

Pour vérifier l'état des différents conteneurs :

```bash
docker-compose ps
```

Pour suivre les logs de l'API :

```bash
docker-compose logs -f bike-api
```

Pour voir les logs du service d'évaluation :

```bash
docker-compose logs evaluation
```

Pour suivre le simulateur de trafic :

```bash
docker-compose logs -f predict
```

Si les dashboards liés à l'API sont vides, vérifier que le conteneur `predict` est bien démarré et génère des requêtes.

Si le dashboard du modèle reste vide ou affiche uniquement des zéros, lancer :

```bash
make evaluation
```

puis vérifier que le conteneur `evaluation` termine correctement sans erreur.
