# Monitoring MLOps — Bike Sharing

Ce projet expose une API FastAPI qui prédit le nombre de vélos partagés (`cnt`) et fournit une stack de supervision complète avec Prometheus, Grafana et Node Exporter.

Le projet comprend :

- une API de prédiction et d'évaluation du modèle ;
- un générateur permanent de trafic réaliste, avec succès et erreurs volontaires ;
- des métriques Prometheus pour l'API, le modèle et l'infrastructure ;
- trois dashboards Grafana provisionnés automatiquement ;
- des alertes Prometheus et Grafana ;
- un script d'évaluation qui mesure les performances et la dérive des données.

## Prérequis

- Docker avec Docker Compose ;
- `make` ;
- `curl`, uniquement pour la cible `fire-alert`.

## Lancer le projet

À la racine du dépôt :

```bash
make
```

La cible par défaut est `all`. Elle construit les images puis démarre tous les services en arrière-plan :

```bash
make all
```

Les interfaces sont ensuite disponibles aux adresses suivantes :

| Service | Adresse |
| --- | --- |
| API FastAPI | http://localhost:8080 |
| Documentation Swagger | http://localhost:8080/docs |
| Métriques de l'API | http://localhost:8080/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Node Exporter | http://localhost:9100/metrics |

Les identifiants Grafana par défaut sont `admin` / `admin`. Grafana peut demander de choisir un nouveau mot de passe à la première connexion.

Le premier démarrage est plus long, car Docker construit les images et l'API entraîne son modèle. Pour suivre le démarrage :

```bash
docker-compose logs -f bike-api
```

Pour arrêter les conteneurs :

```bash
make stop
```

## Commandes du Makefile

| Commande | Rôle |
| --- | --- |
| `make` ou `make all` | Construit et démarre l'API, Prometheus, Grafana, Node Exporter, l'évaluation et le simulateur de trafic. |
| `make stop` | Arrête les services du projet. Les volumes Prometheus et Grafana sont conservés. |
| `make evaluation` | Reconstruit et exécute le conteneur d'évaluation afin de mettre à jour les métriques du modèle et de dérive. |
| `make predict` | Reconstruit et démarre le générateur permanent de trafic sur `/predict`. |
| `make fire-alert` | Force `model_rmse_score` à `1000` pour tester l'alerte Grafana sur la RMSE. |

Après `make fire-alert`, attendre au moins 10 secondes pour que l'alerte Grafana passe en état d'alerte. Une nouvelle évaluation remet la RMSE à sa valeur réelle :

```bash
make evaluation
```

## Simulation permanente du trafic avec `predict`

Le service Docker `predict` exécute `src/predict/run_predict.py`. Ce script n'entraîne ni n'évalue le modèle : il simule uniquement des utilisateurs qui appellent continuellement l'endpoint `http://bike-api:8080/predict`.

Par défaut, il :

- tourne sans limite tant que son conteneur reste démarré ;
- envoie une requête toutes les 0,5 seconde ;
- génère environ 80 % de requêtes valides ;
- génère environ 20 % d'erreurs volontaires ;
- continue à fonctionner si l'API est temporairement indisponible.

Les requêtes erronées sont choisies aléatoirement parmi trois cas : un champ obligatoire absent, un type invalide et une mauvaise méthode HTTP (`GET` au lieu de `POST`). Elles produisent notamment des réponses `422` ou `405`. Ce mélange alimente les métriques de débit, de latence, de codes HTTP et de taux d'erreur du dashboard API.

Le service est déjà lancé par `make all`. `make predict` permet de le reconstruire et de le relancer séparément. Ses logs montrent le type et le statut de chaque requête ainsi que les compteurs cumulés :

```bash
docker-compose logs -f predict
```

Son comportement est configurable avec les variables d'environnement suivantes :

| Variable | Valeur par défaut | Description |
| --- | --- | --- |
| `PREDICT_URL` | `http://bike-api:8080/predict` | Endpoint appelé. |
| `REQUEST_INTERVAL_SECONDS` | `0.5` | Attente entre deux requêtes. |
| `ERROR_RATE` | `0.2` | Proportion d'erreurs simulées, entre `0` et `1`. |
| `REQUEST_COUNT` | `0` | Nombre de requêtes ; `0` signifie sans limite. |
| `REQUEST_TIMEOUT_SECONDS` | `5` | Timeout HTTP d'une requête. |

## Entraînement et prédiction

Au démarrage du conteneur API, le modèle de référence `RandomForestRegressor` est entraîné une fois, puis chargé pour l'inférence. Les données de référence sont strictement celles du mois de janvier 2011, du 1er au 31 janvier inclus.

L'endpoint `POST /predict` accepte les onze variables utilisées par le modèle. Exemple :

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

Le service `evaluation` télécharge le dataset Bike Sharing, prélève la période courante configurée dans `run_evaluation.py`, puis envoie au maximum 1 000 observations avec leur valeur réelle `cnt` à `POST /evaluate`.

L'API :

1. calcule les prédictions sur ces observations ;
2. compare les valeurs réelles et prédites ;
3. compare les données courantes aux données de référence de janvier avec Evidently ;
4. met à jour les gauges RMSE, MAE, R² et dérive exposées à Prometheus.

Docker Compose peut démarrer `evaluation` avant que l'API soit totalement prête. Pour absorber cette course au démarrage ou une indisponibilité passagère, le script tente l'appel jusqu'à 10 fois, avec 10 secondes d'attente entre deux échecs. Si les 10 tentatives échouent, le conteneur termine avec un code d'erreur non nul. Pendant ce temps, le service `predict` peut continuer à générer du trafic indépendamment.

## Métriques Prometheus

L'API expose notamment :

- `api_requests_total{endpoint,method,status_code}` : nombre de requêtes HTTP ;
- `api_request_duration_seconds{endpoint,method,status_code}` : histogramme des durées ;
- `model_rmse_score` : erreur quadratique moyenne racine ;
- `model_mae_score` : erreur absolue moyenne ;
- `model_r2_score` : coefficient de détermination ;
- `evidently_data_drift_detected_status` : statut binaire de dérive globale.

### Choix de la métrique personnalisée

La métrique personnalisée est `evidently_data_drift_detected_status`, une `Gauge` qui vaut :

- `0` lorsqu'aucune dérive globale n'est détectée ;
- `1` lorsqu'Evidently détecte une dérive entre les données courantes et la référence de janvier.

Cette métrique complète les scores RMSE, MAE et R². Ces scores mesurent la qualité des prédictions seulement lorsqu'une vérité terrain `cnt` est disponible. La dérive surveille, elle, l'évolution de la distribution des variables d'entrée. Elle peut donc signaler plus tôt que les données reçues ne ressemblent plus aux données d'entraînement, même si les scores du modèle ne se sont pas encore fortement dégradés. Son format binaire est aussi directement exploitable dans un dashboard et dans une règle d'alerte Prometheus.

La gauge est mise à jour lors de chaque appel réussi à `/evaluate`, pas par les appels à `/predict`.

## Dashboards et alertes Grafana

Au démarrage, Grafana charge automatiquement la datasource Prometheus, les dashboards JSON et l'alerte ML depuis `deployment/grafana/provisioning`. Aucune importation manuelle n'est nécessaire.

Dans le dossier Grafana `MLOps`, trois dashboards sont disponibles :

- **Bike API Observability** : débit, latence P95, erreurs et statuts HTTP de l'API ;
- **Bike Model Observability** : RMSE, MAE, R² et état de dérive ;
- **Node Exporter Full** : CPU, mémoire, disque et métriques de l'hôte.

Ces dashboards sont des versions modifées de fichiers récupérés dans le hub Grafana.

Les valeurs du dashboard modèle restent à zéro tant que l'évaluation n'a pas réussi au moins une fois. Pour les initialiser ou les actualiser :

```bash
make evaluation
```

L'alerte Grafana **Bike model RMSE too high** se déclenche si `model_rmse_score` reste supérieur à `100` pendant 10 secondes. `make fire-alert` appelle l'endpoint de test `/false_rmse`, qui fixe temporairement la RMSE à `1000`.

Prometheus charge également deux règles depuis `deployment/prometheus/rules/alert_rules.yml` :

- `BikeApiDown` si l'API est indisponible pendant une minute ;
- `DataDriftDetected` si la gauge de dérive vaut `1` pendant cinq minutes.

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

```bash
# État des conteneurs
docker-compose ps

# Logs de l'API
docker-compose logs -f bike-api

# Logs de l'évaluation
docker-compose logs evaluation

# Logs du simulateur permanent
docker-compose logs -f predict
```

Si les dashboards API sont vides, vérifier que `predict` est en cours d'exécution. Si le dashboard modèle est vide ou à zéro, exécuter `make evaluation` et vérifier que le conteneur `evaluation` termine sans erreur.
