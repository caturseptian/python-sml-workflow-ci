# python-sml-workflow-ci - Tian

Repository **Kriteria 3** submission MSML: CI dengan **MLflow Project** + **GitHub Actions**,
build & push **Docker image** ke Docker Hub (Advanced).

> Repository ini **HARUS public**.

## Struktur

```
python-sml-workflow-ci/
├── .github/workflows/ci.yml         # workflow CI (push & workflow_dispatch)
└── MLProject/
    ├── MLProject                    # definisi MLflow Project
    ├── conda.yaml                   # environment (python 3.12.7)
    ├── modelling.py                 # training, dijalankan via `mlflow run`
    ├── breast_cancer_preprocessing/ # dataset hasil preprocessing (train/test)
    └── DockerHub.txt                # tautan image Docker Hub
```

## Alur CI (`.github/workflows/ci.yml`)

Trigger: **push** ke `main`/`master` dan **workflow_dispatch** (manual).

Tahapan:
1. Checkout (`actions/checkout@v3`)
2. Set up Python **3.12.7**
3. Check Env
4. Install dependencies (mlflow 2.19.0 + sklearn stack)
5. **Run mlflow project** (`mlflow run ./MLProject --env-manager=local`)
6. Get latest MLflow `run_id`
7. Upload artefak MLflow ke GitHub (`actions/upload-artifact`)
8. **Build Docker Model** (`mlflow models build-docker`)
9. Login Docker Hub
10. Tag image
11. **Push image** ke Docker Hub

## Menjalankan MLflow Project secara lokal

```bash
pip install mlflow==2.19.0 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 joblib==1.4.2
cd python-sml-workflow-ci
mlflow run ./MLProject --env-manager=local -P n_estimators=300
```

## Mengatur GitHub Secrets (WAJIB untuk push Docker)

Repo GitHub → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Nilai |
|--------|-------|
| `DOCKERHUB_USERNAME` | `caturseptian` |
| `DOCKERHUB_TOKEN` | Personal Access Token dari Docker Hub (Account Settings → Security → New Access Token) |

Setelah secret diisi, jalankan workflow (push atau **Actions → Run workflow**).
Image akan terbit di: `https://hub.docker.com/r/caturseptian/breast-cancer-model`.
