import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent

def download_file(url, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        print(f"{save_path.name} ya existe, se usa localmente.")
        return
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {save_path.name} from GitHub")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Archivo {url} no encontrado en GitHub, se omite.")
        else:
            raise

def fetch_artifacts():
    files = {
        "model": "models/best_model.pkl",
        "model_card": "models/model_card_LogReg_C1.0_lbfgs.json",
    }

    repo_base = "https://github.com/kngarcia/emotion_analisis_nlp.git"

    local_paths = {}
    for key, path in files.items():
        url = repo_base + path
        local_path = BASE_DIR / path
        download_file(url, local_path)
        local_paths[key] = local_path
    return local_paths