from fastapi.testclient import TestClient
import kaggle.api
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_data_gouv():
    response = client.get("/data-gouv?query=education")
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)

def test_dbpedia():
    response = client.get("/dbpedia?entity=Paris")
    assert response.status_code == 200
    assert "entity" in response.json()
    assert "description" in response.json()
    assert "data" in response.json()

def test_kaggle_ner(monkeypatch):
    import pandas as pd

    # Mock dataset download
    def mock_download_files(dataset, path, unzip):
        pass

    monkeypatch.setattr(kaggle.api, "dataset_download_files", mock_download_files)

    # Mock pandas.read_csv
    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame({
            "Word": ["Paris", "Eiffel"],
            "POS": ["NNP", "NNP"],
            "Tag": ["B-LOC", "I-LOC"]
        })

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    response = client.get("/kaggle-ner")
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 2


def test_aggregate():
    payload = [
        {
            "entities": [{"name": "Paris", "type": "Location"}],
            "relations": [{"source": "Paris", "target": "Eiffel Tower", "type": "LocatedIn"}]
        },
        {
            "entities": [{"name": "Eiffel Tower", "type": "Landmark"}],
            "relations": [{"source": "Paris", "target": "Eiffel Tower", "type": "LocatedIn"}]
        }
    ]
    response = client.post("/aggregate", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "aggregated_entities" in result
    assert "aggregated_relations" in result
    assert len(result["aggregated_entities"]) == 2
    assert len(result["aggregated_relations"]) == 1
