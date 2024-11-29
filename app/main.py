from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from kaggle import api
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Kaggle Dataset Identifier
KAGGLE_DATASET = "abhinavwalia95/entity-annotated-corpus"
DATA_FILE = "ner_dataset.csv"
KAGGLE_DIR = "kaggle_data"


# Kaggle Authentication
def kaggle_authenticate():
    """
    Authenticate Kaggle API using environment variables or kaggle.json.
    """
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        logging.info("Using Kaggle environment variables for authentication.")
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
    else:
        logging.info("Using kaggle.json for authentication.")
        api.authenticate()


# Authenticate Kaggle on startup
kaggle_authenticate()

# Models
class Entity(BaseModel):
    name: str
    type: str


class Relation(BaseModel):
    source: str
    target: str
    type: str


class AggregatedData(BaseModel):
    entities: List[Entity]
    relations: List[Relation]


@app.get("/")
async def root():
    return {"message": "API is running!"}


# 1. Extraction depuis data.gouv.fr
@app.get("/data-gouv")
async def fetch_data_gouv(query: str = "education"):
    """
    Fetch datasets from data.gouv.fr.
    """
    url = f"https://www.data.gouv.fr/api/1/datasets/?q={query}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        datasets = response.json().get("data", [])
        results = [
            {
                "title": dataset.get("title"),
                "description": dataset.get("description", "No description available"),
                "last_update": dataset.get("last_update"),
                "url": dataset.get("page"),
            }
            for dataset in datasets
        ]
        return {"results": results}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


# 2. Extraction depuis DBpedia
@app.get("/dbpedia")
async def fetch_dbpedia(entity: str = "Paris"):
    """
    Query entity data from DBpedia.
    """
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    query = f"""
    SELECT ?property ?value
    WHERE {{
        dbr:{entity} ?property ?value
    }}
    LIMIT 20
    """
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        data = [
            {
                "property": res["property"]["value"],
                "value": res["value"]["value"],
            }
            for res in results["results"]["bindings"]
        ]
        return {
            "entity": entity,
            "description": f"Information about {entity} from DBpedia",
            "data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying DBpedia: {str(e)}")


# 3. Extraction depuis un fichier Kaggle
@app.get("/kaggle-ner")
async def fetch_kaggle_data():
    """
    Fetch and process Kaggle's Named Entity Recognition dataset.
    """
    # Ensure Kaggle directory exists
    os.makedirs(KAGGLE_DIR, exist_ok=True)

    try:
        # Check if the dataset file exists
        dataset_path = os.path.join(KAGGLE_DIR, DATA_FILE)
        if not os.path.exists(dataset_path):
            logging.info("Downloading dataset from Kaggle...")
            api.dataset_download_files(KAGGLE_DATASET, path=KAGGLE_DIR, unzip=True)
            logging.info(f"Dataset downloaded to {KAGGLE_DIR}")

        # Load and process the dataset
        df = pd.read_csv(dataset_path, encoding="latin1")
        df_cleaned = df.dropna(subset=["Word", "Tag"]).head(100)  # Example cleaning step

        results = [
            {
                "word": row["Word"],
                "pos": row["POS"],
                "ner_tag": row["Tag"],
            }
            for _, row in df_cleaned.iterrows()
        ]
        return {"results": results}

    except Exception as e:
        logging.error(f"Error processing Kaggle data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing Kaggle data: {str(e)}")


# 4. Agrégation et nettoyage
@app.post("/aggregate")
async def aggregate_data(data_sources: List[AggregatedData]):
    """
    Aggregate and clean data from multiple sources.
    """
    all_entities = []
    all_relations = []

    for source in data_sources:
        all_entities.extend(source.entities)
        all_relations.extend(source.relations)

    unique_entities = {e.name: e for e in all_entities}.values()
    unique_relations = {(r.source, r.target, r.type): r for r in all_relations}.values()

    return {
        "aggregated_entities": list(unique_entities),
        "aggregated_relations": list(unique_relations),
    }
