from fastapi import FastAPI, HTTPException, Depends, UploadFile, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain_ollama.embeddings import OllamaEmbeddings
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
import pandas as pd
import os
import json
import requests
import logging
from datetime import datetime
import torch
from dotenv import load_dotenv
import tempfile
from gliner import GLiNER, GLiNERConfig  # Assurez-vous que gliner est installé et accessible

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Mega Data API",
    description="A comprehensive API for data ingestion, processing, and management.",
    version="2.1.0",
)

# Configuration CORS si nécessaire
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration de la base de données
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle SQLAlchemy Mis à Jour
class DataModel(Base):
    __tablename__ = "scrapflow_data"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    entity_name = Column(String, index=True)  # Champ ajouté pour les jointures
    meta_data = Column(JSON, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)

Base.metadata.create_all(bind=engine)

# Dépendance pour la session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Schémas Pydantic
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

class DataResponse(BaseModel):
    source: str
    content: Union[str, Dict]
    meta_data: Optional[Dict] = None

class AggregatedDataRequest(BaseModel):
    sources: List[DataResponse]

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class ExportResponse(BaseModel):
    message: str

class QueryFilters(BaseModel):
    field: str
    value: str

# Configuration Kaggle
KAGGLE_DATASET = "abhinavwalia95/entity-annotated-corpus"
DATA_FILE = "ner_dataset.csv"
KAGGLE_DIR = "kaggle_data"

def kaggle_authenticate():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
    else:
        logging.warning("Kaggle credentials not found in environment variables.")

kaggle_authenticate()

# Service d'Embedding
class EmbeddingService:
    def __init__(self, model_name: str = "llama3.2"):
        self.embedding_model = OllamaEmbeddings(model=model_name)

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return []

embedding_service = EmbeddingService()

# Service de Nettoyage des Données
class DataCleaningService:
    @staticmethod
    def clean_text(text: str) -> str:
        # Exemple de nettoyage : suppression des espaces en trop, des caractères spéciaux, etc.
        cleaned = text.strip().replace("\n", " ").replace("\r", "")
        return cleaned

    @staticmethod
    def validate_data(data: Dict) -> bool:
        # Implémentez des validations spécifiques selon vos besoins
        return True

data_cleaning_service = DataCleaningService()

# Service de Transformation en CoNLL-U
class ConlluService:
    def __init__(self):
                self.gl = GLiNER(GLiNERConfig())
    def to_conllu(self, data: List[Dict]) -> str:
        """
        Convertit les données agrégées en format CoNLL-U.

        :param data: Liste de dictionnaires contenant les entités agrégées.
        :return: Chaîne de caractères au format CoNLL-U.
        """
        conllu_sentences = []
        for entry in data:
            tokens = entry.get("tokens", [])
            entities = entry.get("entities", [])

            # Générer les informations CoNLL-U pour chaque token
            conllu_tokens = []
            for idx, token in enumerate(tokens, start=1):
                # Identifier si le token fait partie d'une entité
                ner_tag = "O"
                for entity in entities:
                    if token.lower() == entity["text"].lower():
                        ner_tag = entity["tag"]
                        break
                conllu_tokens.append(f"{idx}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{ner_tag}")

            # Joindre les tokens pour former une phrase CoNLL-U
            sentence = "\n".join(conllu_tokens) + "\n"
            conllu_sentences.append(sentence)

        # Joindre toutes les phrases
        return "\n".join(conllu_sentences)

conllu_service = ConlluService()

# Fonctions de Récupération de Données
async def retrieve_data_gouv(query: str, limit: int, db: Session):
    url = f"https://www.data.gouv.fr/api/1/datasets/?q={query}&page_size={limit}"
    response = requests.get(url)
    response.raise_for_status()
    datasets = response.json().get("data", [])

    cleaned_items = []
    for dataset in datasets:
        cleaned_content = data_cleaning_service.clean_text(dataset.get("page", ""))
        if data_cleaning_service.validate_data(dataset):
            entity_name = dataset.get("title", "").lower()  # Exemple de normalisation
            db_item = DataModel(
                source="data-gouv",
                entity_name=entity_name,
                meta_data={
                    "title": dataset.get("title"),
                    "description": dataset.get("description"),
                    "last_update": dataset.get("last_update"),
                },
                content=cleaned_content,
            )
            db.add(db_item)
            cleaned_items.append(db_item)
    db.commit()
    return cleaned_items

async def retrieve_dbpedia(query: str, limit: int, db: Session):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(f"""
        SELECT ?s ?p ?o
        WHERE {{
            ?s ?p ?o .
            FILTER(CONTAINS(LCASE(STR(?s)), "{query.lower()}"))
        }}
        LIMIT {limit}
    """)
    sparql.setReturnFormat(SPARQL_JSON)
    results = sparql.query().convert()

    cleaned_items = []
    for result in results["results"]["bindings"]:
        entity_name = result["s"]["value"].split('/')[-1].replace('_', ' ').lower()
        cleaned_content = data_cleaning_service.clean_text(result["s"]["value"])
        if data_cleaning_service.validate_data(result):
            db_item = DataModel(
                source="dbpedia",
                entity_name=entity_name,
                meta_data={
                    "predicate": result["p"]["value"],
                    "object": result["o"]["value"],
                },
                content=cleaned_content,
            )
            db.add(db_item)
            cleaned_items.append(db_item)
    db.commit()
    return cleaned_items

async def retrieve_kaggle_ner(db: Session):
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    dataset_path = os.path.join(KAGGLE_DIR, DATA_FILE)

    if not os.path.exists(dataset_path):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=KAGGLE_DIR, unzip=True)

    df = pd.read_csv(dataset_path, encoding="latin1").dropna(subset=["Word", "Tag"]).head(100)
    data = df.to_dict(orient="records")

    cleaned_items = []
    for row in data:
        cleaned_word = data_cleaning_service.clean_text(row["Word"])
        entity_name = cleaned_word.lower()
        if data_cleaning_service.validate_data(row):
            db_item = DataModel(
                source="kaggle-ner",
                entity_name=entity_name,
                meta_data={"pos": row.get("POS"), "ner_tag": row.get("Tag")},
                content=cleaned_word
            )
            db.add(db_item)
            cleaned_items.append(db_item)
    db.commit()
    return cleaned_items

async def retrieve_uploaded_data(db: Session):
    # Supposons que les données uploadées sont déjà dans la base de données
    uploaded_data = db.query(DataModel).filter(DataModel.source == "uploaded").all()
    return uploaded_data

# Routes Existantes (Simplifiées pour la Clarté)
@app.get("/")
async def root():
    return {"message": "API is up and running!"}

@app.get("/data-gouv")
async def fetch_data_gouv_endpoint(query: str, limit: int = 10, db: Session = Depends(get_db)):
    return await retrieve_data_gouv(query, limit, db)

@app.get("/dbpedia")
async def fetch_dbpedia_endpoint(query: str, limit: int = 10, db: Session = Depends(get_db)):
    return await retrieve_dbpedia(query, limit, db)

@app.get("/kaggle-ner")
async def fetch_kaggle_data_endpoint(db: Session = Depends(get_db)):
    return await retrieve_kaggle_ner(db)

@app.post("/upload-data")
async def upload_data(file: UploadFile, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Endpoint pour uploader un fichier de données et le traiter en arrière-plan.
    """
    try:
        content = await file.read()
        data = json.loads(content)

        for item in data:
            cleaned_content = data_cleaning_service.clean_text(item.get("content", ""))
            entity_name = cleaned_content.lower()
            if data_cleaning_service.validate_data(item):
                db_item = DataModel(
                    source="uploaded",
                    entity_name=entity_name,
                    meta_data=item.get("meta_data"),
                    content=cleaned_content
                )
                db.add(db_item)
        db.commit()

        # Optionnel : Ajouter une tâche en arrière-plan pour traiter les données uploadées
        # background_tasks.add_task(process_uploaded_data, db)

        return {"message": "File uploaded and processing completed."}
    except Exception as e:
        logging.error(f"Failed to upload data: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload data")

# Nouvelle Route pour Exécuter le Pipeline Complet
@app.post("/run-pipeline")
async def run_pipeline(query: str, limit: int = 10, db: Session = Depends(get_db)):
    """
    Exécute l'ensemble du pipeline :
    1. Récupère et nettoie les données des sources.
    2. Fusionne les données dans une table SQL via des jointures.
    3. Enrichit la table avec des entités NER générées par GLiNER.
    4. Génère et sauvegarde un dataset NER au format SQL et CoNLL-U.
    """
    try:
        # Étape 1 : Récupérer et nettoyer les données
        logging.info("Step 1: Retrieving and cleaning data...")
        data_gouv = await retrieve_data_gouv(query, limit, db)
        dbpedia = await retrieve_dbpedia(query, limit, db)
        kaggle_ner = await retrieve_kaggle_ner(db)
        uploaded_data = await retrieve_uploaded_data(db)

        # Étape 2 : Fusionner les données
        logging.info("Step 2: Merging data...")
        entities = db.query(DataModel.entity_name).distinct().all()
        entities = [entity[0] for entity in entities]

        merged_data = []
        for entity in entities:
            records = db.query(DataModel).filter(DataModel.entity_name == entity).all()
            aggregated_meta = {}
            content = ""
            for record in records:
                # Fusionner les contenus
                content += f" {record.content}"
                # Fusionner les métadonnées
                if record.meta_data:
                    for key, value in record.meta_data.items():
                        aggregated_meta.setdefault(key, []).append(value)

            # Nettoyage final de la fusion
            content = data_cleaning_service.clean_text(content)

            merged_data.append({
                "entity_name": entity,
                "meta_data": aggregated_meta,
                "content": content
            })

        # Étape 3 : Enrichir avec GLiNER
        logging.info("Step 3: Enriching data with GLiNER...")
        enriched_data = []
        for data in merged_data:
            content = data["content"]
        ner_results = conllu_service.gl.predict_entities(
            text=content,  # Liste des contenus à analyser
            labels=[],  # Liste des labels NER (facultatif)
            flat_ner=True,  # Utiliser une structure plate
            threshold=0.5,  # Seuil de confiance pour les prédictions
            multi_label=True  # Autoriser plusieurs entités par token
        )           

        enriched_data.append({
                "entity_name": data["entity_name"],
                "meta_data": data["meta_data"],
                "content": content,
                "entities": ner_results
            })

        # Sauvegarder les données enrichies en base de données
        logging.info("Step 4: Saving enriched data to SQL...")
        for data in enriched_data:
            db_item = DataModel(
                source="enriched-pipeline",
                entity_name=data["entity_name"],
                meta_data=data["meta_data"],
                content=data["content"]
            )
            db.add(db_item)
        db.commit()

        # Étape 4 : Générer le dataset CoNLL-U
        logging.info("Step 5: Generating NER dataset in CoNLL-U format...")
        conllu_data = conllu_service.to_conllu(enriched_data)

        # Sauvegarder le dataset CoNLL-U dans un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.conllu') as tmp_file:
            tmp_file.write(conllu_data)
            tmp_filename = tmp_file.name

        # Retourner le fichier CoNLL-U
        return FileResponse(
            path=tmp_filename,
            media_type='text/plain',
            filename='ner_dataset.conllu'
        )

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail="Pipeline execution failed")

# Route d'exportation des données
@app.post("/export-data")
async def export_data(
    formats: List[str] = Body(
        ...,  # Ce champ est requis
        title="Formats d'exportation",
        description="Liste des formats dans lesquels exporter les données.",
        example=["json", "csv", "conllu"],
    ),
    db: Session = Depends(get_db),
):
    """
    Exporte les données enrichies dans les formats spécifiés : JSON, CSV ou CoNLL-U.
    """
    try:
        # Récupérer toutes les données enrichies depuis la base de données
        enriched_data = db.query(DataModel).filter(DataModel.source == "enriched-pipeline").all()

        # Convertir les données enrichies en un format commun
        data_list = [
            {
                "entity_name": data.entity_name,
                "meta_data": data.meta_data,
                "content": data.content,
                "entities": data.embedding,  # Contient les entités NER (par exemple)
            }
            for data in enriched_data
        ]

        # Initialiser un dictionnaire pour les fichiers exportés
        exported_files = {}

        # Générer un fichier JSON si demandé
        if "json" in formats:
            json_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
            json.dump(data_list, json_file, indent=4)
            json_file.close()
            exported_files["json"] = json_file.name

        # Générer un fichier CSV si demandé
        if "csv" in formats:
            csv_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
            df = pd.DataFrame(data_list)
            df.to_csv(csv_file.name, index=False)
            exported_files["csv"] = csv_file.name

        # Générer un fichier CoNLL-U si demandé
        if "conllu" in formats:
            conllu_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.conllu')
            conllu_data = conllu_service.to_conllu(data_list)
            conllu_file.write(conllu_data)
            conllu_file.close()
            exported_files["conllu"] = conllu_file.name

        # Si un seul format est demandé, retourner directement le fichier
        if len(exported_files) == 1:
            format_name, file_path = next(iter(exported_files.items()))
            return FileResponse(
                path=file_path,
                media_type="application/octet-stream",
                filename=f"exported_data.{format_name}"
            )

        # Si plusieurs formats sont demandés, retourner un fichier ZIP
        zip_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with tempfile.NamedTemporaryFile(delete=False) as zip_buffer:
            import zipfile
            with zipfile.ZipFile(zip_file.name, 'w') as zf:
                for format_name, file_path in exported_files.items():
                    zf.write(file_path, arcname=f"exported_data.{format_name}")

        return FileResponse(
            path=zip_file.name,
            media_type="application/zip",
            filename="exported_data.zip"
        )

    except Exception as e:
        logging.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail="Exportation des données échouée")