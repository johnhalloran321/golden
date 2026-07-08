"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""

from golden.golden_retriever import Golden_Retriever
from typing import Final
from golden.golden_embeddings import Embedding
import argparse
import os

NUM_TEST: Final[int] = 100
TOP_K: Final[int] = 2
PERSIST_DIR = "Uruguay"

parser = argparse.ArgumentParser()
parser.add_argument('--test-write-db', action='store_true')
parser.add_argument('--test-load-db', action='store_true')
args = parser.parse_args()

texts = ["Uruguay (official full name in  ; pron.  , Eastern Republic of  Uruguay) is a country located in the southeastern part of South America.  It is home to 3.3 million people, of which 1.7 million live in the capital Montevideo and its metropolitan area."]

if args.test_write_db:
    embedding = Embedding(model_id = "sentence-transformers/all-MiniLM-L6-v2",
                          tokenizer_id = "sentence-transformers/all-MiniLM-L6-v2",
                          )
    if os.path.isdir(PERSIST_DIR):
        try:
            from shutil import rmtree
            rmtree(PERSIST_DIR)
        except OSError as e:
            print(f"Persist directory {PERSIST_DIR} could not be removed")
            raise e    
    db = Golden_Retriever.from_texts(texts,
                                    embedding = embedding,
                                    similarity_fn = "l2", # one of l2, cosine, ip
                                    # model_id = "sentence-transformers/all-MiniLM-L6-v2",
                                    # tokenizer_id = "sentence-transformers/all-MiniLM-L6-v2",
                                    chunk_size = 150,
                                    chunk_overlap = 10,
                                    max_batch_size = 512,
                                    batch_size = "auto",
                                    persist_directory = PERSIST_DIR,
                                    )
if args.test_load_db and os.path.isdir(PERSIST_DIR):
    try:
        db = Golden_Retriever.load(PERSIST_DIR)
        print(f"Successfully loaded Vector DB from {PERSIST_DIR}")
    except:
        raise
    

question = "Uruguay"
for d, score in db.similarity_search_with_score(question, k = TOP_K):
    print(d.page_content, score)
