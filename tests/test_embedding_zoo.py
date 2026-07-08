"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""

from golden.golden_retriever import Golden_Retriever
from typing import Final
from golden.golden_embeddings import MODEL_ZOO
import argparse
import os

TOP_K: Final[int] = 2
PERSIST_DIR = "Uruguay"

supported_embeddings = [ "sentence-transformers/all-MiniLM-L6-v2",
                        "togethercomputer/m2-bert-80M-8k-retrieval",
                        "thenlper/gte-large",
                        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
]
message = "Supported embeddings:\n\t" + "\n\t".join(supported_embeddings)
parser = argparse.ArgumentParser()
parser.add_argument('--embedding', type = str, default = "thenlper/gte-large", help = message)
args = parser.parse_args()

texts = ["Uruguay (official full name in  ; pron.  , Eastern Republic of  Uruguay) is a country located in the southeastern part of South America.  It is home to 3.3 million people, of which 1.7 million live in the capital Montevideo and its metropolitan area."]

if args.embedding in MODEL_ZOO:
    model_id = args.embedding
    model = MODEL_ZOO[model_id]["alias"]
    tokenizer_id = MODEL_ZOO[model_id]["tokenizer_id"]

    print(model, model_id, tokenizer_id)
    if os.path.isdir(PERSIST_DIR):
        try:
            from shutil import rmtree
            rmtree(PERSIST_DIR)
        except OSError as e:
            print(f"Persist directory {PERSIST_DIR} could not be removed")
            raise e    
    db = Golden_Retriever.from_texts(texts,
                                    embedding = None,
                                    similarity_fn = "cosine",
                                    model_id = model_id,
                                    tokenizer_id = tokenizer_id,
                                    chunk_size = 50,
                                    chunk_overlap = 0,
                                    max_batch_size = 512,
                                    batch_size = "auto",
                                    persist_directory = PERSIST_DIR,
                                    )
    question = "Uruguay"
    for d, score in db.similarity_search_with_score(question, k = TOP_K):
        print(d.page_content, score)
else:
    print(f"{args.embedding} not in supported embeddings list.")
    print(message)
