"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""

from golden.golden_faiss_retriever import Golden_Faiss_Retriever
from golden.golden_retriever import Golden_Retriever
import csv
import gzip
import time
import os
from langchain_core.documents import Document


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no-faiss', action='store_true')
parser.add_argument('--no-chroma', action='store_true')
args = parser.parse_args()


# Current directory
cwd = os.getcwd()

# where data should be
x = os.path.join(cwd, "..", "golden", "data", "first100_psgs.tsv.gz")

if not os.path.isfile(x):
    raise Exception(f"Test file {x} does not exist.")

with gzip.open(x, "rt") as f:
    texts = []
    t0 = time.time()
    for row in csv.reader(f, delimiter="\t", ): 
        if row[0]!="id":
            if row[2]:
                text = row[2] + " " + row[1] 
            else:
                text = row[1]
            texts.append(text)
    t1 = time.time()
    print(f"Text corpus load time: {t1-t0}s")

    # Convert to docs
    texts = [Document(text) for text in texts]

    if not args.no_faiss:
        t0 = time.time()
        db = Golden_Faiss_Retriever.from_documents(texts,
                                                max_seq_length = 512,
                                                batch_size = "auto",
                                                max_batch_size = 540,
                                                num_workers = 10,
                                                persist_directory = "wiki",
                                                do_chunking = False,
                                                )
        t1 = time.time()
        print(f"FAISS Vector DB build time: {t1-t0}s")
        db.persist()

        try:
            db = Golden_Faiss_Retriever.load("wiki")
            print(f"Successfully loaded FAISS Vector DB")
        except:
            raise

    if not args.no_chroma:
        t0 = time.time()
        db = Golden_Retriever.from_documents(texts,
                                             max_seq_length = 512,
                                             batch_size = "auto",
                                             max_batch_size = 540,
                                             num_workers = 10,
                                             persist_directory = "wiki",
                                             do_chunking = False,
                                             )
        t1 = time.time()
        print(f"Chroma Vector DB build time: {t1-t0}s")
        # db.persist()

        try:
            db = Golden_Retriever.load("wiki")
            print(f"Successfully loaded Chroma Vector DB")
        except:
            raise
