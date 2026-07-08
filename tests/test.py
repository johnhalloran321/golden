"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""

from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from golden.golden_retriever import Golden_Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Final
import torch
from pathlib import Path
from git import Repo

try:
    from transformers import BitsAndBytesConfig
    bandb = True
except ImportError:
    bandb = False

from langchain.chains import RetrievalQA


NUM_TEST: Final[int] = 100
TOP_K: Final[int] = 10

# Directory where repositories are cached
CACHE_DIR = Path.home() / ".cache" / "code_rag"

# Repository information
REPO_NAME = "langchain"
REPO_URL = "https://github.com/langchain-ai/langchain.git"

# Clone once if necessary
repo_root = CACHE_DIR / REPO_NAME
if not repo_root.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_NAME}...")
    Repo.clone_from(REPO_URL, repo_root)
else:
    print(f"Using cached repository: {repo_root}")

# Source directory to index
source_dir = repo_root / "libs" / "langchain" / "langchain"

loader = GenericLoader.from_filesystem(
    str(source_dir),
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(
        language=Language.PYTHON,
        parser_threshold=500,
    ),
)
documents = loader.load()
python_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=20
)
texts = python_splitter.split_documents([document for document in documents])[0:NUM_TEST]

# metadata={"hnsw:space": "l2"}
# metadata={"hnsw:space": "cosine"}
metadata={"hnsw:space": "ip"}
if bandb:
    q_config = BitsAndBytesConfig(
    load_in_8it=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    q_config = {"yo": 1}

db = Golden_Retriever.from_documents(texts, 
                                     similarity_fn = "cosine",
                                     max_seq_length = 1200,
                                     max_batch_size = 512,
                                     batch_size = "auto",
                                     quantization_config = q_config,
                                     ) # collection_metadata = metadata)

question = "How can I load a source code as documents, for a QA over code, spliting the code in classes and functions?"
db._select_relevance_score_fn()
for d, score in db.similarity_search_with_score(question, k = TOP_K):
    print(score)
    print(d.page_content, score)
