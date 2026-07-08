"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""
from langchain_community.vectorstores.chroma import *

from golden.golden_embeddings import (Embedding,
                                      golden_embedding_options,
                                      MODEL_ID,
                                      TOKENIZER_ID,
                                      MAX_SEQ_LENGTH,
                                      BATCH_SIZE,
                                      NUM_WORKERS,
                                      CHUNK_OVERLAP,
                                      DEVICE,
                                      DEFAULT_DEVICE,
                                      split_documents_given_language,
                                      )
import chromadb
import uuid
from typing import Final
import traceback
import os
from tqdm import tqdm

import logging
import sys
import json

from golden.utils import clear_torch_cache, split_list

logger = logging.getLogger(__name__)
format = '%(asctime)s %(message)s'
logging.basicConfig(filename="golden_retriever.log",
                    format=format,
                    filemode='w',
                    level=logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)

SIMILARITY_KEY: Final[str] = "hnsw:space"
VALID_SIMILARITY_FNS: Final[set] = {"l2", "ip", "cosine"}
EMBEDDING_SETTING_FILE = "embedding_settings.json"

# ChromaDB's upsert is limited to 5461 vectors per call on some backends.
# Batching above this threshold prevents silent truncation or API errors.
MAX_UPSERT: Final[int] = 5461


def set_similarity_fn(fn: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> Dict:
    """Set the similarity function key in the collection metadata dict.

    Always returns the (possibly newly-created) metadata dict.
    Bug fix: the original had no return statement, so the result was always None.
    """
    if metadata is None:
        metadata = {}

    if fn:
        fn = fn.lower()
        if fn in VALID_SIMILARITY_FNS:
            metadata[SIMILARITY_KEY] = fn
        else:
            metadata.setdefault(SIMILARITY_KEY, "l2")
    else:
        metadata.setdefault(SIMILARITY_KEY, "l2")

    return metadata


class Golden_Retriever(Chroma):
    """`ChromaDB` vector store.

    To use, you should have the ``chromadb`` python package installed.

    Example:
        .. code-block:: python

                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma("langchain_store", embeddings)
    """
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.Client] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        similarity_fn: Optional[str] = None,  # l2 = euclidean, cosine, ip = inner product
        **kwargs,
    ) -> None:

        if embedding_function is None:
            kwargs.setdefault("model_id", MODEL_ID)
            kwargs.setdefault("tokenizer_id", TOKENIZER_ID)
            kwargs.setdefault("max_seq_length", MAX_SEQ_LENGTH)
            embedding_function = Embedding(**golden_embedding_options(kwargs))
        else:
            # Sync any embedding attributes into kwargs for downstream use
            for key in ["model_id", "tokenizer_id", "max_seq_length", "max_batch_size", "batch_size"]:
                if hasattr(embedding_function, key):
                    kwargs[key] = getattr(embedding_function, key)

        if persist_directory:
            self.write_embedding_settings_on_init(embedding_function, persist_directory)

        # Bug fix: set_similarity_fn now returns the updated dict
        collection_metadata = set_similarity_fn(similarity_fn, collection_metadata)

        super().__init__(
            collection_name,
            embedding_function,
            persist_directory,
            client_settings,
            collection_metadata,
            client,
            relevance_score_fn,
        )

    @staticmethod
    def write_embedding_settings_on_init(embedding_function, persist_directory: str) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        output_file = os.path.join(persist_directory, EMBEDDING_SETTING_FILE)
        if embedding_function and isinstance(embedding_function, Embedding):
            if hasattr(embedding_function, "model_id"):
                embedding_settings: Dict[str, Any] = {
                    "model_id": embedding_function.model_id,
                    "tokenizer_id": getattr(embedding_function, "tokenizer_id", TOKENIZER_ID),
                    "batch_size": getattr(embedding_function, "batch_size", 1),
                    "max_batch_size": getattr(embedding_function, "max_batch_size", 1),
                }
                # TODO: add support for quantization config
                logger.info(f"Saving embedding settings to {output_file}")
                logger.debug(f"Embedding settings: {embedding_settings}")
                with open(output_file, "w") as f:
                    json.dump(embedding_settings, f)

    def clean_then_persist(self) -> None:
        """Persist the collection.

        Since Chroma 0.4.x the manual persistence method is no longer
        supported as docs are automatically persisted.
        """
        major, minor, _ = chromadb.__version__.split(".")
        if int(major) == 0 and int(minor) < 4:
            if self._persist_directory is None:
                raise ValueError(
                    "You must specify a persist_directory on "
                    "creation to persist the collection."
                )
            if os.path.isdir(self._persist_directory):
                try:
                    from shutil import rmtree
                    rmtree(self._persist_directory)
                except OSError as e:
                    logger.error(
                        f"Persist directory {self._persist_directory} could not be removed: {e}"
                    )
                    raise
            if self._embedding_function is not None:
                self.write_embedding_settings_on_init(
                    self._embedding_function, self._persist_directory
                )
            self._client.persist()
        else:
            logger.info(
                "Since Chroma 0.4.x the manual persistence method is no longer supported "
                "as docs are automatically persisted."
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Texts to add to the vectorstore.
            metadatas: Optional list of metadatas per text.
            ids: Optional list of IDs; generated if not provided or mismatched.
            num_workers: Override embedding's num_workers if provided.
            device: Override embedding's device if provided.
            batch_size: Override embedding's batch_size if provided.

        Returns:
            List of IDs of the added texts.
        """
        embeddings = None
        texts = list(texts)

        logger.info(f"Adding {len(texts)} texts to collection")
        if batch_size is not None:
            logger.debug(f"Using batch_size override: {batch_size}")

        if self._embedding_function is not None:
            if isinstance(self._embedding_function, Embedding):
                logger.debug(f"Generating embeddings with num_workers={num_workers or 'default'}")
                embeddings = self._embedding_function.embed_documents(
                    texts=texts,
                    num_workers=num_workers,  # uses instance default if None
                    device=device,
                    batch_size=batch_size,    # uses instance default if None
                )
            else:
                embeddings = self._embedding_function.embed_documents(texts)

        if ids is None or len(ids) != len(texts):
            ids = [str(uuid.uuid1()) for _ in texts]

        if metadatas:
            # Fill metadatas with empty dicts where not provided
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff

            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)

            if non_empty_ids:
                metadatas_with = [metadatas[idx] for idx in non_empty_ids]
                texts_with = [texts[idx] for idx in non_empty_ids]
                embeddings_with = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas_with,
                        embeddings=embeddings_with,
                        documents=texts_with,
                        ids=ids_with,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise

            if empty_ids:
                texts_without = [texts[j] for j in empty_ids]
                embeddings_without = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without,
                    documents=texts_without,
                    ids=ids_without,
                )

        else:
            # No metadata path: batch upserts at MAX_UPSERT to avoid ChromaDB
            # backend limits that silently truncate or raise errors above 5461 vectors.
            if len(texts) > MAX_UPSERT:
                num_batches = len(texts) // MAX_UPSERT + 1
                logger.info(
                    f"Greater than {MAX_UPSERT} vectors detected, "
                    f"splitting into {num_batches} upsert batches"
                )
                for i in tqdm(range(0, len(texts), MAX_UPSERT)):
                    self._collection.upsert(
                        embeddings=embeddings[i:i + MAX_UPSERT],
                        documents=texts[i:i + MAX_UPSERT],
                        ids=ids[i:i + MAX_UPSERT],
                    )
            else:
                self._collection.upsert(
                    embeddings=embeddings,
                    documents=texts,
                    ids=ids,
                )

        logger.info(f"Successfully added {len(ids)} texts to collection")
        return ids

    def update_documents(self,
                         ids: List[str],
                         documents: List[Document],
                         language: str = "",
                         chunk_size: Optional[int] = MAX_SEQ_LENGTH,
                         chunk_overlap: Optional[int] = CHUNK_OVERLAP,
                         num_workers: Optional[int] = None,
                         device: Optional[str] = None,
                         batch_size: Optional[int] = None,
                         **kwargs,
                         ) -> None:
        """Update documents in the collection.

        Args:
            ids: List of document IDs to update.
            documents: Replacement documents.
            num_workers: Override embedding's num_workers if provided.
            device: Override embedding's device if provided.
            batch_size: Override embedding's batch_size if provided.
        """
        texts = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]

        if self._embedding_function is None:
            raise ValueError(
                "For update, you must specify an embedding function on creation."
            )

        # Note: embed_documents handles tokenisation; chunking is a pre-processing
        # step for callers (from_texts / from_documents), not part of embedding.
        if isinstance(self._embedding_function, Embedding):
            embeddings = self._embedding_function.embed_documents(
                texts=texts,
                num_workers=num_workers,
                device=device,
                batch_size=batch_size,
            )
        else:
            embeddings = self._embedding_function.embed_documents(texts)

        if hasattr(self._collection._client, "max_batch_size"):  # Chroma >= 0.4.10
            from chromadb.utils.batch_utils import create_batches
            for batch in create_batches(
                api=self._collection._client,
                ids=ids,
                metadatas=metadata,
                documents=texts,
                embeddings=embeddings,
            ):
                self._collection.update(
                    ids=batch[0],
                    embeddings=batch[1],
                    documents=batch[3],
                    metadatas=batch[2],
                )
        else:
            self._collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata,
            )

    @classmethod
    def load(cls,
             folder_path: str,
             **kwargs: Any,
             ):
        similarity_fn = kwargs.pop("similarity_fn", "cosine")
        embedding = kwargs.pop("embedding", None)
        embedding_settings_file = os.path.join(folder_path, EMBEDDING_SETTING_FILE)
        if os.path.isfile(embedding_settings_file):
            logger.info(f"Loading embedding settings from {embedding_settings_file}")
            with open(embedding_settings_file, "r") as json_file:
                embedding_settings = json.load(json_file)
                logger.debug(f"Loaded embedding settings: {embedding_settings}")
                if embedding_settings.get("model_id") and embedding_settings.get("tokenizer_id"):
                    embedding = Embedding(**embedding_settings)
        return cls(
            persist_directory=folder_path,
            embedding_function=embedding,
            similarity_fn=similarity_fn,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,
        collection_metadata: Optional[Dict] = None,
        # Document processing parameters
        chunk_size: Optional[int] = MAX_SEQ_LENGTH,
        chunk_overlap: Optional[int] = CHUNK_OVERLAP,
        do_chunking: bool = True,
        language: str = "",
        max_embedding_buffer: int = 500000,
        # Embedding override parameters (override embedding's defaults if provided)
        num_workers: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        # Embedding creation parameters (only used if embedding is None)
        model_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        similarity_fn: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create a Golden_Retriever vectorstore from raw text documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts: List of texts to add to the collection.
            embedding: Pre-configured embedding function. If None, a new Embedding
                is created from the embedding creation parameters below.
            collection_name: Name of the collection to create.
            persist_directory: Directory to persist the collection.
            metadatas: Optional list of metadatas.
            ids: Optional list of document IDs.

            Document Processing Parameters:
                chunk_size: Size of text chunks.
                chunk_overlap: Overlap between chunks.
                do_chunking: Whether to chunk documents.
                language: Language for chunking.
                max_embedding_buffer: Max texts to embed before flushing to the index.

            Embedding Override Parameters (override embedding's defaults):
                num_workers: Number of workers for DataLoader.
                device: Device to use for embedding.
                batch_size: Batch size override for embedding.

            Embedding Creation Parameters (only used if embedding=None):
                model_id, tokenizer_id, max_seq_length, max_batch_size, similarity_fn.

        Returns:
            Golden_Retriever: Populated vectorstore instance.
        """
        logger.info(f"Creating Golden_Retriever from {len(texts)} texts")
        if chunk_size != MAX_SEQ_LENGTH or chunk_overlap != CHUNK_OVERLAP:
            logger.debug(
                f"Custom chunking: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
            )

        # Build embedding creation kwargs (only relevant when embedding is None)
        embedding_kwargs: Dict[str, Any] = {}
        if model_id is not None:
            embedding_kwargs['model_id'] = model_id
        if tokenizer_id is not None:
            embedding_kwargs['tokenizer_id'] = tokenizer_id
        if max_seq_length is not None:
            embedding_kwargs['max_seq_length'] = max_seq_length
        if max_batch_size is not None:
            embedding_kwargs['max_batch_size'] = max_batch_size
        if batch_size is not None and embedding is None:
            embedding_kwargs['batch_size'] = batch_size
        if num_workers is not None and embedding is None:
            embedding_kwargs['num_workers'] = num_workers
        if device is not None and embedding is None:
            embedding_kwargs['device'] = device

        embedding_kwargs.update(kwargs)

        golden_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            similarity_fn=similarity_fn,
            **(embedding_kwargs if embedding is None else {}),
        )
        logger.info(f"Created collection '{collection_name}'")

        if len(texts) <= max_embedding_buffer:
            if do_chunking:
                logger.info(f"Chunking {len(texts)} documents")
                texts = split_documents_given_language(texts,
                                                       language=language,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
                ids = [str(uuid.uuid1()) for _ in texts]
                logger.info(f"Max chunk length: {max(len(t) for t in texts)}")
                logger.info(f"Produced {len(texts)} chunks")
            else:
                logger.info("No chunking specified")
            if not ids:
                ids = [str(uuid.uuid1()) for _ in texts]
            golden_collection.add_texts(
                texts=texts,
                metadatas=None,
                ids=ids,
                num_workers=num_workers,
                device=device,
                batch_size=batch_size,
            )
        else:
            partition = 0
            num_texts = 0
            num_split_texts = 0
            for _texts in split_list(texts, max_embedding_buffer):
                _metadatas = None  # TODO: align metadata partitioning with chunking
                num_texts += len(_texts)

                if do_chunking:
                    logger.info(f"Chunking {len(_texts)} documents")
                    _texts = split_documents_given_language(_texts,
                                                            language=language,
                                                            chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap)
                    ids = [str(uuid.uuid1()) for _ in _texts]
                    logger.info(f"Max chunk length: {max(len(t) for t in _texts)}")
                    logger.info(f"Produced {len(_texts)} chunks")
                else:
                    logger.info("No chunking specified")
                if not ids:
                    ids = [str(uuid.uuid1()) for _ in _texts]

                num_split_texts += len(_texts)
                logger.info(f"Evaluating text partition {partition}")
                try:
                    golden_collection.add_texts(
                        texts=_texts,
                        metadatas=_metadatas,
                        ids=ids,
                        num_workers=num_workers,
                        device=device,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    logger.info(
                        f"Partition {partition}: processed {num_texts} raw documents, "
                        f"{num_split_texts} chunks"
                    )
                    logger.info(f"Encountered exception {e}, saving progress and exiting")
                    golden_collection.persist()
                    logger.debug(traceback.format_exc())

                del _texts, ids
                clear_torch_cache()
                partition += 1
            logger.info(
                f"Golden Retriever completed: processed {partition} partitions, "
                f"{num_texts} raw documents, {num_split_texts} chunks"
            )

        return golden_collection

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,
        collection_metadata: Optional[Dict] = None,
        similarity_fn: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create from documents — delegates to from_texts with all kwargs."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            similarity_fn=similarity_fn,
            **kwargs,
        )
