"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""

from langchain_community.vectorstores.faiss import FAISS
import uuid

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
import traceback

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.base import Docstore
from langchain_core.documents import Document

from golden.utils import clear_torch_cache, split_list
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

import logging
import sys

logger = logging.getLogger(__name__)
format = '%(asctime)s %(message)s'
logging.basicConfig(filename="golden_faiss_retriever.log",
                    format=format,
                    filemode='w',
                    level=logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)


class Golden_Faiss_Retriever(FAISS):
    """`Meta Faiss` vector store.

    To use, you must have the ``faiss`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS

            embeddings = OpenAIEmbeddings()
            texts = ["FAISS is an important library", "LangChain supports FAISS"]
            faiss = FAISS.from_texts(texts, embeddings)

    """

    def __init__(
        self,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        if not embedding_function:
            embedding_function = Embedding(
                model_id=MODEL_ID,
                tokenizer_id=TOKENIZER_ID,
                max_seq_length=MAX_SEQ_LENGTH,
            )
        super().__init__(
            embedding_function,
            index,
            docstore,
            index_to_docstore_id,
            relevance_score_fn,
            normalize_L2,
            distance_strategy,
        )
        self.persist_directory = ""
        self.index_name: str = "index"

    def _embed_documents(self,
                         texts: List[str],
                         language: str = "",
                         chunk_size: Optional[int] = MAX_SEQ_LENGTH,
                         chunk_overlap: Optional[int] = CHUNK_OVERLAP,
                         num_workers: int = NUM_WORKERS,
                         device: str = DEFAULT_DEVICE,
                         ) -> List[List[float]]:
        if isinstance(self.embedding_function, Embeddings):
            do_chunking = chunk_size > 0
            return self.embedding_function.embed_documents(
                texts=texts,
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                num_workers=num_workers,
                device=device,
                do_chunking=do_chunking,
            )
        else:
            return [self.embedding_function(text) for text in texts]

    def persist(self):
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Delegates to:
            save_local(folder_path: str, index_name: str = "index") -> None
        """
        self.save_local(folder_path=self.persist_directory,
                        index_name=self.index_name)

    @classmethod
    def load(cls,
             folder_path: str,
             embeddings: Optional[Embeddings] = None,
             index_name: str = "index",
             **kwargs: Any,
             ):
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries.
            index_name: index file name to load.
        """
        if not embeddings:
            embeddings = Embedding(**golden_embedding_options(kwargs))
        return cls.load_local(
            folder_path,
            embeddings,
            index_name,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        language: str = "",
        chunk_size: Optional[int] = MAX_SEQ_LENGTH,
        chunk_overlap: Optional[int] = CHUNK_OVERLAP,
        num_workers: int = NUM_WORKERS,
        max_embedding_buffer: int = 500000,
        **kwargs: Any,
    ):  # -> Golden_Faiss_Retriever
        """Construct a FAISS wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates an in-memory docstore.
            3. Initialises the FAISS database.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        if not embedding:
            embedding = Embedding(**golden_embedding_options(kwargs))

        batch_size: Union[str, int] = kwargs.pop("batch_size", "auto")
        max_batch_size: int = kwargs.pop("max_batch_size", 512)
        max_seq_length: int = kwargs.pop("max_seq_length", MAX_SEQ_LENGTH)
        similarity_fn: str = kwargs.pop("similarity_fn", "cosine")
        persist_directory: str = kwargs.pop("persist_directory", "")
        do_chunking: bool = kwargs.pop("do_chunking", True)
        num_workers = kwargs.pop("num_workers", NUM_WORKERS)
        language = kwargs.pop("language", "")
        device: str = kwargs.pop("device", DEFAULT_DEVICE)

        if len(texts) <= max_embedding_buffer:
            if do_chunking:
                logger.info(f"Chunking {len(texts)} documents")
                texts = split_documents_given_language(texts,
                                                       language=language,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
                ids = [str(uuid.uuid1()) for _ in texts]
                logger.info(f"Produced {len(texts)} chunks")
            else:
                logger.info("No chunking specified")
            if not ids:
                ids = [str(uuid.uuid1()) for _ in texts]

            if isinstance(embedding, Embedding):
                embeddings = embedding.embed_documents(
                    texts=texts,
                    num_workers=num_workers,
                    device=device,
                )
            else:
                embeddings = embedding.embed_documents(texts)

            # Classmethod namespace gets mangled due to dunder name use in the FAISS class
            retriever = cls._FAISS__from(
                texts,
                embeddings,
                embedding,
                metadatas=metadatas,
                ids=ids,
                **kwargs,
            )
            setattr(retriever, "persist_directory", persist_directory)

        else:
            partition = 0
            num_texts = 0
            num_split_texts = 0
            for _texts in split_list(texts, max_embedding_buffer):
                # TODO: move metadata to match split and chunking
                _metadatas = None
                num_texts += len(_texts)

                if do_chunking:
                    logger.info(f"Chunking {len(_texts)} documents")
                    _texts = split_documents_given_language(_texts,
                                                            language=language,
                                                            chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap)
                    ids = [str(uuid.uuid1()) for _ in _texts]
                    logger.info(f"Produced {len(_texts)} chunks")
                else:
                    logger.info("No chunking specified")
                if not ids:
                    ids = [str(uuid.uuid1()) for _ in _texts]

                num_split_texts += len(_texts)
                logger.info(f"Evaluating text partition {partition}")

                if isinstance(embedding, Embedding):
                    embeddings = embedding.embed_documents(
                        texts=_texts,
                        num_workers=num_workers,
                        device=device,
                    )
                else:
                    embeddings = embedding.embed_documents(_texts)

                # Classmethod namespace gets mangled due to dunder name use in the FAISS class
                if partition == 0:
                    retriever = cls._FAISS__from(
                        _texts,
                        embeddings,
                        embedding,
                        metadatas=_metadatas,
                        ids=ids,
                        **kwargs,
                    )
                    setattr(retriever, "persist_directory", persist_directory)
                else:
                    try:
                        retriever._FAISS__add(
                            _texts,
                            embeddings,
                            _metadatas,
                            ids,
                        )
                    except Exception as e:
                        logger.info(
                            f"Partition {partition}: processed {num_texts} raw documents, "
                            f"{num_split_texts} chunks"
                        )
                        logger.info(f"Encountered exception {e}, saving progress and exiting")
                        retriever.persist()
                        logger.debug(traceback.format_exc())

                del _texts, ids, embeddings
                clear_torch_cache()
                partition += 1
            logger.info(
                f"FAISS DB completed: processed {partition} partitions, "
                f"{num_texts} raw documents, {num_split_texts} chunks"
            )
        return retriever

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        chunk_size: Optional[int] = MAX_SEQ_LENGTH,
        chunk_overlap: Optional[int] = CHUNK_OVERLAP,
        **kwargs: Any,
    ):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if not embedding:
            embedding = Embedding(**golden_embedding_options(kwargs))
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
