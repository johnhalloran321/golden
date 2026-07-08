"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""
from typing import List, Final, Dict, Optional, Union, Any
import transformers
import torch
import numpy as np
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from golden.utils import (find_executable_batch_size, 
                          clear_torch_cache, 
                          should_reduce_batch_size, 
                          should_reduce_batch_size_but_handle_error)
import json
import os
import logging
import sys
import torch.nn.functional as F

eval_logger = logging.getLogger("golden-embeddings")
eval_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
eval_logger.addHandler(handler)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = {"cuda" : ("cuda", "CUDA GPU detected") if torch.cuda.is_available() else ('cpu', "CUDA not available, defaulting to CPU"),
          "mps"  : ("mps", "Apple Silicon GPU detected") if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else ('cpu', "MPS not available, defaulting to CPU"),
          "cpu"  : ("cpu", "CPU selected"),
          }

# Best available device: MPS (Apple Silicon) > CUDA > CPU
def _detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEFAULT_DEVICE: str = _detect_default_device()


# Per-embedding normalization specific functions, standarizes API calls
def m2_bert_norm(model_output, attention_mask = None): # "togethercomputer/m2-bert-80M-8k-retrieval"
    return model_output["sentence_embedding"]

# sentence-transformers/all-MiniLM-L6-v2
# See: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def mean_pooling_and_norm(model_output, attention_mask): 
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return F.normalize(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9), p=2, dim=1)

# thenlper/gte-large
# See: https://huggingface.co/thenlper/gte-large
def average_pool_and_norm(outputs: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = outputs.last_hidden_state
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return F.normalize(last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None], p=2, dim=1)

# Alibaba-NLP/gte-Qwen2-1.5B-instruct
# See: https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
def last_token_pool_and_norm(outputs: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = outputs.last_hidden_state
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return F.normalize(last_hidden_states[:, -1],  p=2, dim=1)
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return F.normalize(last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths], p=2, dim=1)

MODEL_ZOO = {"togethercomputer/m2-bert-80M-8k-retrieval" : {"tokenizer_id" : "bert-base-uncased", 
                                                            "alias" : "m2-bert",
                                                            "norm" : m2_bert_norm,
                                                            },
             "sentence-transformers/all-MiniLM-L6-v2" : {"tokenizer_id" : "sentence-transformers/all-MiniLM-L6-v2",
                                                         "alias" : "all-MiniLM-L6-v2",
                                                         "norm" : mean_pooling_and_norm,
                                                         },
             "thenlper/gte-large" : {"tokenizer_id" : "thenlper/gte-large",
                                     "alias" : "gte-large",
                                     "norm" : average_pool_and_norm,
                                     },
             "Alibaba-NLP/gte-Qwen2-1.5B-instruct" : {"tokenizer_id" : "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                                     "alias" : "gte-Qwen2-1.5B-instruct",
                                     "norm" : last_token_pool_and_norm,
                                     },                                     
                                     }

MODEL_ID: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_ID: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH: Final[int] = 512
BATCH_SIZE: Final[int] = 32
NUM_WORKERS: Final[int] = 8
CHUNK_OVERLAP: Final[int] = 20
DICT_LANGUAGE_MAP: Final[Dict[str, Language]] = {l.name.lower(): l for l in Language}


# TODO: add support for multi-gpus via world-size, leveraging accelerate

def golden_embedding_options(kwargs: Dict[Any,Any] = {}):
    options = {"model_id" : MODEL_ID,
               "tokenizer_id" : TOKENIZER_ID,
               "max_seq_length" : MAX_SEQ_LENGTH,
               "device" : DEFAULT_DEVICE,  # Use smart device detection instead of hardcoded cuda
               "trust_remote_code" : True,
               "max_batch_size" : 256,
               "batch_size" : 1,
               "quantization_config" : None,
               "num_workers" : NUM_WORKERS,
               "use_flash_attention_2" : None,  # Add this: None = auto-detect
               "multi_gpu" : None,
    }
    if kwargs:
        for k in kwargs:
            if k in options:
                if k in ['batch_size', 'max_batch_size', 'max_seq_length', 'num_workers']:
                    if kwargs[k] != "auto":
                        try:
                            options[k] = int(kwargs[k])
                        except ValueError:
                            raise ValueError(f"{k} must be an integer or 'auto', got: {kwargs[k]}")
                    else:
                        options[k] = kwargs[k]
                elif k in ['use_flash_attention_2', 'multi_gpu']:
                    # Convert to boolean if it's a string
                    if isinstance(kwargs[k], str):
                        options[k] = kwargs[k].lower() in ('true', '1', 'yes')
                    else:
                        options[k] = bool(kwargs[k]) if kwargs[k] is not None else None
                else:
                    options[k] = kwargs[k]           

    # Replace print statements with logging
    eval_logger.debug(f"Input kwargs: {kwargs}")
    eval_logger.debug(f"Resolved embedding options: {options}")    
    return options

def check_flash_attention_2():
    """Check if Flash Attention 2 is installed and available.
    
    Returns:
        bool: True if flash attention 2 is available, False otherwise
    """
    try:
        import flash_attn
        # Check if the flash_attn version supports flash attention 2
        flash_attn_version = getattr(flash_attn, '__version__', '0.0.0')
        major_version = int(flash_attn_version.split('.')[0])
        if major_version >= 2:
            eval_logger.info(f"Flash Attention 2 is installed (version {flash_attn_version})")
            return True
        else:
            eval_logger.info(f"Flash Attention is installed but version {flash_attn_version} < 2.0")
            return False
    except ImportError:
        eval_logger.info("Flash Attention 2 is not installed")
        return False

def model_supports_flash_attention_2(model_id: str) -> bool:
    """Check if a model supports Flash Attention 2.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        bool: True if model supports flash attention 2, False otherwise
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Check if model config explicitly supports flash attention 2
        supports_fa2 = getattr(config, '_flash_attn_2_enabled', False)
        
        # Additional check: some models support it through model_type
        model_type = getattr(config, 'model_type', None)
        # List of known model types that support flash attention 2
        known_fa2_models = {
            'llama', 'mistral', 'mixtral', 'qwen2', 'phi', 'gemma', 
            'bert', 'roberta', 'gpt2', 'gpt_neox', 'opt', 'bloom'
        }
        
        if supports_fa2 or model_type in known_fa2_models:
            eval_logger.info(f"Model {model_id} (type: {model_type}) supports Flash Attention 2")
            return True
        else:
            eval_logger.info(f"Model {model_id} (type: {model_type}) does not support Flash Attention 2")
            return False
            
    except Exception as e:
        eval_logger.warning(f"Could not determine Flash Attention 2 support for {model_id}: {e}")
        return False

def split_documents_given_language(texts: List[str],
                                   language: str = "",
                                   chunk_size = MAX_SEQ_LENGTH,
                                   chunk_overlap = CHUNK_OVERLAP)-> List[str]:
    if not language or language not in DICT_LANGUAGE_MAP:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language = DICT_LANGUAGE_MAP[language],
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap)        

    return [getattr(doc, "page_content", "") for doc in
            text_splitter.split_documents([Document(text) for text in texts])]

def get_gpu_memory_info(self):
    """Get GPU memory information for all GPUs.
    
    Returns:
        dict: GPU memory information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "num_gpus": self.num_gpus,
        "using_multi_gpu": self.use_multi_gpu,
        "strategy": "DataParallel" if self.use_multi_gpu else "Single GPU",
        "gpus": []
    }
    
    for i in range(self.num_gpus):
        gpu_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved(i) / 1e9,
        }
        info["gpus"].append(gpu_info)
    
    return info

def _estimate_model_size_gb(self):
    """Estimate model size in GB.
    
    Returns:
        float: Estimated model size in GB
    """
    if not hasattr(self, 'model'):
        return 0.0
    
    # Get model from DataParallel wrapper if needed
    model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_gb = (param_size + buffer_size) / 1e9
    return size_gb

def print_multi_gpu_info(self):
    """Print detailed multi-GPU configuration information."""
    print("\n" + "="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    if not self.use_multi_gpu:
        print(f"Mode: Single GPU")
        print(f"Device: {self.device}")
    else:
        model_size = self._estimate_model_size_gb()
        print(f"Mode: DataParallel (Multi-GPU)")
        print(f"Number of GPUs: {self.num_gpus}")
        print(f"Primary Device: {self.device}")
        print(f"\nHow DataParallel works:")
        print(f"  • Model is replicated on each GPU (~{model_size:.2f} GB per GPU)")
        print(f"  • Input batch is split across GPUs")
        print(f"  • Each GPU processes its portion in parallel")
        print(f"  • Results are gathered back to GPU 0")
        print(f"\nBatch splitting:")
        if isinstance(self.batch_size, int):
            per_gpu = self.batch_size // self.num_gpus
            print(f"  • Total batch size: {self.batch_size}")
            print(f"  • Per GPU: ~{per_gpu} samples")
        else:
            print(f"  • Batch size: {self.batch_size} (will be split evenly)")
        print(f"\nMemory requirements:")
        print(f"  • Model size: ~{model_size:.2f} GB")
        print(f"  • Total model memory: ~{model_size * self.num_gpus:.2f} GB ({self.num_gpus} copies)")
        print(f"  • Each GPU must have at least {model_size:.2f} GB available")
    
    # GPU memory info
    gpu_info = self.get_gpu_memory_info()
    if gpu_info['available']:
        print(f"\nGPU Memory Status:")
        for gpu in gpu_info['gpus']:
            print(f"  GPU {gpu['id']} ({gpu['name']}):")
            print(f"    Total: {gpu['total_memory_gb']:.2f} GB")
            print(f"    Allocated: {gpu['allocated_memory_gb']:.2f} GB")
            print(f"    Cached: {gpu['cached_memory_gb']:.2f} GB")
            free = gpu['total_memory_gb'] - gpu['allocated_memory_gb']
            print(f"    Free: {free:.2f} GB")
            if self.use_multi_gpu:
                model_size = self._estimate_model_size_gb()
                if free < model_size:
                    print(f"    ⚠ WARNING: Free memory ({free:.2f} GB) < Model size ({model_size:.2f} GB)")
    
    print("="*70 + "\n")

class Embedding(Embeddings):
    """ Given embedding model, associated tokenizer, and a corpus:
            - Embed each member of the corpus
            - Create FAISS index over all embeddings
            
        Given a set of queries and a specified top_k:
            - Return top_k closest db indices per query
    """
    def __init__(self,
                 model_id: str = MODEL_ID,
                 tokenizer_id: str = TOKENIZER_ID,
                 max_seq_length: int = MAX_SEQ_LENGTH,
                 device: str = DEFAULT_DEVICE,  # Use smart device detection instead of hardcoded 'cuda'
                 trust_remote_code: Optional[bool] = True,
                 max_batch_size: Optional[int] = 256,
                 batch_size: Optional[Union[int, str]] = "auto",
                 quantization_config = None,
                 num_workers: Optional[int] = NUM_WORKERS, 
                 use_flash_attention_2: Optional[bool] = None,
                 multi_gpu: Optional[bool] = None,
                 batch_size_cache_path: Optional[str] = None,
                 ):
        self._world_size = 1
        self.trust_remote_code = trust_remote_code
        self._config = None
        self.AUTO_MODEL_CLASS = None

        # # # TODO: add metal support

        # Validate and set max_batch_size
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be a positive integer, got: {max_batch_size}")
        self.max_batch_size = max_batch_size

        # Validate and set batch_size
        if isinstance(batch_size, str):
            if batch_size.lower() != "auto":
                raise ValueError(f"batch_size must be an integer or 'auto', got: {batch_size}")
            self.batch_size = batch_size  # Will be set later by set_batch_size()
        elif isinstance(batch_size, int):
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got: {batch_size}")
            if batch_size > max_batch_size:
                eval_logger.warning(f"batch_size ({batch_size}) > max_batch_size ({max_batch_size}), setting batch_size = max_batch_size")
                batch_size = max_batch_size
            self.batch_size = batch_size
        else:
            raise TypeError(f"batch_size must be int or str 'auto', got type: {type(batch_size)}")

        # Path for persisting the detected batch size across restarts
        self.batch_size_cache_path = batch_size_cache_path

        # Validate and set num_workers
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError(f"num_workers must be a non-negative integer, got: {num_workers}")
        self.num_workers = num_workers
        
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id

        # Handle device and multi-GPU configuration
        if device not in DEVICE:
            raise Exception(f"{device} specified, but must be one of 'cpu', 'cuda', or 'mps'")
        device, msg = DEVICE[device]
        eval_logger.info(msg)

        # MPS (Apple Silicon) does not support pin_memory and uses 'spawn'
        # multiprocessing, which causes a recursive re-import crash when
        # num_workers > 0.  Force single-process data loading on MPS.
        if device == "mps" and self.num_workers > 0:
            eval_logger.info(
                f"MPS device detected: overriding num_workers={self.num_workers} → 0 "
                "(spawn-based multiprocessing is incompatible with MPS)"
            )
            self.num_workers = 0

        # Determine multi-GPU usage (DataParallel only)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = False
        
        if multi_gpu is None:
            # Auto-detect: use DataParallel if multiple GPUs available and device is cuda
            if self.num_gpus > 1 and device == 'cuda':
                self.use_multi_gpu = True
                eval_logger.info(f"Auto-detected {self.num_gpus} GPUs, enabling DataParallel")
        elif multi_gpu:
            if self.num_gpus < 2:
                eval_logger.warning(f"Multi-GPU requested but only {self.num_gpus} GPU(s) available, using single GPU")
                self.use_multi_gpu = False
            elif device != 'cuda':
                eval_logger.warning("Multi-GPU requires CUDA device, disabling")
                self.use_multi_gpu = False
            else:
                self.use_multi_gpu = True
                eval_logger.info(f"DataParallel enabled across {self.num_gpus} GPUs")
        else:
            eval_logger.info("Multi-GPU disabled, using single GPU")
        
        # Set primary device
        self.device = torch.device('cuda:0' if device == 'cuda' else device)
        if self.use_multi_gpu:
            self._world_size = self.num_gpus

        # Determine if we should use Flash Attention 2
        self.use_flash_attention_2 = False
        if use_flash_attention_2 is None:
            # Auto-detect: use if available and on CUDA
            if self.device.type == 'cuda':
                fa2_available = check_flash_attention_2()
                fa2_supported = model_supports_flash_attention_2(model_id)
                self.use_flash_attention_2 = fa2_available and fa2_supported
                if self.use_flash_attention_2:
                    eval_logger.info(f"Flash Attention 2 will be used for {model_id}")
                elif fa2_available and not fa2_supported:
                    eval_logger.info(f"Flash Attention 2 is available but not supported by {model_id}")
                elif not fa2_available and fa2_supported:
                    eval_logger.info(f"Model {model_id} supports Flash Attention 2 but it's not installed. Install with: pip install flash-attn --no-build-isolation")
            else:
                device_name = "MPS (Apple Silicon)" if self.device.type == 'mps' else self.device.type.upper()
                eval_logger.info(f"Flash Attention 2 requires CUDA, skipping on {device_name}")
        elif use_flash_attention_2:
            # User explicitly requested Flash Attention 2
            if self.device.type != 'cuda':
                device_name = "MPS (Apple Silicon)" if self.device.type == 'mps' else self.device.type.upper()
                eval_logger.warning(f"Flash Attention 2 requested but device is {device_name}, disabling (requires CUDA)")
                self.use_flash_attention_2 = False
            else:
                fa2_available = check_flash_attention_2()
                if not fa2_available:
                    raise ValueError(
                        "Flash Attention 2 requested but not installed. "
                        "Install with: pip install flash-attn --no-build-isolation"
                    )
                fa2_supported = model_supports_flash_attention_2(model_id)
                if not fa2_supported:
                    eval_logger.warning(
                        f"Flash Attention 2 requested but model {model_id} may not support it. "
                        "Attempting to use it anyway..."
                    )
                self.use_flash_attention_2 = True
                eval_logger.info(f"Flash Attention 2 explicitly enabled for {model_id}")

        # Check if quantization file was supplied, then try to import BitsAndBytes...
        model_loaded = False
        model_kwargs = {
            'trust_remote_code': self.trust_remote_code,
        }
        # Add Flash Attention 2 if enabled
        if self.use_flash_attention_2:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            eval_logger.info("Loading model with Flash Attention 2")
        # Check if quantization file was supplied, then try to import BitsAndBytes...
        model_loaded = False
        if quantization_config:
            try:
                from transformers import BitsAndBytesConfig
                bandb = True
            except ImportError:
                bandb = False
            if bandb:
                if not type(quantization_config)==BitsAndBytesConfig:
                    eval_logger.info("Quantization config not of type BitsAndBytesConfig, loading full precision model")
                else:
                    if model_id == "togethercomputer/m2-bert-80M-8k-retrieval":
                        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                            model_id, 
                            trust_remote_code=self.trust_remote_code,
                            quantization_config=quantization_config,
                            ).to(self.device)
                    else:
                        self.model = transformers.AutoModel.from_pretrained(
                            model_id, 
                            trust_remote_code=self.trust_remote_code,
                            quantization_config=quantization_config,
                            ).to(self.device) 
                    model_loaded = True
            else:
                eval_logger.info("Quantization config detected but BitsAndBytesConfig failed to import, loading full precision model")
        if not model_loaded:
            if model_id == "togethercomputer/m2-bert-80M-8k-retrieval":
                self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_id, 
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)
            else:
                self.model = transformers.AutoModel.from_pretrained(
                    model_id, 
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)                

        # Move model to device first
        self.model.to(self.device)
        
        # Apply DataParallel if requested
        if self.use_multi_gpu:
            eval_logger.info(f"Wrapping model with DataParallel across GPUs: {list(range(self.num_gpus))}")
            self.model = torch.nn.DataParallel(self.model)
            eval_logger.info(f"Model replicated on {self.num_gpus} GPUs, input batches will be split across GPUs")
        # Set model to eval mode
        self.model.eval()

        self.max_seq_length: int = max_seq_length
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_id,
                model_max_length=max_seq_length,
                )
        except Exception as Argument:
            raise Exception(f"Failed to load specified tokenizer {tokenizer_id} with error {Argument}")            
        # Model is ready, check we're doing auto-batch-size detection
        if isinstance(batch_size, str): # Already checked string value above
            self.set_batch_size()

    @property
    def world_size(self):
        return self._world_size

    def _model_call(self, inps):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        embeddings returned from the model's forward
        """
        with torch.no_grad(): # Call forward
            return self.model(inps)

    def _batch_size_cache_key(self) -> str:
        """Unique key for the current hardware/model configuration."""
        return f"{self.model_id}|{self.device}|{self.max_batch_size}|{self.max_seq_length}"

    def _load_cached_batch_size(self) -> Optional[int]:
        """Return a previously saved batch size for this config, or None."""
        if not self.batch_size_cache_path:
            return None
        try:
            with open(self.batch_size_cache_path, "r") as f:
                cache = json.load(f)
            key = self._batch_size_cache_key()
            value = cache.get(key)
            if isinstance(value, int) and value > 0:
                eval_logger.info(
                    f"Loaded cached batch size {value} for key '{key}' "
                    f"from {self.batch_size_cache_path}"
                )
                return value
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return None

    def _save_cached_batch_size(self, batch_size: int) -> None:
        """Persist the detected batch size so future runs skip detection."""
        if not self.batch_size_cache_path:
            return
        try:
            try:
                with open(self.batch_size_cache_path, "r") as f:
                    cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                cache = {}
            cache[self._batch_size_cache_key()] = batch_size
            os.makedirs(os.path.dirname(os.path.abspath(self.batch_size_cache_path)), exist_ok=True)
            with open(self.batch_size_cache_path, "w") as f:
                json.dump(cache, f, indent=2)
            eval_logger.info(f"Saved batch size {batch_size} to {self.batch_size_cache_path}")
        except OSError as e:
            eval_logger.warning(f"Could not save batch size cache: {e}")

    def set_batch_size(self):
        """Detect and set the optimal batch size, loading from cache when available."""
        # Try cache first — skips the binary-search on subsequent runs
        cached = self._load_cached_batch_size()
        if cached is not None:
            self.batch_size = cached
            return

        eval_logger.info("Detecting largest batch size.")
        try:
            detected_batch_size = self._detect_batch_size()
            if not isinstance(detected_batch_size, int) or detected_batch_size <= 0:
                eval_logger.error(
                    f"Batch size detection returned invalid value: {detected_batch_size}, "
                    f"defaulting to 1"
                )
                self.batch_size = 1
            else:
                self.batch_size = detected_batch_size
                eval_logger.info(f"Determined largest batch size: {self.batch_size}")
                self._save_cached_batch_size(self.batch_size)
        except Exception as e:
            eval_logger.error(f"Batch size detection failed with error: {e}, defaulting to 1")
            eval_logger.debug(f"Full traceback:", exc_info=True)
            self.batch_size = 1
        
    def embed_documents(self, 
                        texts: List[str],
                        num_workers: Optional[int] = None,
                        device: Optional[str] = None,
                        batch_size: Optional[int] = None,  # Allow override
                        return_numpy: bool = False) -> Union[np.ndarray, List[List[float]]]:
        """ Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            num_workers: Override instance num_workers if provided
            device: Override instance device if provided  
            batch_size: Override instance batch_size if provided
            return_numpy: Return numpy array if True, else list of lists
            
        Returns:
            Embeddings as numpy array or list of lists
        """
        # Use instance defaults unless overridden
        _num_workers = num_workers if num_workers is not None else self.num_workers
        _batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Validate overridden batch_size
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(f"batch_size must be a positive integer, got: {batch_size}")
            if batch_size > self.max_batch_size:
                eval_logger.warning(f"batch_size ({batch_size}) > max_batch_size ({self.max_batch_size}), using max_batch_size")
                _batch_size = self.max_batch_size
        
        embeddings = np.array([])      
        for batch in tqdm(torch.utils.data.DataLoader(texts,
                                                    batch_size=_batch_size,
                                                    shuffle=False,
                                                    num_workers=_num_workers,
                                                    pin_memory=(self.device == "cuda"),
                                                    batch_sampler=None,
                                                    sampler=None)):
            if not np.any(embeddings):
                embeddings = self.embed_batch(batch)
            else:
                embeddings = np.concatenate((embeddings, self.embed_batch(batch)), axis=0)
        
        if return_numpy:
            return embeddings
        else:
            return embeddings.tolist()
        
    def embed_batch(self,
                    batch: List[str]) -> np.ndarray:
                    # device: str = '') -> np.ndarray:
        """ Make sure to transfer back to CPU
        """
        device = self.device
        with torch.no_grad():
            input_ids = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_seq_length
            ).to(device)
            outputs = self.model(**input_ids)
            if self.model_id in MODEL_ZOO:
                return MODEL_ZOO[self.model_id]["norm"](outputs, input_ids['attention_mask']).to('cpu').numpy()
            else:
                try:
                    return MODEL_ZOO[MODEL_ID]["norm"](outputs, input_ids['attention_mask']).to('cpu').numpy()
                except Exception as e:
                    eval_logger.error(
                        f"Could not find normalization function for {self.model_id}, "
                        f"tried {MODEL_ID} normalization but failed with error: {e}"
                    )                    
                    raise
            
        
    def embed_query(self, 
                    text: str,
                    device: str = '',
                    ) -> List[float]:
        return self.embed_batch(batch=[text]).squeeze().tolist()

    def _detect_batch_size(self, requests=None, pos: int = 0):
        """Detect the largest batch size that fits in memory using binary search.
        
        Uses binary search to efficiently find the maximum batch size that can
        be processed without running out of memory. Tests each candidate batch
        size with multiple forward passes to ensure stability.
        
        Args:
            requests: Optional list of requests for context length calculation
            pos: Position in requests list to use for length calculation
            
        Returns:
            int: Largest working batch size (minimum 1)
        """
        # Determine sequence length to test
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_seq_length + 1):][:-1]
            )
        else:
            max_length = self.max_seq_length

        def test_batch_size(batch_size):
            """Test if a batch size works by running forward passes.
            
            Args:
                batch_size: Batch size to test
                
            Returns:
                bool: True if batch size works, False if OOM or similar error
                
            Raises:
                RuntimeError: For non-recoverable errors
            """
            if batch_size <= 0:
                return False
                
            try:
                # Create test inputs with proper structure (input_ids + attention_mask)
                test_input_ids = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
                
                # Create attention mask (all ones = attend to all tokens)
                test_attention_mask = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
                
                # Create input dict similar to what tokenizer returns
                test_inputs = {
                    'input_ids': test_input_ids,
                    'attention_mask': test_attention_mask
                }
                
                # Run 2 forward passes to ensure stability
                # (first pass may allocate caches, second tests actual memory usage)
                for _ in range(5):
                    with torch.no_grad():
                        _ = self.model(**test_inputs)
                        
                return True
                
            except RuntimeError as e:
                if should_reduce_batch_size(e):
                    # OOM or similar - this batch size doesn't work
                    return False
                elif should_reduce_batch_size_but_handle_error(e):
                    # Non-recoverable error (e.g., Cublas handle creation failure)
                    raise RuntimeError(
                        f"Batch size {batch_size} caused non-recoverable error. "
                        "This likely indicates an excessively large batch size or GPU issue. "
                        "Try reducing max_batch_size parameter and ensure GPU has sufficient resources."
                    )
                else:
                    # Unexpected error - re-raise for debugging
                    raise
            except AttributeError as e:
                # Handle cases where model has unexpected requirements
                if "'NoneType' object has no attribute" in str(e):
                    eval_logger.error(
                        f"Model requires specific input format that wasn't provided. "
                        f"Error: {e}. Falling back to batch_size=1"
                    )
                    return False
                else:
                    raise

        # Binary search for optimal batch size
        # left: largest known working batch size
        # right: smallest known failing batch size (or upper bound + 1)
        left = 0
        right = self.max_batch_size + 1
        
        eval_logger.info(
            f"Auto-detecting optimal batch size "
            f"(max_batch_size={self.max_batch_size}, seq_length={max_length})"
        )
        
        # Binary search loop
        while left + 1 < right:
            mid = (left + right) // 2
            eval_logger.info(f"Testing batch_size={mid} (range: [{left}, {right}))")
            
            # Clear cache before each test to get accurate memory measurements
            clear_torch_cache()
            
            if test_batch_size(mid):
                # Success - this batch size works, try larger
                left = mid
                eval_logger.info(f"✓ batch_size={mid} works")
            else:
                # Failure - this batch size is too large, try smaller
                right = mid
                eval_logger.info(f"✗ batch_size={mid} failed (OOM or memory error)")
        
        # Clean up after detection
        clear_torch_cache()
        
        # left is now the largest working batch size
        if left == 0:
            eval_logger.warning(
                "Could not find any working batch size in range [1, %d]. "
                "Defaulting to batch_size=1. Consider checking GPU memory availability.",
                self.max_batch_size
            )
            return 1
        
        eval_logger.info(f"Optimal batch size detected: {left}")
        return left
