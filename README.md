# Golden
Custom LangChain retrieval APIs for rapid ML experimentation

## Install
Install the package locally after cloning:

    make install
    # OR
    pip install .
## Development
Install the package locally in editable mode after cloning:

    make develop
    # OR
    pip install -e .[all]

## Use
Use wherever and however you would usually use ChromaDB.  I.e., 

    from golden.golden_retriever import Golden_Retriever
    db = Golden_Retriever.from_documents(texts)

## Golden Retriever Custom Parameters
The following are additional input variables and their options:

    language = "python" # language for input text ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html', 'sol', 'csharp', 'cobol']
    similarity_fn = "l2" # l2 = euclidean, cosine = cosine, ip = inner product (i.e., unnormalized cosine)

## Golden Embedding Custom Parameters
### Constructor

    model_id = "togethercomputer/m2-bert-80M-8k-retrieval" # transformers model id
    tokenizer_id = "bert-base-uncased" # transformers token id
    max_seq_length = 8192 # max context window
    max_batch_size = 64 # Max batch to consider
    batch_size = "auto" # Starting from max_batch_size, automatically detect batch size which fits in memory
    quantization_config = None # either none or an instance of a BitsAndBytesConfig
    
### embed_documents

    language = "python" # language for input text ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html', 'sol', 'csharp', 'cobol']
    chunk_size: 8192
    chunk_overlap: 20
    batch_size = 32 # Set to "auto" to automatically detect the batch size which fits in memory
    num_workers = 3 # Num CPU workers during dataloading
    device = "cuda" # "cuda" = GPU accelerator available, "cpu" otherwise
