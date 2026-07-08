"""
Author: John T. Halloran <johnhalloran321@gmail.com>
"""

from golden.golden_embeddings import Embedding
x = Embedding(max_batch_size=512, max_seq_length= 1200)
batch_size = x._detect_batch_size()
print(f"Determined Largest batch size: {batch_size}")
