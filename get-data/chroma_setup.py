import chromadb
from chromadb.utils import embedding_functions

print("Initializing Chroma client...")
chroma_client = chromadb.PersistentClient(path="./chroma_memory")

print("Loading embedding model (may take a moment on first run)...")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("Creating/getting collection...")
review_memory_collection = chroma_client.get_or_create_collection(
    name="review_memory",
    embedding_function=embedding_fn
)

print("✅ Chroma collection ready:", review_memory_collection.name)
print("✅ Current item count:", review_memory_collection.count())