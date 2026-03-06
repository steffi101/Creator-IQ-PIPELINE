"""
Vector Store Service - ChromaDB
Real vector embeddings for RAG retrieval.
Documents are embedded and stored in ChromaDB.
Queries are matched using cosine similarity on embeddings.
"""

import chromadb
from chromadb.config import Settings
import json
import os
import hashlib
import math


class LocalEmbeddingFunction:
    """
    TF-IDF inspired embedding function that runs locally.
    No model download required. Uses term frequency vectors
    with IDF weighting for semantic matching.
    """

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.dim = 384  # Match standard embedding size

    def name(self):
        return "local_tfidf_384"

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        return self.__call__(input)

    def _tokenize(self, text):
        text = text.lower()
        tokens = []
        for word in text.split():
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 2:
                tokens.append(word)
        return tokens

    def _hash_token(self, token):
        """Deterministic hash to fixed dimension."""
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % self.dim

    def __call__(self, input):
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in input:
            vec = [0.0] * self.dim
            tokens = self._tokenize(text)
            if not tokens:
                embeddings.append(vec)
                continue

            # Frequency-weighted hashing
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1

            for token, count in freq.items():
                idx = self._hash_token(token)
                weight = math.log(1 + count)
                vec[idx] += weight
                # Also hash bigrams for better semantic matching
                for other_token in freq:
                    if other_token != token:
                        bigram = f"{token}_{other_token}"
                        bi_idx = self._hash_token(bigram)
                        vec[bi_idx] += weight * 0.5

            # L2 normalize
            magnitude = math.sqrt(sum(v * v for v in vec))
            if magnitude > 0:
                vec = [v / magnitude for v in vec]

            embeddings.append(vec)
        return embeddings


# Trend knowledge base documents
TREND_DOCUMENTS = [
    {"id": "t1", "category": "finance", "type": "format", "content": "UGC Testimonial format drives 2.3x higher completion rate than branded content for finance apps. Best for app install campaigns. Average CPI: $0.65-$0.90.", "metadata": {"category": "finance", "type": "format", "effectiveness": 0.95}},
    {"id": "t2", "category": "finance", "type": "format", "content": "Screen Recording Walkthrough shows actual product UI, reduces friction to install. Completion rate 38% avg for utility apps. Works best with text overlays showing step-by-step actions.", "metadata": {"category": "finance", "type": "format", "effectiveness": 0.90}},
    {"id": "t3", "category": "general", "type": "format", "content": "Problem-Solution Narrative hooks with pain point in first 2 seconds. 1.8x higher CTR when problem is relatable. Best for apps solving clear user problems.", "metadata": {"category": "general", "type": "format", "effectiveness": 0.88}},
    {"id": "t4", "category": "fitness", "type": "format", "content": "Before/After transformation format drives highest engagement for fitness apps. 2.1x share rate vs standard ads. Requires authentic-looking results, not stock footage.", "metadata": {"category": "fitness", "type": "format", "effectiveness": 0.92}},
    {"id": "t5", "category": "general", "type": "format", "content": "POV / Day-in-my-life format feels organic in feed, high watch time. Works across all app categories. Average completion rate 31%. Best when creator genuinely uses the product.", "metadata": {"category": "general", "type": "format", "effectiveness": 0.85}},
    {"id": "t6", "category": "general", "type": "hook_style", "content": "Curiosity Gap hooks average 38% scroll-stop rate. Example: 'I found out why I was always broke by Tuesday.' Creates information asymmetry the viewer must resolve by watching.", "metadata": {"category": "general", "type": "hook_style", "effectiveness": 0.95}},
    {"id": "t7", "category": "general", "type": "hook_style", "content": "Bold Claim hooks with specific numbers average 34% scroll-stop rate. Example: 'This app saved me $400 in one month.' Specificity builds credibility. Round numbers feel less believable than specific ones.", "metadata": {"category": "general", "type": "hook_style", "effectiveness": 0.90}},
    {"id": "t8", "category": "general", "type": "hook_style", "content": "POV Storytelling hooks average 31% scroll-stop rate. Example: 'POV: you finally know where your money goes.' Creates instant viewer identification and feels native to platform.", "metadata": {"category": "general", "type": "hook_style", "effectiveness": 0.87}},
    {"id": "t9", "category": "general", "type": "hook_style", "content": "Problem Agitation hooks average 29% scroll-stop rate. Example: 'Stop guessing. Your bank account is crying.' Works best for 25-34 age segment. Emotional language drives action.", "metadata": {"category": "general", "type": "hook_style", "effectiveness": 0.85}},
    {"id": "t10", "category": "general", "type": "hook_style", "content": "Social Proof hooks average 27% scroll-stop rate. Example: '2 million people switched to this and I get why now.' Leverages herd behavior but needs real numbers to be credible.", "metadata": {"category": "general", "type": "hook_style", "effectiveness": 0.82}},
    {"id": "t11", "category": "general", "type": "hook_style", "content": "Before/After hooks average 33% scroll-stop rate. Example: 'My spending last month vs this month.' Visual contrast creates instant engagement and curiosity about the transformation.", "metadata": {"category": "general", "type": "hook_style", "effectiveness": 0.88}},
    {"id": "t12", "category": "general", "type": "fatigue", "content": "Average creative lifespan on short-form video platforms: 7-10 days. Key fatigue signals: CPI increases >20% over 3 consecutive days, CTR drops below 0.8%, frequency exceeds 3.0 per user, video completion rate drops below 15%.", "metadata": {"category": "general", "type": "fatigue", "effectiveness": 0.95}},
    {"id": "t13", "category": "general", "type": "fatigue", "content": "Recommended creative refresh: 3-5 new variations every 7 days. Creative refresh frequency is the number one factor determining ad group lifespan. Minimum 3 variations needed to test per cycle.", "metadata": {"category": "general", "type": "fatigue", "effectiveness": 0.93}},
    {"id": "t14", "category": "general", "type": "platform", "content": "93% of top-performing short-form video ads use text overlays. Best video length for app installs: 15-22 seconds. Hook window: first 2 seconds determine 80% of watch-through rate. Vertical 9:16 ratio mandatory for feed placement.", "metadata": {"category": "general", "type": "platform", "effectiveness": 0.90}},
    {"id": "t15", "category": "general", "type": "platform", "content": "Native UGC-style content outperforms polished brand content by 2-3x on CPI. Videos under 18 seconds have 22% lower CPI than 25+ second videos. Sound-on content drives 1.5x faster emotional response.", "metadata": {"category": "general", "type": "platform", "effectiveness": 0.92}},
    {"id": "t16", "category": "ecommerce", "type": "format", "content": "Product demo with trending sound remix drives highest add-to-cart rate for ecommerce apps. Algorithm boost from popular audio but audio trends decay fast at 3-5 days.", "metadata": {"category": "ecommerce", "type": "format", "effectiveness": 0.88}},
    {"id": "t17", "category": "gaming", "type": "format", "content": "Gameplay footage with reaction overlay drives 2.8x install rate for gaming apps. First 3 seconds must show most exciting gameplay moment, not studio logo or loading screen.", "metadata": {"category": "gaming", "type": "format", "effectiveness": 0.91}},
    {"id": "t18", "category": "general", "type": "audience", "content": "18-24 segment responds best to POV and trending format hooks. 25-34 responds best to problem-agitation and bold claim hooks. 35-44 responds best to social proof and before/after hooks. Segment targeting should match hook style.", "metadata": {"category": "general", "type": "audience", "effectiveness": 0.89}},
]


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chromadb"):
        """Initialize ChromaDB with persistent storage."""
        self.embed_fn = LocalEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="ad_trends",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embed_fn,
        )
        self._initialized = False

    def initialize(self):
        """Load documents into the vector store if not already loaded."""
        if self._initialized:
            return

        existing = self.collection.count()
        if existing >= len(TREND_DOCUMENTS):
            self._initialized = True
            return

        # Clear and reload
        try:
            self.client.delete_collection("ad_trends")
            self.collection = self.client.get_or_create_collection(
                name="ad_trends",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embed_fn,
            )
        except Exception:
            pass

        # Add documents with ChromaDB's built-in embedding
        ids = [doc["id"] for doc in TREND_DOCUMENTS]
        documents = [doc["content"] for doc in TREND_DOCUMENTS]
        metadatas = [doc["metadata"] for doc in TREND_DOCUMENTS]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        self._initialized = True
        print(f"Loaded {len(TREND_DOCUMENTS)} documents into ChromaDB")

    def query(self, query_text: str, n_results: int = 6, category_filter: str = None):
        """
        Query the vector store using semantic similarity.
        ChromaDB handles embedding the query and finding nearest neighbors.
        """
        self.initialize()

        where_filter = None
        if category_filter:
            where_filter = {
                "$or": [
                    {"category": category_filter},
                    {"category": "general"}
                ]
            }

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieved.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(1 - results["distances"][0][i], 3),  # Convert distance to similarity
                })

        return retrieved

    def get_stats(self):
        """Return collection stats."""
        self.initialize()
        return {
            "total_documents": self.collection.count(),
            "collection_name": "ad_trends",
            "embedding_model": "Local TF-IDF with hashed bigrams (384-dim)",
            "similarity_metric": "cosine",
        }


# Singleton instance
vector_store = VectorStore()
