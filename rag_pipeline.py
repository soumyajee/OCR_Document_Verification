import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class ResumeRAG:
    def __init__(self, model="gpt-4o-mini"):
        self.index = None
        self.text_chunks = []
        self.model = model
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )

    def build_index(self, texts):
        self.text_chunks = []
        embeddings = []
        for text in texts:
            chunks = text.split("\n\n")
            for chunk in chunks:
                self.text_chunks.append(chunk)
                embeddings.append(embed_model.encode(chunk))
        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, "vector.index")
        with open("chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)

    def query(self, question, top_k=3):
        q_emb = embed_model.encode(question).astype("float32")
        D, I = self.index.search(np.array([q_emb]), top_k)
        context = "\n\n".join([self.text_chunks[i] for i in I[0]])

        prompt = f"""
You are a resume assistant. Use the resume information below to answer the question.

Resume:
{context}

Question: {question}
Answer:
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
