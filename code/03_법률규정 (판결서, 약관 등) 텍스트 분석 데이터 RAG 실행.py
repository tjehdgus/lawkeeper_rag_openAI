import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

# ✅ OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ ChromaDB 설정
chroma_path = "C:/Users/sdh/Desktop/chroma_law_db"
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection = chroma_client.get_collection("legal_cases")

# ✅ 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# ✅ 사용자 질문
query = "누가 잘못했나요?"
query_embedding = embedding_model.encode([query])[0].tolist()

# ✅ 유사 문서 검색
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

contexts = results["documents"][0]
context_text = "\n\n".join(contexts)

# ✅ 프롬프트 구성
prompt = f"""다음은 법률 판결문입니다. 주어진 문서들을 참고하여 사용자의 질문에 답해주세요.

문서들:
{context_text}

질문:
{query}

답변:"""

# ✅ GPT-4o 호출
try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 법률 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    print("\n🧠 GPT-4o 응답:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"❌ OpenAI 오류 발생: {e}")