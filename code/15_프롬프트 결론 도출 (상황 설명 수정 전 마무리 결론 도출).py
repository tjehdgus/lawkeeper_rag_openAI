import os
import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import openai
from IPython.display import display, Markdown

# ───── API Key 및 모델 설정 ─────
openai.api_key = os.getenv("OPENAI_API_KEY")  # 실제 키는 보안을 위해 마스킹 처리
MODEL_NAME = "jhgan/ko-sbert-nli"
N_RESULTS = 3

# ───── 초기 상황 설명 프롬프트 (1회 출력용) ─────
INITIAL_PROMPT = """
정확한 상담을 위해 아래 내용을 포함해 사건 상황을 간단히 정리해 주세요:

- 사건 발생 시점
- 상대방과의 관계 (가족, 친구, 회사, 낯선 사람 등)
- 피해 유형 및 규모
- 증거의 존재 여부 (문자, 녹음, 계약서 등)
- 신고/법적 조치 여부

🧾 예시:
- “3개월 전에 친구에게 100만 원을 빌려줬는데 갚지 않고 있습니다. 카카오톡 대화 내용은 저장해놨고, 아직 고소는 하지 않았습니다.”
- “어제 회사 동료가 협박성 문자를 보냈고, 아직 신고하지 않았습니다. 문자 캡처는 보관 중입니다.”

📢 위와 같이 상황을 편하게 정리해 주시면 정확한 조언에 큰 도움이 됩니다.
"""

# ───── 시스템 역할 정의 프롬프트 ─────
SYSTEM_PROMPT = """
당신은 법률에 익숙하지 않은 일반인을 위한 상담 전문 AI입니다.

📌 사용자가 상황 설명을 입력한 경우에는 아래 항목을 반드시 포함하여 문서 기반 상담을 제공하세요:

━━━━━━━━━━━━━━━━━━━━━━
🧠 필수 상담 항목
━━━━━━━━━━━━━━━━━━━━━━

1. ✅ **관련 법적 쟁점 요약**  
   - 사건에서 핵심적으로 다뤄야 할 법률 쟁점을 일반적인 언어로 요약해 주세요.

2. ⚖️ **적용 가능한 법령**  
   - 관련 법률 조항의 조문 번호와 내용을 명시하고, 일반인이 이해하기 쉽게 설명하세요.  
   - **반드시 포함되어야 합니다.**

3. 📚 **유사 사례 (판례)**  
   - 참고 문서에 등장한 유사한 판결, 판례, 결정례가 있다면 1~2건 이상 소개하고,  
     각 사건의 요지 및 판결 요점을 요약하세요.  
   - **반드시 포함되어야 합니다.**

4. 🧭 **현실적인 대응방안**  
   - 사용자가 취할 수 있는 법적 조치 또는 기관 신고, 그에 따른 절차 또는 예상 흐름

5. 🧾 **지금 할 수 있는 실천적 행동**  
   - 예: 증거 확보, 통화 녹음, 대화 캡처, 서면 요청 등

6. 🧠 **주의사항 및 실수 예방**  
   - 시효, 오해, 감정 대응 등 법적 리스크 경고

7. 🗂️ **상담 요약 (간단히 정리)**  
   - 전체 핵심을 4~5줄 이내로 정리

8. ❓ **전문가 상담 전 준비 질문**  
   - 변호사나 노동청에 갈 때 꼭 확인해야 할 질문 3~5개 제안

━━━━━━━━━━━━━━━━━━━━━━
⚠️ 유의사항
━━━━━━━━━━━━━━━━━━━━━━

- 문서 기반 정보를 우선적으로 사용하고, 부족할 경우 일반적인 판례 경향 또는 실무 경험을 바탕으로 조심스럽게 조언하세요.  
- 법률 판단은 항상 **유보적 표현** 사용 (“~일 수 있습니다”, “판단될 가능성이 있습니다”)  
- 설명은 **따뜻하고 쉬운 언어**로 구성하세요.

⚠️ 본 답변은 참고용이며, 실제 법적 분쟁이나 계약 체결 전에는 반드시 변호사의 자문을 받으시기 바랍니다.

"""

# ───── 컬렉션 목록 ─────
COLLECTIONS = [
    {"name": "legal_qa_rag_docs", "path": r"C:\\law\\chroma_qa"},
    {"name": "legal_rag_docs", "path": r"C:\\law\\chroma_rag"},
    {"name": "legal_cases", "path": r"C:\\law\\chroma_판결문_db"},
    {"name": "terms_clauses", "path": r"C:\\law\\chroma_약관_DB"},
]

embedding_model = SentenceTransformer(MODEL_NAME)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

collections = []
for col in COLLECTIONS:
    client = PersistentClient(path=col["path"])
    try:
        client.delete_collection(name=col["name"])
    except Exception as e:
        print(f"⚠️ 컬렉션 삭제 실패 또는 존재하지 않음: {col['name']} -> {e}")

    collection = client.get_or_create_collection(
        name=col["name"],
        embedding_function=embedding_function
    )
    collections.append(collection)

chat_history = []
INITIAL_MODE = True
case_context = ""

def run_chat():
    global INITIAL_MODE, case_context

    print("📚 법률 챗봇에 오신 걸 환영합니다! 질문을 입력하세요 ('종료' 입력 시 종료).")
    if INITIAL_MODE:
        print(INITIAL_PROMPT)

    while True:
        query = input("\n👤 질문: ").strip()
        if query.lower() in ["종료", "exit", "quit"]:
            print("👋 챗봇을 종료합니다.")
            break

        if query.lower() in ["처음부터", "리셋", "reset", "start", "시작"]:
            INITIAL_MODE = True
            chat_history.clear()
            case_context = ""
            print("🔄 초기 질문 유도 단계로 돌아갑니다.")
            print(INITIAL_PROMPT)
            continue

        # 초기 상황 설명을 받은 경우 저장
        if INITIAL_MODE:
            case_context = query
            INITIAL_MODE = False
            print("✅ 상황 설명이 등록되었습니다. 이제 상담을 시작합니다.")

        # 유사도 검색을 위해 상황 설명을 기준으로 검색
        query_embedding = embedding_model.encode([case_context])[0].tolist()
        threshold = 0.3
        all_docs = []

        for collection in collections:
            results = collection.query(query_embeddings=[query_embedding], n_results=N_RESULTS)
            docs = results["documents"][0] if results["documents"] else []
            scores = results["distances"][0] if results["distances"] else []

            for doc, score in zip(docs, scores):
                if score >= threshold:
                    all_docs.append(doc)

        context_text = "\n\n".join(all_docs[:N_RESULTS * len(collections)])
        if not context_text.strip():
            context_text = "❗ 참고 문서가 부족합니다. 일반적인 법률 상식과 실무 관행에 따라 조심스럽게 조언해 주세요."

        full_prompt = f"""
📌 사용자 질문: {query}

📄 참고 문서:
{context_text}
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case_context},
        ] + chat_history + [{"role": "user", "content": query}]

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            answer = response.choices[0].message.content

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})

            display(Markdown(f"🧠 **GPT 응답:**\n\n{answer}"))

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

# ───── 실행 ─────
run_chat()