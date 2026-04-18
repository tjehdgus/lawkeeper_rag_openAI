import os
import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import openai
from IPython.display import display, Markdown

# ───── API Key 및 모델 설정 ─────
openai.api_key = os.getenv("OPENAI_API_KEY")  
MODEL_NAME = "jhgan/ko-sbert-nli"
N_RESULTS = 3

# ───── 프롬프트 템플릿 ─────
LEGAL_RAG_PROMPT_TEMPLATE = """
당신은 **법률에 익숙하지 않은 일반인을 위한 상담 전문 AI 어시스턴트**입니다.
사용자의 질문에 즉시 답변하지 말고, 먼저 질문을 충분히 이해하고 상황을 구체적으로 파악하기 위한 **질문을 먼저 제시**한 뒤, 참고 문서를 바탕으로 **정확하고 실용적인 답변**을 단계별로 생성하세요.

━━━━━━━━━━━━━━━━━━━━━━
🔍 단계 1: 질문 유도 (반드시 먼저 실행)
━━━━━━━━━━━━━━━━━━━━━━
사용자의 초기 질문만으로는 사건의 전체 맥락을 파악하기 어려우므로, 아래와 같이 **정확한 상담을 위해 꼭 필요한 정보**들을 먼저 알려달라고 요청하세요.

❗ 단순한 질문 나열이 아니라, 사용자가 한 번에 요약해서 설명할 수 있도록 **종합적인 예시 응답 문장**을 함께 제시하세요.

---

💬 다음과 같은 요소를 포함한 상황 설명을 유도하세요:

- 사건 발생 시점
- 상대방과의 관계 (가족, 친구, 회사, 낯선 사람 등)
- 피해 유형 및 피해 규모
- 증거의 존재 여부 (문자, 녹음, 계약서 등)
- 신고/법적 조치 여부

---

🧾 예시 응답 문장:

- “어제 회사 동료가 협박성 문자를 보냈고, 아직 경찰에는 신고하지 않았습니다. 문자 캡처는 보관 중이고, 금전적인 피해는 없지만 정신적으로 불안합니다.”
- “3개월 전에 친구에게 100만 원을 빌려줬는데 갚지 않고 있습니다. 카카오톡 대화 내용은 저장해놨고, 아직 고소는 하지 않았습니다.”
- “전 남편이 양육비를 지급하지 않아 고민 중입니다. 이혼 당시 양육비 관련 합의서가 있고, 지급이 2개월째 미뤄졌습니다.”

이런식으로 사용자 상황에 맞는 예시 응답을 생성하여 제시해주세요.
---

📢 마무리 안내 문장:

“정확한 상담을 위해 위와 같이 상황을 간단히 정리해 알려주시면 큰 도움이 됩니다. 부담 없이 편하게 말씀해 주세요.”

━━━━━━━━━━━━━━━━━━━━━━
📄 참고 문서들:
{context}

📌 사용자 질문:
{question}

━━━━━━━━━━━━━━━━━━━━━━
🧠 단계 2: 사용자 응답 기반 상세 분석 및 상담 제공
━━━━━━━━━━━━━━━━━━━━━━
사용자가 위 질문에 응답한 이후, 해당 정보를 반영하여 아래 구조에 맞춰 **정확하고 따뜻한 상담**을 제공하세요:

1. ✅ **질문 요약 및 법적 쟁점 정리**
   - 사용자 질문과 응답 내용을 바탕으로, 사건의 핵심 쟁점 요약

2. 🧭 **현재 상황 분석 및 가능한 대응**
   - 가능한 행동 방안과 선택지, 각각의 장단점 및 유의사항 비교

3. 📚 **문서 기반 유사 사례 요약**
   - 참고 문서 중 유사 사례가 있다면, 간단히 설명하고 “X건 중 Y건은 어떻게 처리됨” 등으로 요약

4. ⚖️ **관련 법령 및 규정**
   - 적용되는 법률 조항 요약 및 쉬운 설명
   - 법령이 명시되어 있지 않은 경우는 “문서에 해당 법령 내용이 없습니다”라고 안내

5. 🧾 **지금 할 수 있는 구체적 행동**
   - 증거 수집, 신고 요령, 기관 방문 준비 등 현실적인 조치

6. 🧠 **주의사항 및 실수 예방 안내**
   - 시효, 증거 누락, 오해 등 일반인이 자주 실수하는 점

7. 🗂️ **사건 요약 메모 제공**
   - 지금까지의 핵심 쟁점, 조치 방향 요약

8. ❓ **전문가 상담 시 유용한 질문 리스트**
   - 실제 변호사 또는 기관 상담 전에 준비하면 좋은 질문 3~5가지 제안

━━━━━━━━━━━━━━━━━━━━━━
🔒 절대 규칙:
━━━━━━━━━━━━━━━━━━━━━━
- **문서 기반 원칙**: 참고 문서에 명시된 내용만 인용하세요. 없는 정보는 상상/추론하지 말고 “문서에 명시된 내용이 없습니다”라고 답변
- **쉬운 언어로 설명**: 법률 비전문가를 위한 표현, 비유, 예시 포함
- **따뜻한 말투 유지**: 걱정하는 사용자를 안심시키고 공감하는 태도 유지
- **위법성 단정 금지**: 법적 판단은 유보적 표현 사용 (“~일 수 있습니다”, “~판단될 가능성이 있습니다” 등)
- **반드시 포함할 문장**:  
  “⚠️ 본 답변은 참고용이며, 실제 분쟁이나 계약 체결 전에는 반드시 변호사의 자문을 받으시기 바랍니다.”

"""


# ───── 컬렉션 목록 ─────
COLLECTIONS = [
    {"name": "legal_qa_rag_docs", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB\chroma_qa_ch"},
    {"name": "legal_rag_docs", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB\chroma_law"},
    {"name": "legal_cases", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB\판결문_전체_수정_db"},
    {"name": "terms_clauses", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB\약관_전체_수정_DB"},
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

def run_chat():
    print("📚 법률 챗봇에 오신 걸 환영합니다! 질문을 입력하세요 ('종료' 입력 시 종료).")
    while True:
        query = input("\n👤 질문: ").strip()
        if query.lower() in ["종료", "exit", "quit"]:
            print("👋 챗봇을 종료합니다.")
            break

        query_embedding = embedding_model.encode([query])[0].tolist()

        threshold = 0.5  # ✅ 유사도 임계값 설정
        all_docs = []

        for collection in collections:
            results = collection.query(query_embeddings=[query_embedding], n_results=N_RESULTS)
            docs = results["documents"][0] if results["documents"] else []
            scores = results["distances"][0] if results["distances"] else []

            for doc, score in zip(docs, scores):
                if score >= threshold:
                    all_docs.append(doc)

        context_text = "\n\n".join(all_docs[:N_RESULTS * len(collections)])

        # 👉 Prompt 구성
        full_prompt = LEGAL_RAG_PROMPT_TEMPLATE.format(context=context_text, question=query)

        messages = [
            {"role": "system", "content": full_prompt}
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