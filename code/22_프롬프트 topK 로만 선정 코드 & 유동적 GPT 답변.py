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

# ───── 초기 프롬프트 ─────
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
"""

SYSTEM_PROMPT = """
당신은 법률에 익숙하지 않은 일반인을 위한 상담 전문 AI입니다.

📌 상황 설명이 충분한 경우, 아래 항목을 반드시 포함하여 단 1회 문서 기반 자문을 제공합니다:

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

⚠️ 유의사항:
- 자문은 상황 설명이 충분해진 시점에 1회 자동으로 출력됩니다.
- 이후 질문은 자문 형식 없이 유동적으로 대응합니다.
- 설명은 쉬운 언어로, 판단은 유보적 표현을 사용하세요.
"""

FOLLOWUP_SYSTEM_PROMPT = """
당신은 법률 자문 이후 추가 질문에 대해 유동적으로 응답하는 상담 AI입니다.
- 자문은 반복하지 말고, 사용자의 질문에 대해 핵심적으로 실용적인 답변을 제공하세요.
- 질문에 답하는 데 필요한 설명만 제공하고, 기존 자문 형식을 반복하지 마세요.
- 문서에 기반할 수 있다면 활용하고, 없다면 GPT의 일반적 법률 지식을 사용해도 됩니다.
- 단, 법적 판단은 유보적으로 표현하고 "실제 사례에 따라 달라질 수 있습니다"는 안내를 포함하세요.
"""

COLLECTIONS = [
    {"name": "legal_qa_rag_docs", "path": r"C:\\law\\chromadb_backup_kobert\\chroma_qa"},
    {"name": "legal_rag_docs", "path": r"C:\\law\\chromadb_backup_kobert\\chroma_rag"},
    {"name": "legal_cases", "path": r"C:\\law\\chromadb_backup_kobert\\chroma_result"},
    {"name": "terms_clauses", "path": r"C:\\law\\chromadb_backup_kobert\\chroma_roll"},
    {"name": "legal_laws", "path": r"C:\\law\\chromadb_backup_kobert\\chroma_law"}
]

embedding_model = SentenceTransformer(MODEL_NAME)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

collections = []
for col in COLLECTIONS:
    client = PersistentClient(path=col["path"])
    
    # 임베딩된 컬렉션은 embedding_function 없이 열기
    collection = client.get_or_create_collection(name=col["name"])
    
    collections.append(collection)

chat_history = []
INITIAL_MODE = True
case_context = ""
advice_given = False

# ───── 핵심 요약 생성 함수 ─────
def summarize_context(context: str) -> str:
    messages = [
        {"role": "system", "content": "당신은 사용자가 입력한 사건 설명에서 핵심만 간결하게 요약하는 역할입니다. 중복된 표현이나 감정은 제외하고 사실 중심으로 2~3줄 이내로 요약하세요."},
        {"role": "user", "content": context}
    ]
    try:
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except:
        return context[-500:]

# ───── 충분성 판단 및 follow-up 유도 ─────
def is_description_sufficient(context: str):
    summarized = summarize_context(context)

    def contains_negative_clues(text):
        negs = ["모르겠", "모름", "없", "기억 안", "확실치 않", "정보 없음", "정확하지 않"]
        return any(neg in text for neg in negs)

    if contains_negative_clues(context) and len(context.split()) > 30:
        return "YES"

    system_prompt = """
당신은 사용자가 설명한 사건 상황이 법률 자문을 시작하기에 충분한지 판단하는 역할입니다.
- 설명이 구체적이라면 'YES'만 출력하세요.
- 설명이 **명확한 사건 상황(예: 특정 발언, 폭행, 금전 요구, 성적 표현 등)**을 포함한다면 'YES'만 출력하세요.
- 설명이 **모호하거나 배경 정보가 부족**할 경우, **필요한 정보를 유도하는 후속 질문**을 1문장으로 생성하세요.
- 특히, 설명이 간결하더라도 **법적으로 문제가 될 만한 사실이 명확히 드러난 경우**, 추가 질문 없이 'YES'로 처리하세요.
- 질문은 상황 설명을 유도하는 방식이어야 하며, 정보가 없다는 응답도 답변으로 간주해야 합니다.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summarized}
    ]
    try:
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 판단 오류: {e}")
        return "ERROR"

# ───── 메인 대화 루프 ─────
def run_chat():
    global INITIAL_MODE, case_context, advice_given

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
            advice_given = False
            print("🔄 초기 질문 유도 단계로 돌아갑니다.")
            print(INITIAL_PROMPT)
            continue

        case_context += "\n" + query

        if INITIAL_MODE:
            result = is_description_sufficient(case_context)
            if result.strip().upper() == "YES":
                INITIAL_MODE = False
                print("✅ 사건 설명이 충분합니다. 아래는 해당 상황에 대한 법률 자문입니다:")
            else:
                print(result)
                continue

        if not advice_given:
            query_embedding = embedding_model.encode([case_context])[0].tolist()
            all_docs = []

            for collection in collections:
                results = collection.query(query_embeddings=[query_embedding], n_results=N_RESULTS)
                docs = results["documents"][0] if results["documents"] else []
                all_docs.extend(docs)

            context_text = "\n\n".join(all_docs[:N_RESULTS * len(collections)])
            if not context_text.strip():
                context_text = "❗ 참고 문서가 부족합니다. 일반적인 법률 상식과 실무 관행에 따라 조심스럽게 조언해 주세요."

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[사건 및 대화 흐름]\n{case_context}"},
                {"role": "user", "content": f"[참고 문서]\n{context_text}"}
            ]
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=800
                )
                answer = response.choices[0].message.content.strip()
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})
                display(Markdown(f"🧠 **법률 자문:**\n\n{answer}"))
                advice_given = True
                continue
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                continue

        messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": f"[이전 사건 요약 및 대화 흐름]\n{case_context}"}
        ] + chat_history + [{"role": "user", "content": query}]

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
                max_tokens=700
            )
            answer = response.choices[0].message.content.strip()
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            display(Markdown(f"🧠 **GPT 응답:**\n\n{answer}"))
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

run_chat()