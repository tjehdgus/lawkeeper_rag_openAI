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
N_RESULTS = 10

# ───── 초기 프롬프트 ─────
INITIAL_PROMPT = """
정확한 상담을 위해 아래 내용을 포함해 사건 상황을 간단히 정리해 주세요:
- 사건 발생 시점
- 상대방과의 관계 (가족, 친구, 회사, 낯선 사람 등)
- 피해 유형 및 규모
- 증거의 존재 여부 (문자, 녹음, 계약서 등)
- 신고/법적 조치 여부

🧾 예시:
- "3개월 전에 친구에게 100만 원을 빌려줬는데 갚지 않고 있습니다. 카카오톡 대화 내용은 저장해놨고, 아직 고소는 하지 않았습니다."
- "어제 회사 동료가 협박성 문자를 보냈고, 아직 신고하지 않았습니다. 문자 캡처는 보관 중입니다."
"""

# ───── 히스토리 반영 RAG 필요성 판단 프롬프트 ─────
RAG_DECISION_PROMPT_WITH_HISTORY = """
당신은 사용자의 질문이 문서 기반 법률 자문(RAG)이 필요한지 판단하는 전문가입니다.

**중요: 대화 히스토리와 현재 질문의 연관성을 고려하여 판단하세요.**

다음 기준에 따라 판단하세요:

**RAG 필요 (YES):**
- 구체적인 법률 조항이나 판례가 필요한 경우
- 복잡한 법적 절차나 요건을 묻는 경우
- 특정 상황에 대한 정확한 법적 근거가 필요한 경우
- 유사 사례나 판례 분석이 필요한 경우
- 전문적인 법률 지식이 요구되는 경우
- **이전 자문에서 언급된 법적 내용에 대한 추가적인 세부 질문**
- **기존 자문의 법률 조항이나 절차에 대한 구체적인 질문**
- **"그 법에서는...", "그 조항에 따르면...", "판례에서는..." 같은 참조형 질문**

**RAG 불필요 (NO):**
- 간단한 일반 상식 수준의 질문
- 절차적 안내나 방법론적 질문 (단, 법적 근거가 필요 없는 경우)
- 감정적 위로나 격려가 필요한 경우
- 이미 제공된 자문에 대한 단순 확인이나 이해 질문
- 개념 설명이나 용어 정의 (기본적인 수준)
- **기존 자문 내용의 단순 반복 요청**
- **"어떻게 해야 하나요?" 같은 일반적인 행동 지침 질문**

**히스토리 분석 포인트:**
1. 현재 질문이 이전 자문의 특정 법적 내용을 깊이 있게 묻는가?
2. 새로운 법적 근거나 추가 판례가 필요한 질문인가?
3. 기존 답변으로 충분히 대답 가능한 단순 질문인가?

전체 대화 맥락을 고려하여 'YES' 또는 'NO'로만 답변하세요.
"""

# ───── 시스템 프롬프트들 ─────
SYSTEM_PROMPT = """
당신은 법률에 익숙하지 않은 일반인을 위한 상담 전문 AI입니다.

📌 상황 설명이 충분한 경우, 아래 항목을 반드시 포함하여 단 1회 문서 기반 자문을 제공합니다:
📌 이번 자문은 반드시 **오로지 참고 문서에 기반**하여 생성되어야 하며, GPT의 일반 지식, 해석, 추론은 **절대 포함되지 않아야 합니다.**
📌 아래 [참고 문서]에서 반드시 실제 문장이나 내용을 **직접 인용하거나 요약**해서 사용해야 합니다.  
문서 내용과 무관한 판단, 설명, 요약은 절대 포함해서는 안 됩니다.  
문서의 표현을 복사하거나 요약하지 않고 생성된 자문은 무효입니다.

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
   
📌 참고 문서에 **직접적으로 등장하지 않는 정보는 절대 포함하지 마세요.**  
📌 문서에 유사한 사례가 없으면 "참고 문서에 해당 정보가 없습니다."라고 명확히 답변하세요.

⚠️ 유의사항:
- 자문은 상황 설명이 충분해진 시점에 1회 자동으로 출력됩니다.
- 이후 질문은 자문 형식 없이 유동적으로 대응합니다.
- 설명은 쉬운 언어로, 판단은 유보적 표현을 사용하세요.
"""

# ───── 자문 이후 follow-up 질문에 대한 프롬프트 ─────
FOLLOWUP_SYSTEM_PROMPT = """
당신은 법률 자문 이후 추가 질문에 대해 유동적으로 응답하는 상담 AI입니다.
- 자문은 반복하지 말고, 사용자의 질문에 대해 핵심적으로 실용적인 답변을 제공하세요.
- 질문에 답하는 데 필요한 설명만 제공하고, 기존 자문 형식을 반복하지 마세요.
- 문서에 기반할 수 있다면 활용하고, 없다면 GPT의 일반적 법률 지식을 사용해도 됩니다.
- 단, 법적 판단은 유보적으로 표현하고 "실제 사례에 따라 달라질 수 있습니다"는 안내를 포함하세요.
- **대화 히스토리를 참고하여 이전 자문과 연관된 질문에 일관성 있게 답하세요.**
"""

# ───── RAG 기반 응답 프롬프트 ─────
RAG_BASED_RESPONSE_PROMPT = """
당신은 법률 문서를 기반으로 정확한 답변을 제공하는 전문 상담 AI입니다.

📌 **문서 기반 응답 원칙:**
- 반드시 제공된 참고 문서의 내용만을 사용하여 답변하세요
- 문서에 없는 내용은 추측하거나 일반 지식으로 보완하지 마세요
- 관련 법조문이나 판례가 있다면 정확히 인용하세요
- 문서에 정보가 부족하면 솔직히 "참고 문서에 해당 정보가 없습니다"라고 답변하세요
- **대화 히스토리를 고려하여 이전 자문과 연관된 내용을 일관성 있게 제공하세요**

답변 형식:
- 간결하고 실용적으로 작성
- 법률 용어는 일반인이 이해하기 쉽게 설명
- 필요시 구체적인 절차나 방법 안내
- 이전 대화 내용과의 연결점 명시
"""

COLLECTIONS = [
    {"name": "legal_qa_rag_docs", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB_sbert\legal_qa_rag_docs"},
    {"name": "legal_rag_docs", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB_sbert\legal_qa_rag_docs"},
    {"name": "legal_cases", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB_sbert\legal_cases"},
    {"name": "terms_clauses", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB_sbert\terms_clauses"},
    {"name": "legal_laws", "path": r"C:\Users\sdh\Desktop\ai_agent_project\ChromaDB_sbert\legal_laws"}
]

# ───── 전역 변수 ─────
embedding_model = SentenceTransformer(MODEL_NAME)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

collections = []
for col in COLLECTIONS:
    try:
        client = PersistentClient(path=col["path"])
        collection = client.get_or_create_collection(name=col["name"])
        collections.append(collection)
        print(f"✅ {col['name']} 컬렉션 로드 완료")
    except Exception as e:
        print(f"❌ {col['name']} 컬렉션 로드 실패: {e}")

chat_history = []
INITIAL_MODE = True
case_context = ""
advice_given = False

# ───── 핵심 요약 생성 함수 ─────
def summarize_context(context: str) -> str:
    """사용자 입력을 핵심 내용으로 요약"""
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
    except Exception as e:
        print(f"❌ 요약 생성 오류: {e}")
        return context[-500:]

# ───── 대화 히스토리 요약 함수 ─────
def summarize_chat_history(chat_history: list, max_length: int = 1000) -> str:
    """대화 히스토리를 요약하여 컨텍스트로 제공"""
    if not chat_history:
        return ""
    
    # 최근 대화 내용을 우선적으로 포함
    recent_messages = chat_history[-6:]  # 최근 3번의 질답 (6개 메시지)
    
    history_text = ""
    for msg in recent_messages:
        role = "사용자" if msg["role"] == "user" else "AI"
        content = msg["content"][:200]  # 각 메시지는 200자로 제한
        history_text += f"[{role}] {content}\n"
    
    # 길이 제한
    if len(history_text) > max_length:
        history_text = history_text[:max_length] + "..."
    
    return history_text

# ───── 히스토리를 반영한 RAG 필요성 판단 함수 ─────
def needs_rag_response_with_history(query: str, context: str = "", chat_history: list = None) -> bool:
    """대화 히스토리를 반영하여 질문이 RAG 기반 응답이 필요한지 판단"""
    print(f"\n🔍 [RAG 판단 디버깅] 히스토리 반영 질문 분석 시작")
    print(f"📝 입력 질문: '{query}'")
    print(f"📋 컨텍스트 길이: {len(context)} 문자")
    print(f"💬 대화 히스토리 항목 수: {len(chat_history) if chat_history else 0}")
    
    # 대화 히스토리 요약
    history_summary = summarize_chat_history(chat_history or [])
    
    # 전체 컨텍스트 구성
    full_context = f"""
[원본 사건 컨텍스트]
{context}

[최근 대화 히스토리]
{history_summary}

[현재 질문]
{query}
    """.strip()
    
    messages = [
        {"role": "system", "content": RAG_DECISION_PROMPT_WITH_HISTORY},
        {"role": "user", "content": full_context}
    ]
    
    print(f"🤖 GPT에게 히스토리 반영 RAG 필요성 판단 요청 중...")
    print(f"📊 히스토리 요약 길이: {len(history_summary)} 문자")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=10
        )
        decision = response.choices[0].message.content.strip().upper()
        
        print(f"🎯 GPT 원본 응답: '{response.choices[0].message.content.strip()}'")
        print(f"✅ 최종 판단: {decision}")
        print(f"🔧 RAG 사용 여부: {'YES' if decision == 'YES' else 'NO'}")
        print("─" * 50)
        
        return decision == "YES"
    except Exception as e:
        print(f"❌ RAG 판단 오류: {e}")
        print(f"🛡️ 오류 시 기본값: RAG 사용")
        print("─" * 50)
        return True  # 오류 시 안전하게 RAG 사용

# ───── 충분성 판단 및 follow-up 유도 ─────
def is_description_sufficient(context: str):
    """사건 설명이 자문을 시작하기에 충분한지 판단"""
    summarized = summarize_context(context)

    def contains_negative_clues(text):
        negs = ["모르겠", "모름", "없", "기억 안", "확실치 않", "정보 없음", "정확하지 않"]
        return any(neg in text for neg in negs)

    # 부정적 단서가 있어도 충분히 길면 진행
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
        print(f"❌ 충분성 판단 오류: {e}")
        return "ERROR"

# ───── RAG 문서 검색 함수 ─────
def search_documents(query: str, max_docs: int = None) -> str:
    """RAG 문서에서 관련 문서 검색"""
    if max_docs is None:
        max_docs = N_RESULTS
        
    query_embedding = embedding_model.encode([query])[0].tolist()
    all_docs = []

    for collection in collections:
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=max_docs)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                print(f"📁 {collection.name}: {len(docs)}개 문서 발견")
                for i, doc in enumerate(docs, 1):
                    preview = doc.strip().replace("\n", " ")[:150]
                    print(f"🔹 문서 {i}: {preview}...")
                all_docs.extend(docs)
            else:
                print(f"⚠️ {collection.name}: 관련 문서 없음")
        except Exception as e:
            print(f"❌ {collection.name} 검색 오류: {e}")

    context_text = "\n\n".join(all_docs[:max_docs * len(collections)])
    if not context_text.strip():
        context_text = "❗ 참고 문서가 부족합니다."
    
    return context_text

# ───── 히스토리를 반영한 RAG 기반 응답 생성 ─────
def generate_rag_response_with_history(query: str, context: str, document_context: str, chat_history: list = None) -> str:
    """대화 히스토리를 반영한 RAG 문서 기반 응답 생성"""
    history_summary = summarize_chat_history(chat_history or [])
    
    messages = [
        {"role": "system", "content": RAG_BASED_RESPONSE_PROMPT},
        {"role": "user", "content": f"[사건 컨텍스트]\n{context}"},
        {"role": "user", "content": f"[대화 히스토리]\n{history_summary}"},
        {"role": "user", "content": f"[참고 문서]\n{document_context}"},
        {"role": "user", "content": f"[질문]\n{query}"}
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ RAG 응답 생성 오류: {e}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

# ───── 일반 응답 생성 ─────
def generate_general_response(query: str, context: str) -> str:
    """일반적인 응답 생성 (RAG 없이)"""
    messages = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {"role": "user", "content": f"[이전 사건 요약 및 대화 흐름]\n{context}"}
    ] + chat_history + [{"role": "user", "content": query}]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 일반 응답 생성 오류: {e}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

# ───── 초기 자문 생성 ─────
def generate_initial_advice(case_context: str) -> str:
    """초기 사건에 대한 포괄적 법률 자문 생성"""
    document_context = search_documents(case_context, N_RESULTS)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[사건 및 대화 흐름]\n{case_context}"},
        {"role": "user", "content": f"[참고 문서]\n{document_context}"},
        {"role": "user", "content": "⛔ 아래 참고 문서에 없는 내용을 인용하거나 자문 형식을 임의로 생성하지 마세요.\n✅ 각 항목에 반드시 문서 내용이 들어가야 하며, 문서 내용과 무관한 주장은 금지됩니다."}
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 초기 자문 생성 오류: {e}")
        return "죄송합니다. 법률 자문 생성 중 오류가 발생했습니다."

# ───── 메인 대화 루프 ─────
def run_chat():
    global INITIAL_MODE, case_context, advice_given

    print("📚 법률 챗봇에 오신 걸 환영합니다! 질문을 입력하세요 ('종료' 입력 시 종료).")
    if INITIAL_MODE:
        print(INITIAL_PROMPT)

    while True:
        query = input("\n👤 질문: ").strip()
        
        # 종료 명령어 처리
        if query.lower() in ["종료", "exit", "quit"]:
            print("👋 챗봇을 종료합니다.")
            break

        # 리셋 명령어 처리
        if query.lower() in ["처음부터", "리셋", "reset", "start", "시작"]:
            INITIAL_MODE = True
            chat_history.clear()
            case_context = ""
            advice_given = False
            print("🔄 초기 질문 유도 단계로 돌아갑니다.")
            print(INITIAL_PROMPT)
            continue

        # 케이스 컨텍스트 업데이트
        case_context += "\n" + query

        # 초기 모드: 사건 설명 충분성 판단
        if INITIAL_MODE:
            result = is_description_sufficient(case_context)
            if result.strip().upper() == "YES":
                INITIAL_MODE = False
                print("✅ 사건 설명이 충분합니다. 포괄적인 법률 자문을 제공합니다:")
                
                # 초기 자문 생성
                advice = generate_initial_advice(case_context)
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": advice})
                display(Markdown(f"🧠 **법률 자문:**\n\n{advice}"))
                advice_given = True
            else:
                print(f"💬 {result}")
            continue

        # 자문 완료 후 추가 질문 처리 (히스토리 반영)
        if advice_given:
            # 히스토리를 반영한 RAG 필요성 판단
            if needs_rag_response_with_history(query, case_context, chat_history):
                print("🔍 문서 기반 정밀 검색을 수행합니다... (히스토리 반영)")
                document_context = search_documents(query, N_RESULTS // 2)
                answer = generate_rag_response_with_history(query, case_context, document_context, chat_history)
                response_type = "📚 **문서 기반 응답 (히스토리 반영)**"
            else:
                print("💡 일반 상식 기반으로 응답합니다... (히스토리 반영)")
                answer = generate_general_response(query, case_context)
                response_type = "🧠 **일반 응답 (히스토리 반영)**"
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            display(Markdown(f"{response_type}:\n\n{answer}"))

if __name__ == "__main__":
    run_chat()