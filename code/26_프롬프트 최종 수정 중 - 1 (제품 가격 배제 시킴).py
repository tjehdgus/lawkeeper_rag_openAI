import os
import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import openai
from IPython.display import display, Markdown
import numpy as np

# ───── API Key 및 모델 설정 ─────
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "./ko-sbert-nli"
N_RESULTS =5

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

# ───── 시스템 프롬프트들 ─────
SYSTEM_PROMPT = """
당신은 법률에 익숙하지 않은 일반인을 위한 상담 전문 AI입니다.

📌 상황 설명이 충분한 경우, 아래 항목을 반드시 포함하여 단 1회 문서 기반 자문을 제공합니다:
📌 제공된 참고 문서를 **우선적으로 활용**하되, 문서 내용을 바탕으로 한 합리적인 해석과 일반적인 법률 상식을 결합하여 도움이 되는 답변을 제공하세요.
📌 문서에서 관련 내용을 찾았다면 이를 **인용 또는 요약**하여 사용하되, 일반인이 이해하기 쉽게 설명하세요.

📚 참고 문서 목록:
legal_qa_rag_docs: 일반인 대상 QA 중심 사례
legal_rag_docs_kosbert: 법률 자문 예시 및 실제 사례 기반 설명
legal_cases: 판례 원문, 요약 및 실제 소송 결과 포함
terms_clauses: 서비스 약관 및 소비자 보호 관련 조항
legal_laws: 법령(조문) 텍스트 데이터

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
   
📌 응답 원칙:
- 문서에서 찾은 내용은 적극 활용하되, 이해하기 쉽게 재구성하세요
- 문서에 없는 내용이라도 법률 상식 범위에서 도움이 되는 정보는 제공하세요
- 각 항목에 "해당 없음"이나 "정보 없음"으로 끝내지 말고 최대한 도움이 되는 답변을 제공하세요
- 확실하지 않은 내용은 "참고용으로만 확인하시고 전문가와 상담하세요"로 안내하세요

⚠️ 유의사항:
- 자문은 상황 설명이 충분해진 시점에 1회 자동으로 출력됩니다.
- 이후 질문은 자문 형식 없이 유동적으로 대응합니다.
- 설명은 쉬운 언어로, 판단은 유보적 표현을 사용하세요.
"""

# ───── 자문 이후 follow-up 질문에 대한 프롬프트 (RAG 전용) ─────
FOLLOWUP_RAG_SYSTEM_PROMPT = """
당신은 법률 자문 이후 추가 질문에 대해 **반드시 문서 기반으로만** 응답하는 상담 AI입니다.

📌 **응답 원칙:**
- 제공된 참고 문서를 우선적으로 활용하되, 문서 내용을 바탕으로 한 합리적인 해석과 연결은 가능합니다
- 관련 법조문이나 판례가 있다면 이를 인용하되, 일반인이 이해하기 쉽게 설명하세요
- 문서에 직접적인 정보가 없더라도 관련된 내용이 있다면 이를 활용해 도움이 되는 답변을 제공하세요
- 완전히 관련 없는 경우에만 "참고 문서에 해당 정보가 부족합니다"라고 답변하세요
- **대화 히스토리를 고려하여 이전 자문과 연관된 내용을 일관성 있게 제공하세요**

답변 형식:
- 간결하고 실용적으로 작성
- 법률 용어는 일반인이 이해하기 쉽게 설명
- 필요시 구체적인 절차나 방법 안내
- 이전 대화 내용과의 연결점 명시
- 확실하지 않은 내용은 "참고용으로만 확인하시고 전문가와 상담하세요"로 안내

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
"""

COLLECTIONS = [
    {"name": "legal_qa_rag_docs", "path": "c:\\law\\chromadb_backup_kobert\\chroma_qa"},
    {"name": "legal_rag_docs_kosbert", "path": "c:\\law\\chromadb_backup_kobert\\chroma_rag"},
    {"name": "legal_cases", "path": "c:\\law\\chromadb_backup_kobert\\chroma_result"},
    {"name": "terms_clauses", "path": "c:\\law\\chromadb_backup_kobert\\chroma_roll"},
    {"name": "legal_laws", "path": "c:\\law\\chromadb_backup_kobert\\chroma_law"}
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
fixed_case_context = ""

# ───── 핵심 요약 생성 함수 ─────
def summarize_context(context: str) -> str:
    """사용자 입력을 핵심 내용으로 요약"""
    messages = [
        {
            "role": "system",
            "content": """
    당신은 사용자의 사건 설명에서 핵심을 2~3줄 이내로 요약하는 역할입니다.
    
    - **감정 표현이나 중복된 문장은 제거**하세요.
    - 요약 시 반드시 **사건의 법적 쟁점이 드러나도록** 하세요. (예: 성희롱, 모욕, 사기, 협박, 금전 분쟁, 계약 위반 등)
    - "성희롱으로 보일 수 있는 외모 발언", "협박성 문자", "대출사기", "채무불이행" 같은 **법적 이슈가 되는 언급은 요약에 포함**하세요.
    - 제품 이름이나 가격 등 **구체적 소비재 정보는 배제**하세요.
    """
        },
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
    """히스토리를 요약된 주제 중심으로 GPT로 압축"""
    if not chat_history:
        return ""

    prompt = """
당신은 법률 상담 챗봇의 대화 히스토리를 요약하는 역할입니다.
- 전체 대화 흐름에서 핵심 법률 문제, 적용된 조문, 쟁점, 그리고 사용자의 관심 주제를 요약하세요.
- 질문을 요약을 할때 특정 제품, 금액이 아닌 문맥을 파악하세요.
- 중복은 제거하고, 사용자 질문과 AI 자문 내용을 주제별로 압축 정리하세요.
- 형식: 핵심 사건 요약, 자문 요약, 남아있는 이슈 또는 사용자의 질문 경향
- 반드시 1000자 이내로 정리하세요.
"""
    history_text = ""
    for msg in chat_history[-10:]:
        role = "사용자" if msg["role"] == "user" else "AI"
        content = msg["content"][:300]
        history_text += f"[{role}] {content}\n"

    try:
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": history_text}
            ],
            temperature=0.2,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 히스토리 요약 GPT 호출 실패: {e}")
        return history_text[:max_length]

# ───── 충분성 판단 및 follow-up 유도 ─────
def is_description_sufficient(context: str):
    """사건 설명이 자문을 시작하기에 충분한지 판단"""
    summarized = context

    def contains_negative_clues(text):
        negs = ["모르겠", "모름", "없", "기억 안", "확실치 않", "정보 없음", "정확하지 않"]
        return any(neg in text for neg in negs)

    # 부정적 단서가 있어도 충분히 길면 진행
    if contains_negative_clues(context) and len(context.split()) > 30:
        return "YES"

    system_prompt = """
당신은 사용자가 설명한 사건 상황이 법률 자문을 시작하기에 충분한지 판단하는 역할입니다.

✅ 아래에 해당하는 경우 'YES'만 출력하세요:
- 설명이 **구체적인 사건, 행위, 발언**을 포함하고 있는 경우
- 특히, **성희롱성 발언, 욕설, 모욕, 위협, 폭언 등 감정적으로 불쾌한 표현**이 직접 등장하는 경우
- 설명이 짧더라도 **불쾌한 언사나 비하 발언 등으로 법적 문제가 될 소지가 있는 경우**, 구체적이면 YES

⚠️ 아래에 해당하면 후속 질문을 생성하세요:
- 설명이 지나치게 추상적이거나, 누가 무엇을 했는지 모호할 경우
- 어떤 발언이 있었는지 **간접적으로만** 설명된 경우 (예: "기분이 나빴다" 등)

출력 형식:
- 충분하면 "YES"만 출력
- 부족하면 **한 문장으로 추가 질문** 생성
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

    print(f"🔍 RAG 검색 수행: '{query[:50]}...'")
    
    for collection in collections:
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=max_docs)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                print(f"📁 {collection.name}: {len(docs)}개 문서 발견")
                for i, doc in enumerate(docs, 1):
                    preview = doc.strip().replace("\n", " ")[:100]
                    print(f"🔹 문서 {i}: {preview}...")
                all_docs.extend(docs)
            else:
                print(f"⚠️ {collection.name}: 관련 문서 없음")
        except Exception as e:
            print(f"❌ {collection.name} 검색 오류: {e}")

    context_text = "\n\n".join(all_docs[:max_docs * len(collections)])
    if not context_text.strip():
        context_text = "❗ 참고 문서가 부족합니다."
    
    print(f"📊 총 검색된 문서 수: {len(all_docs)}")
    return context_text

# ───── 히스토리를 반영한 RAG 기반 응답 생성 (후속질문용) ─────
def generate_followup_rag_response(query: str, context: str, document_context: str, chat_history: list = None) -> str:
    """후속 질문에 대한 RAG 문서 기반 응답 생성 (항상 사용)"""
    history_summary = summarize_chat_history(chat_history or [])
    
    messages = [
        {"role": "system", "content": FOLLOWUP_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"[초기 사건 설명 요약]\n{context}"},
        {"role": "user", "content": f"[기존 자문 요약 및 핵심 쟁점]\n{history_summary}"},
        {"role": "user", "content": f"[참고 문서]\n{document_context}"},
        {"role": "user", "content": f"[사용자 질문]\n{query}"}
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
    global INITIAL_MODE, case_context, advice_given, fixed_case_context

    print("📚 법률 챗봇에 오신 걸 환영합니다! 질문을 입력하세요 ('종료' 입력 시 종료).")
    print("🔄 후속 질문은 항상 문서 기반 RAG로 응답합니다.")
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
            fixed_case_context = ""
            print("🔄 초기 질문 유도 단계로 돌아갑니다.")
            print(INITIAL_PROMPT)
            continue

        # 케이스 컨텍스트 업데이트 (초기 모드에서만)
        if INITIAL_MODE:
            case_context += "\n" + query

        # 초기 모드: 사건 설명 충분성 판단
        if INITIAL_MODE:
            result = is_description_sufficient(case_context)
            if result.strip().upper() == "YES":
                INITIAL_MODE = False
                fixed_case_context = case_context.strip()
                fixed_case_context = summarize_context(fixed_case_context)
                print("✅ 사건 설명이 충분합니다. 포괄적인 법률 자문을 제공합니다:")
                
                # 초기 자문 생성
                advice = generate_initial_advice(fixed_case_context)
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": advice})
                display(Markdown(f"🧠 **법률 자문:**\n\n{advice}"))
                advice_given = True
            else:
                print(f"💬 {result}")
            continue

        # 자문 완료 후 추가 질문 처리 (항상 RAG 사용)
        if advice_given:
            print("🔍 문서 기반 정밀 검색을 수행합니다... (모든 후속질문 RAG 적용)")
            
            # 검색 쿼리 구성 (사건 컨텍스트 + 현재 질문)
            search_query = summarize_context(fixed_case_context + "\n" + query)
            
            # RAG 문서 검색
            document_context = search_documents(search_query, N_RESULTS)
            
            # RAG 기반 응답 생성
            answer = generate_followup_rag_response(query, fixed_case_context, document_context, chat_history)
            
            # 히스토리 업데이트
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            
            # 응답 출력
            display(Markdown(f"📚 **문서 기반 응답:**\n\n{answer}"))

if __name__ == "__main__":
    run_chat()