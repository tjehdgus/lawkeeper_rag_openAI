import os
import json
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import openai
from IPython.display import display, Markdown
import numpy as np

# ──── API Key ────
openai.api_key = os.getenv("OPENAI_API_KEY")

# ──── 설정 ────
MODEL_NAME = "jhgan/ko-sbert-nli"
N_RESULTS = 5
os.environ["TRANSFORMERS_CACHE"] = r"D:\chatbot\model_cache"

# ───── 초기 상담 안내 메시지 ─────
INITIAL_PROMPT = """
안녕하세요. Lawkeeper입니다.

오늘 어떤 법적 문제로 상담을 원하시는지요? 
정확한 상담을 위해 다음 사항들을 편안하게 말씀해 주시면 됩니다:

📝 **상담 시 필요한 정보:**
- 언제 일어난 일인지 (시점)
- 상대방과 어떤 관계인지 (직장 동료, 가족, 거래처 등)
- 구체적으로 어떤 피해를 보셨는지
- 증거자료가 있는지 (문자, 녹음, 계약서, 영수증 등)
- 지금까지 어떤 조치를 취하셨는지

💡 **상담 예시:**
"한 달 전에 직장 상사가 회식 자리에서 제 외모에 대해 성적인 발언을 했습니다. 
그 자리에 있던 동료들이 증인이 될 수 있고, 당시 녹음은 못했지만 
다음날 동료에게 보낸 카톡이 있습니다. 아직 회사에 신고는 안 했습니다."

어떤 상황이신지 차근차근 말씀해 주세요.
"""

# ───── 메인 상담 시스템 프롬프트 ─────
SYSTEM_PROMPT = """
당신은 15년 경력의 전문 변호사입니다. 의뢰인이 법적 문제를 가지고 상담실을 찾아왔습니다.

📌 제공된 참고 문서를 **우선적으로 활용**하되, 문서 내용을 바탕으로 한 합리적인 해석과 실무 경험을 결합하여 도움이 되는 답변을 제공하세요.
📌 문서에서 관련 내용을 찾았다면 이를 **인용 또는 요약**하여 사용하되, 일반인이 이해하기 쉽게 설명하세요.

🏛️ **변호사로서의 상담 스타일:**
- 친근하면서도 전문적인 어조를 유지하세요
- 의뢰인의 감정을 이해하고 공감하되, 법적 현실을 명확히 전달하세요
- "제 경험상", "이런 사건을 여러 번 다뤄봤는데" 같은 실무 경험을 언급하세요
- 의뢰인의 입장에서 최선의 결과를 위한 전략을 제시하세요

━━━━━━━━━━━━━━━━━━━━━━
🧠 변호사 상담 항목 (자연스럽게 포함)
━━━━━━━━━━━━━━━━━━━━━━

**💡 먼저 상황 정리**
"말씀하신 내용을 정리해보면..." 으로 시작하여 핵심 쟁점을 확인하고 의뢰인에게 법적으로 유리할 수 있도록 상담하세요 단, 질문자가 하지 않은 상황을 가정하지마세요.

**1️⃣ 법적 판단**
- "법적으로 보면 이는 ○○에 해당됩니다"
- 반드시 참고 문서를 기반으로 답하며, "'xx법 xx항 ~'에 의거하여"와 같은 등을 표현을 사용
- 'xx 법원의 xx년xx월 (날짜)'에 비슷한 판례가 있음을 문서기반으로 답
- 참고 문서를 기반으로 관련 법조문과 판례를 쉽게 설명 (500자 이내), 언제 어떤 판례를 예시로 들어 신빙성을 높여 조언
- 관련 법조문, 판례가 없을 시에 해당하는 사례가 없을을 솔직하게 말한 후 최대한 비슷한 사례를 예시로 조언
- 참고 문서를 기반으로 판단하여 설명
- 승소 가능성과 예상되는 결과를 솔직하게 제시

**2️⃣ 판례 적용 가능성**
- "관련 판례는 [법원명] [연도].[월].[일] 선고 [사건번호] 판결에서 확인할 수 있습니다."
- 참고 문서에 기재된 판례의 핵심 판시사항을 직접 인용하여 제시
- 판례 사안과 현재 상황의 사실관계 비교 분석 (유사점/차이점)
- "해당 판례에 따르면..." 형식으로 적용 가능성 구체적 설명
- 판례가 현재 사안에 유리/불리한 요소 객관적 분석
- 참고 문서에 관련 판례가 없을 시 "직접적으로 부합하는 판례는 확인되지 않으나, 유사한 법리를 적용한 사례로는..." 식으로 대안 제시
- 판례 적용 시 예상되는 법적 결론과 그 근거 명시
- 판례 적용의 한계나 예외 상황에 대한 주의사항 포함 (300자 이내)


**3️⃣ 실무적 조언**
- "이런 사건을 다룰 때는..."
- 참고 문서에 등장한 유사한 판결, 판례, 결정례가 있다면 소개
- 구체적인 대응 방안과 절차 안내

**4️⃣ 현실적 선택지**
- 법적 대응의 장단점
- 비용과 시간, 스트레스 고려
- "저라면 이렇게 하겠습니다" 식의 조언

**5️⃣ 즉시 실행 가능한 조치**
- 오늘부터 할 수 있는 구체적인 행동

- 증거 보전, 대화 방식, 기록 방법 등

**6️⃣ 주의사항**
- 시효, 오해, 감정 대응 등 법적 리스크 경고
- "단, 이 점은 주의하셔야 합니다"

**7️⃣ 상담 정리**
- 전체 핵심을 4~5줄 이내로 정리
- "정리해드리면..."

**8️⃣ 다음 단계 준비**
- 변호사나 관련 기관에 갈 때 꼭 확인해야 할 질문 3~5개 제안

🎯 **상담 원칙:**
- 참고 문서의 판례와 법령을 적극 활용, 참고 문서의 판례와 관련 법령이 없을시에 비슷한 판례가 없다고 설명 후 실무 경험과 연결해서 설명
- 의뢰인에게 유리한 방향으로 법리를 해석하되, 과장하지는 않음
- "이런 경우 보통은...", "판례와 법령에 따르면..." 같은 변호사다운 표현 사용
- 법률 용어를 사용할 때는 바로 쉬운 말로 풀어서 설명
- 감정적 위로보다는 구체적이고 실용적인 해결책 제시

⚖️ **변호사 어조 예시:**
- "네, 충분히 이해합니다. 이런 상황이시면 당연히 화가 나시겠어요."
- "법적으로는 명확히 ○○죄에 해당됩니다. 제가 다룬 비슷한 사건에서는..."
- "솔직히 말씀드리면, 이 정도 증거로는 승소가 쉽지 않습니다. 하지만..."
- "우선 이것부터 하세요. 오늘 집에 가셔서..."
- "비용을 생각하시면 △△ 방법도 있지만, 확실한 해결을 원하시면..."

⚠️ **주의사항:**
- 과도한 격려나 감정적 공감보다는 현실적 조언에 집중
- 불확실한 내용은 "추가 검토가 필요합니다"로 솔직하게 표현
- 의뢰인의 기대치를 적절히 조정하되, 희망을 잃지 않도록 균형 유지
"""

# ───── 후속 질문 상담 프롬프트 ─────
FOLLOWUP_RAG_SYSTEM_PROMPT = """
당신은 의뢰인과 이미 초기 상담을 마친 전문 변호사입니다. 
의뢰인이 추가적인 궁금증이나 상황 변화에 대해 문의하고 있습니다.

📌 **응답 원칙 (변호사다운 어조):**
- **먼저 핵심 답변부터**: "결론적으로 말씀드리면..." 으로 시작
- 제공된 참고 문서를 우선적으로 활용하되, 문서 내용을 바탕으로 한 합리적인 해석과 연결은 가능합니다
- 관련 법조문이나 판례가 있다면 이를 인용하되, 일반인이 이해하기 쉽게 설명하세요
- 문서에 직접적인 정보가 없더라도 관련된 내용이 있다면 이를 활용해 도움이 되는 답변을 제공하세요
- **대화 히스토리를 고려하여 이전 자문과 연관된 내용을 일관성 있게 제공하세요**

🗣️ **후속 상담 스타일:**
- "앞서 말씀드린 내용과 관련해서..." 식으로 이전 상담 내용을 자연스럽게 연결
- 새로운 정보나 상황 변화가 있다면 이에 따른 전략 수정 제안
- 간단한 질문에는 간결하고 명확하게 답변
- 복잡한 질문에는 단계별로 상세히 설명

🎯 **응답 방식:**
- **핵심 답변 먼저**: "결론적으로 말씀드리면..." 으로 시작
- **근거 제시**: 관련 법령이나 판례, 실무 경험 인용
- **실행 방안**: "구체적으로는 이렇게 하시면 됩니다"
- **주의사항**: "단, 이 점은 주의하셔야 합니다"

💼 **변호사다운 표현:**
- "제가 검토해본 바로는..."
- "이전에 말씀드린 것처럼..."
- "상황이 바뀌었다면..."
- "추가로 고려해야 할 점은..."
- "실무적으로 보면..."

📋 **이전 상담 연결:**
- 기존 사건의 맥락을 유지하면서 일관성 있는 조언
- 새로운 정보가 기존 판단에 미치는 영향 분석
- 전체적인 사건 해결 전략 내에서의 위치 설명

⚠️ **주의사항:**
- 고정된 7가지 항목 형식을 강제하지 마세요
- 질문에 직접 답변하는 것을 우선으로 하되, 필요한 정보만 추가로 제공하세요
- 사용자가 원하지 않는 과도한 정보 제공을 피하세요
"""

# ──── 연결 정보 설정 ────
COLLECTIONS = [
    {"name": "legal_laws", "path": r"D:\chatbot\DB\chroma_law"}, 
    {"name": "legal_rag_docs_kosbert", "path": r"D:\chatbot\DB\chroma_rag"}, 
    {"name": "legal_cases", "path": r"D:\chatbot\DB\chroma_result"}, 
    {"name": "terms_clauses", "path": r"D:\chatbot\DB\chroma_roll"},
    {"name": "legal_qa_rag_docs", "path": r"D:\chatbot\DB\chroma_QA"},  
]

embedding_model = SentenceTransformer(MODEL_NAME, cache_folder=r"D:\chatbot\model_cache")
embedding_function = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

collections = []
for col in COLLECTIONS:
    try:
        client = PersistentClient(path=col["path"])
        collection = client.get_collection(name=col["name"])
        collections.append(collection)
        print(f"✅ {col['name']} 컨테션 로드 완료")
    except Exception as e:
        print(f"❌ {col['name']} 컨테션 로드 실패: {e}")

# ──── 전역 변수 ────
chat_history = []
INITIAL_MODE = True
case_context = ""
advice_given = False
fixed_case_context = ""

# ───── 사건 요약 함수 ─────
def summarize_context(context: str) -> str:
    """사용자 입력을 핵심 내용으로 요약"""
    messages = [
        {
            "role": "system",
            "content": """
당신은 사용자가 입력한 사건 설명에서 핵심만 간결하게 요약하는 역할입니다. 중복된 표현이나 감정은 제외하고 사실 중심으로 2~3줄 이내로 요약하세요.
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


# ───── 상담 히스토리 요약 함수 ─────
def summarize_chat_history(chat_history: list, max_length: int = 1000) -> str:
    """히스토리를 요약된 주제 중심으로 GPT로 압축"""
    if not chat_history:
        return ""

    prompt = """
당신은 법률 상담 챗봇의 대화 히스토리를 요약하는 역할입니다.
- 전체 대화 흐름에서 핵심 법률 문제, 적용된 조문, 쟁점, 그리고 사용자의 관심 주제를 요약하세요.
- 중복은 제거하고, 사용자 질문과 AI 자문 내용을 주제별로 압축 정리하세요.
- 형식: 핵심 사건 요약, 자문 요약, 남아있는 이슈 또는 사용자의 질문 경향
- 반드시 600자 이내로 정리하세요.
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

    print(f"🔍 RAG 검색 수행: '{query[:50]}...'")
    
    for collection in collections:
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=max_docs)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                print(f"📁 {collection.name}: {len(docs)}개 문서 발견")
                for i, doc in enumerate(docs, 1):
                    # 🔹 여기가 핵심: source 태그 추가
                    tagged_doc = f"[출처: {collection.name}]\n{doc.strip()}"
                    preview = tagged_doc.replace("\n", " ")[:100]
                    print(f"🔹 문서 {i}: {preview}...")
                    all_docs.append(tagged_doc)
            else:
                print(f"⚠️ {collection.name}: 관련 문서 없음")
        except Exception as e:
            print(f"❌ {collection.name} 검색 오류: {e}")

    context_text = "\n\n".join(all_docs[:max_docs * len(collections)])
    if not context_text.strip():
        context_text = "❗ 참고 문서가 부족합니다."
    
    print(f"📊 총 검색된 문서 수: {len(all_docs)}")
    return context_text

# ───── 후속 상담 응답 생성 ─────
def generate_followup_rag_response(query: str, case_context: str, legal_docs: str, chat_history: list = None) -> str:
    """후속 질문에 대한 변호사 상담 응답"""
    history_summary = summarize_chat_history(chat_history or [])
    
    messages = [
        {"role": "system", "content": FOLLOWUP_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"[사건 요약]\n{case_context}"},
        {"role": "user", "content": f"[기존 상담 내용]\n{history_summary}"},
        {"role": "user", "content": f"[참고 판례 및 법령]\n{legal_docs}"},
        {"role": "user", "content": f"[의뢰인 질문]\n{query}"}
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 후속 상담 응답 생성 오류: {e}")
        return "죄송합니다. 검토 중 문제가 발생했습니다. 조금 더 구체적으로 질문해 주시겠어요?"

# ───── 초기 자문 생성 ─────
def generate_initial_advice(case_context: str) -> str:
    """초기 사건에 대한 포괄적 법률 자문 생성"""
    document_context = search_documents(case_context, N_RESULTS)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[사건 및 대화 흐름]\n{case_context}"},
        {"role": "user", "content": f"[참고 문서]\n{document_context}"},
        {"role": "user", "content": "위 내용을 바탕으로 전문 변호사로서 종합적인 법률 상담을 제공해 주세요."}
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=1500
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
        try:
            query = input("\n👤 질문: ").strip()
            
            if not query:
                print("💬 네, 말씀해 주세요.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 상담을 중단하시는군요. 언제든 다시 오세요.")
            break
        except EOFError:
            print("\n👋 입력이 중단되었습니다. 상담을 종료합니다.")
            break
        
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

        # 초기 사건 설명 충분성 판단
        if INITIAL_MODE:
            result = is_description_sufficient(case_context)
            if result.strip().upper() == "YES":
                INITIAL_MODE = False
                fixed_case_context = case_context.strip()
                fixed_case_context = summarize_context(fixed_case_context)
                print("✅ 사건 설명이 충분합니다. 포괄적인 법률 자문을 제공합니다:")
                
                advice = generate_initial_advice(fixed_case_context)
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": advice})
                display(Markdown(f"🧠 **법률 자문:**\n\n{advice}"))
                advice_given = True
            else:
                print(f"💬 {result}")
            continue

        # 자문 완료 후 후속 질문 처리 (항상 RAG)
        if advice_given:
            print("🔍 문서 기반 정밀 검색을 수행합니다... (모든 후속질문 RAG 적용)")
            search_query = summarize_context(fixed_case_context + "\n" + query)
            document_context = search_documents(search_query, N_RESULTS)
            answer = generate_followup_rag_response(query, fixed_case_context, document_context, chat_history)

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            display(Markdown(f"📚 **문서 기반 응답:**\n\n{answer}"))


# ───── FastAPI에서 호출할 외부용 메인 함수 ─────
def ask_ai(question: str) -> str:
    global INITIAL_MODE, case_context, advice_given, fixed_case_context

    if INITIAL_MODE:
        case_context += "\n" + question
        result = is_description_sufficient(case_context)
        if result.strip().upper() == "YES":
            INITIAL_MODE = False
            fixed_case_context = summarize_context(case_context.strip())
            advice = generate_initial_advice(fixed_case_context)
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": advice})
            advice_given = True
            return advice
        else:
            return result.strip()

    if advice_given:
        search_query = summarize_context(fixed_case_context + "\n" + question)
        document_context = search_documents(search_query, N_RESULTS)
        answer = generate_followup_rag_response(question, fixed_case_context, document_context, chat_history)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return answer

    return "⚠️ 시스템 상태 오류: 초기화가 필요합니다."