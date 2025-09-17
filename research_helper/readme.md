# 논문 빠른 리서치 도우미

## 문제 정의

연구자가 특정 주제 관련 논문들을 한눈에 정리하기 어렵습니다. 방대한 양의 논문 데이터베이스에서 관련성 높은 논문을 찾고, 핵심 내용을 파악하는 데 많은 시간이 소요 되는 것이 주요 문제입니다.

## 아이템 설명

RAG로 논문 초록/메타데이터를 검색하고, Tool use로 "arXiv API"나 "Semantic Scholar API"를 연결합니다. LLM이 요약 및 공통 키워드를 정리하여 연구 동향을 파악할 수 있도록 지원합니다.

## 최종 구현 목표

사용자가 "Diffusion 모델 응용 논문"이라고 검색하면, 최신 논문 5개 요약 + 키워드 트렌드를 제공하는 시스템을 구현합니다.

## 구현 설명

Arxiv API에서도 ArxivRetriever를 사용하여 논문을 검색할 수 있기 때문에 RAG 없이도 구현이 가능함을 확인했습니다. 때문에, 설명서와 조금 다르게 구현을 해보았습니다. ArxivRetriever로 사용자의 입력 주제와 관련된 논문을 찾은 뒤, 이 때 메타데이터와 url를 추출하고, 다시 한 번 Arxiv API로 논문 텍스트 전문을 받은 뒤 이를 Chroma db에 논문을 키로 저장합니다. 이후 langchain을 활용하여 처음 user의 쿼리에 대해서는 "retriever결과를 함께 작성한 summary_prompt | llm | StrOutputParser()" 조합으로 답변을 구성하며, 이후의 후속 질문에 대해선 우선 "route_chain = route_prompt | llm | StrOutputParser()"으로 서칭된 논문 관련 질문인지 라우팅 해주는 모델 chain을 통과합니다. 판단 결과 후속 질문이 서칭된 논문과 관련 있다면 "rag_chain = ({"context": local_retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x: x)} | rag_prompt | llm  | StrOutputParser())", 즉 Chroma db에 저장된 논문 전문들과 RAG를 통해 신뢰성 있는 답변을 주게 하고 서칭된 질문과 관련 없다면 "nonrag_chain = nonrag_prompt | llm | StrOutputParser()"으로 답하도록 설계되었습니다.

## 결과 및 한계

코어 기술 부분은 어느 정도 구현이 된 모습을 볼 수 있습니다. 그러나 다음과 같은 한계점들이 있었습니다.

1. ArxivRetriever가 max docs를 20개로 해도 1~3개의 논문밖에 찾지 못합니다. 여러 번 실험해본 결과 그냥 API자체의 성능이 매우 한계를 갖고 있다는 것을 확인하였습니다. 때문에 이후에는 Semantic Scholar API를 시도해보거나, 다른 서비스와 융합하거나, 아니면 자체로 RAG 형태의 retriever를 구현해야 할 것입니다.
2. 후속 질문 뒤에 새로운 주제에 대한 질문이나, 또 다른 논문을 탐색하는 기능이나, db의 처리까진 구현되지 않았습니다.
