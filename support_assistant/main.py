# %% [markdown]
# # Task 1
# 
# Load all 8 documents, chunk them (a simple per-document chunk, or a smaller fixed-size chunking scheme, is fine given their length), embed each chunk with all-MiniLM-L6-v2, and store the embeddings in a ChromaDB collection.
# 
# 

# %%
import os
2
print(os.getcwd())
from pathlib import Path

docs_path = Path("docs")

# %%
# Load the documents 
from pathlib import Path

docs_path = Path("docs")


def load_documents(docs_path):
    documents = []

    for file in docs_path.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append({
                "text": f.read(),
                "source": file.name
            })

    return documents



# %%
# chunk the documents
def chunk_documents(text, source_name):

    paragraphs = text.strip().split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()

        if not para:
            continue

        if para.startswith("====="):
            continue

        chunks.append({
            "text": para,
            "source": source_name
        })

    return chunks


# Load documents
documents = load_documents(docs_path)

# Chunk documents
all_chunks = []

for document in documents:
    chunks = chunk_documents(
        document["text"],
        document["source"]
    )

    all_chunks.extend(chunks)


print("Documents:", len(documents))
print("Chunks:", len(all_chunks))

for chunk in all_chunks[:8]:
    print("\nSource:", chunk["source"])
    print("Text:", chunk["text"])

# %% [markdown]
# Storing the chunks in Chroma DB


# %%
import chromadb

# %%
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="zepto_policies")

# %%
documents = []
ids = []
metadata = []

for i, chunk in enumerate(all_chunks):
  documents.append(chunk['text'])
  ids.append(f"chunk_{i}")
  metadata.append({"source": chunk["source"]})

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadata
)

# %%
print(documents[0])
print(ids[0])
print(metadata[0])

# %% [markdown]
# # Task 2
# Design a structured prompt template following the role–context–task–format–length skeleton, including at least one explicit negative constraint (e.g., "do not answer using information not present in the provided context") and at least one few-shot example embedded in the prompt.

# %%
prompt_template = """
ROLE:
You are a customer support assistant for Zepto. Your job is to answer
customer questions accurately and clearly using only the information
provided in the context.

CONTEXT:
The following information has been retrieved from Zepto's policy documents:

{context}

TASK:
Answer the customer's question using the provided context.

If the answer is not present in the context, clearly say that the
provided information does not contain the answer.

Do not answer using information that is not present in the provided context.
Do not make up, assume, or infer Zepto policies that are not explicitly
supported by the context.

FEW-SHOT EXAMPLE:

Example 1:
Context:
"Customers can cancel an order before it is dispatched."

Question:
"Can I cancel my order before it is dispatched?"

Answer:
"Yes. According to the provided policy, you can cancel your order before
it is dispatched."

Example 2:
Context:
"Customers can contact support through the Zepto app."

Question:
"Can I contact support by phone?"

Answer:
"The provided information does not specify whether phone support is
available."

FORMAT:
Provide the answer in plain text.
If the context contains relevant information, answer directly and mention
the applicable policy.
If the information is unavailable, clearly state that it is not provided
in the context.

LENGTH:
Keep the answer concise, preferably 2–4 sentences.

CUSTOMER QUESTION:
{question}
"""

# %% [markdown]
# # Task 3
# 
# Build a LangGraph StateGraph with a TypedDict state and at least 3 nodes. Every node's generation step must branch on the MOCK_LLM toggle from above — the mock branch is the required, graded baseline; the real-LLM branch is the optional MOCK_LLM=0 extension:
# 

# %%
# Classify the intent

def classify_intent(query):
    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm == "1":
        keywords = [
            "delivery",
            "return",
            "refund",
            "membership",
            "tracking",
            "cancel",
            "gift card",
            "support hours"
        ]

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in keywords):
            return "policy_question"
        else:
            return "general_question"

# %%
print(classify_intent("What are the delivery charges?"))
print(classify_intent("How can I get a refund?"))
print(classify_intent("Can I cancel my order?"))
print(classify_intent("What is the capital of India?"))
print(classify_intent("Tell me a joke"))

# %%
# Retrive and answer

def retrieve_and_answer(query, intent):

    if intent != "policy_question":
        return "This question does not require retrieval from the Zepto policy corpus."

    # Retrieve top 3 similar chunks
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    # Most similar chunk
    top_chunk = retrieved_chunks[0]

    # Short excerpt
    top_chunk_snippet = top_chunk[:200]

    # Mock mode
    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm == "1":
        return f"Based on the retrieved context: {top_chunk_snippet}"


# %%
query = "What is the delivery policy?"
intent = classify_intent(query)
answer = retrieve_and_answer(query, intent)
print(answer)

# %%
# mock mode direct answer

def direct_answer(query):

    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm == "1":
        return "I can only answer questions about Zepto policies right now."

    # Optional MOCK_LLM=0 extension:
    # Call the LLM directly without retrieval

# %% [markdown]
# Wire a conditional edge from classify_intent that routes to retrieve_and_answer or direct_answer based on the classification, mirroring a graph-based intent router. This routing logic does not itself depend on MOCK_LLM — only the generation step inside each node does.

# %%
# Define the routing function

def route_by_intent(state):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"


# %%
import langgraph


# %%
# Define the graph
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    query: str
    intent: str
    answer: str

def classify_intent_node(state):
    intent = classify_intent(state["query"])

    return {
        "intent": intent
    }


def retrieve_and_answer_node(state):
    answer = retrieve_and_answer(
        state["query"],
        state["intent"]
    )

    return {
        "answer": answer
    }


def direct_answer_node(state):
    answer = direct_answer(state["query"])

    return {
        "answer": answer
    }

# %%
# Create the conditional edge

graph = StateGraph(State)

graph.add_node("classify_intent", classify_intent_node)
graph.add_node("retrieve_and_answer", retrieve_and_answer_node)
graph.add_node("direct_answer", direct_answer_node)

graph.add_edge(START, "classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

graph_app = graph.compile()

# %%
# Invoking the app — policy question (routes to retrieve_and_answer)
result = graph_app.invoke({
    "query": "What is the refund policy?"
})

print(result["intent"])
print(result["answer"])

# %%
# Invoking the app — general question (routes to direct_answer)
result_general = graph_app.invoke({
    "query": "What is the capital of India?"
})

print(result_general["intent"])
print(result_general["answer"])

# %% [markdown]
# # Task 4
# 
# Enforce a JSON output schema on the final answer via a Pydantic model with fields answer (string), sources (list of chunk/document IDs used, empty for general_question answers), and confidence (float 0–1). In mock mode, populate this schema deterministically from your own code — there is no LLM output to fail validation, since none was generated: e.g. sources = the ids of the chunks retrieved for policy_question, empty for general_question; confidence = a fixed value such as 1.0. In the optional MOCK_LLM=0 extension, if the real LLM's raw output fails to validate against this schema, retry up to 2 additional times with a corrective instruction before giving up and returning a clearly marked error response.
# 
# 

# %%
# Define the Pydantic schema

from pydantic import BaseModel, Field
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# %%
# Update retrieve_and_answer

def retrieve_and_answer(query, intent):

    if intent != "policy_question":
        return AnswerResponse(
            answer="This question does not require retrieval from the Zepto policy corpus.",
            sources=[],
            confidence=1.0
        )

    # Retrieve top 3 chunks directly from ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]
    retrieved_ids = results["ids"][0]

    # Mock mode
    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm == "1":

        return AnswerResponse(
            answer="Based on the retrieved context: " + retrieved_chunks[0][:200],
            sources=retrieved_ids,
            confidence=1.0
        )

    # Real-LLM mode (MOCK_LLM=0) — retry up to 2 additional times on validation failure
    prompt = prompt_template.format(
        context="\n\n".join(retrieved_chunks),
        question=query
    )

    last_error = None
    for attempt in range(3):
        try:
            # Placeholder: replace with actual LLM call, e.g.:
            # raw = llm_client.chat(prompt)
            raw_answer = ""  # real LLM call would go here
            return AnswerResponse(
                answer=raw_answer,
                sources=retrieved_ids,
                confidence=0.9
            )
        except Exception as e:
            last_error = e
            prompt += "\n\nThe previous response did not match the required JSON schema. Please respond with a valid answer string only."

    return AnswerResponse(
        answer=f"[ERROR] Could not produce a valid answer after 3 attempts: {last_error}",
        sources=retrieved_ids,
        confidence=0.0
    )


# %%
# direct answer

def direct_answer(query):

    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm == "1":

        return AnswerResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )

    # Real-LLM mode (MOCK_LLM=0) — retry up to 2 additional times on validation failure
    direct_prompt = f"Answer the following customer question concisely:\n\n{query}"

    last_error = None
    for attempt in range(3):
        try:
            # Placeholder: replace with actual LLM call, e.g.:
            # raw = llm_client.chat(direct_prompt)
            raw_answer = ""  # real LLM call would go here
            return AnswerResponse(
                answer=raw_answer,
                sources=[],
                confidence=0.9
            )
        except Exception as e:
            last_error = e
            direct_prompt += "\n\nThe previous response did not match the required JSON schema. Please respond with a valid answer string only."

    return AnswerResponse(
        answer=f"[ERROR] Could not produce a valid answer after 3 attempts: {last_error}",
        sources=[],
        confidence=0.0
    )

# %%
result = retrieve_and_answer(
    "What is the refund policy?",
    "policy_question"
)

print(result.model_dump_json(indent=2))

# %% [markdown]
# # Task 5
# Wrap the graph in a FastAPI application with a POST /ask endpoint that accepts a Pydantic request model ({"query": str}) and returns the validated Pydantic response model above. Run it locally with uvicorn and demonstrate at least 2 example calls (one that should trigger retrieval, one that should not) with their raw JSON responses recorded in your README — run with MOCK_LLM left at its default, since that is what gets graded.

# %%
# Install FastAPI and Uvicorn


# %%
import fastapi
import uvicorn

print("FastAPI:", fastapi.__version__)
print("Uvicorn installed successfully")

# %%
import sys

# %%
from fastapi import FastAPI
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


app = FastAPI()


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):

    result = graph_app.invoke({
        "query": request.query
    })

    return result["answer"]


# %%
from fastapi.testclient import TestClient

client = TestClient(app)

# %%
response = client.post(
    "/ask",
    json={"query": "What is the refund policy?"}
)

print(response.status_code)
print(response.json())

# %%
response = client.post(
    "/ask",
    json={"query": "What is the capital of India?"}
)

print(response.status_code)
print(response.json())



