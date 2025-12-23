
from typing import Annotated, Any, Dict, Optional, TypedDict
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.messages import BaseMessage, SystemMessage
import re
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from PIL import Image
import pytesseract
import speech_recognition as sr
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import  RecursiveCharacterTextSplitter, Language
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",  
)

def extract_file_path(text: str) -> Optional[str]:
    match = re.search(r"([\w\-./\\]+\.pdf)", text)
    if match:
        path = match.group(1)
        if os.path.exists(path):
            return path
    return None

class RouterOutput(BaseModel):
    task: Literal["pdf", "image", "audio", "youtube", "code", "text"] = Field(
        description="Task type to route the user input"
    )

class IntentOutput(BaseModel):
    intent: Literal["summarize", "qa", "sentiment"]= Field(
        description="Task intent to route based on the user query"
    )

class AgentState(TypedDict):
    input: str
    extracted_text: Optional[str]
    file_path: Optional[str] 
    task: Optional[str] # from router_node to pdf, image, audio, youtube, text, code nodes
    intent: Optional[str] # from intent_router_node to summarizer, rag, sentiment nodes
    output: Optional[str]

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

router_parser = PydanticOutputParser(pydantic_object=RouterOutput)
intent_parser = PydanticOutputParser(pydantic_object=IntentOutput)

# router node to route the incoming text from user
def router_node(state: AgentState):
    print("\ninside router node")

    # file-based routing first
    if state.get("file_path"):
        path = state["file_path"].lower()
        if path.endswith(".pdf"):
            return {"task": "pdf"}
        if path.endswith((".png", ".jpg", ".jpeg")):
            return {"task": "image"}
        if path.endswith((".wav", ".mp3")):
            return {"task": "audio"}

    # url-based
    if "youtube.com" in state["input"] or "youtu.be" in state["input"]:
        return {"task": "youtube"}

    # fallback to text/code
    prompt = f"""
    Classify the input modality.

    Valid values: pdf, image, audio, youtube, code, text

    {router_parser.get_format_instructions()}

    Input:
    {state['input']}
    """
    parsed = router_parser.parse(llm.invoke(prompt).content)
    return {"task": parsed.task}

# pf node to process pdf
def pdf_node(state: AgentState):
    print("\ninside pdf node")
    # pdf_path = extract_file_path(state["input"])
    pdf_path = state.get("file_path")
    if not pdf_path:
        raise ValueError("pdf_node called without file_path")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text = "\n".join(d.page_content for d in docs)
    print("\n pdf text :" , text[:100])
    return {"extracted_text": text}

# image node to process image
def image_node(state: AgentState):
    print("\ninside image node")
    image_path = state.get("file_path")
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    print("\n image text :" , text[:100])
    return {"extracted_text": text}

# audio node to process audio
def audio_node(state: AgentState):
    print("\ninside audio node")
    audio_path = state.get("file_path")
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    # with sr.AudioFile(state["input"]) as source:
    #     audio = r.record(source)

    text = r.recognize_google(audio)
    print("\n audio text :" , text[:100])
    return {"extracted_text": text}

# youtube node to process youtube
def youtube_node(state: AgentState):
    print("\nFetching YouTube transcript for URL:", state["input"])
    video_id = re.search(r"v=([^&]+)", state["input"]).group(1)
    # transcript = YouTubeTranscriptApi.get_transcript(video_id)
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    # print("transcript:", transcript)
    text = " ".join(chunk.text for chunk in transcript)
    print("\nExtracted transcript text:", text[:100])  # Print first 500 characters
    return {"extracted_text": text}

# to process text 
def text_node(state: AgentState):
    print("\ninside text node")
    return {"extracted_text": state["input"]}

# to process code
def code_node(state: AgentState):
    print("\ninside code node")
    prompt = f"""
    Explain what code does, detect bugs, and mention time complexity.

    {state['input']}
    """
    explanation = llm.invoke(prompt).content
    print("explanation:", explanation)
    return {"output": explanation}

# to summarize the output of previous node
def summarizer_node(state: AgentState):
    print("\ninside summarizer node")
    prompt = f"""
    Summarize the following content clearly:
    Output must include:
    ● 1-line summary
    ● 3 bullets
    ● 5-sentence summary

    {state['extracted_text']}
    """

    summary = llm.invoke(prompt).content
    print("summary:", summary)
    return {"output": summary}

# to calculate sentiment
def sentiment_node(state: AgentState):
    print("\ninside sentiment node")
    prompt = f"""
    Analyze sentiment and give score (-1 to +1):
    output should contain : Label + confidence + one-line justification.

    {state['extracted_text']}
    """

    sentiment = llm.invoke(prompt).content
    print("sentiment:", sentiment)
    return {"output": sentiment}

# for rag implementation
def rag_node(state: AgentState):
    print("\ninside rag_node")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.create_documents([state["extracted_text"]])

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(state["input"])
    print(f"\nRetrieved {len(retrieved_docs)} documents from vector store.")
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {state['input']}
    """

    answer = llm.invoke(prompt).content
    return {"output": answer}

# to diverge the output according to the intent
def intent_router_node(state: AgentState):
    print("\ninside intent_router_node")
    prompt = f"""
    You are an intent classifier.

    User message:
    {state['input']}

    Decide what the user wants to do with the content.

    {intent_parser.get_format_instructions()}
    """

    parsed = intent_parser.parse(llm.invoke(prompt).content)
    print("\nparsed intent output:", parsed)
    return {"intent": parsed.intent}

def route_by_task(state: AgentState) -> str:
    print("\ninside route_by_task function")
    """
    Routes based on the task decided by the router node.
    """
    if not state.get("task"):
        raise ValueError("Router did not set 'task' in AgentState")

    return state["task"]


builder = StateGraph(AgentState)

# creating nodes
builder.add_node("router", router_node)
builder.add_node("pdf", pdf_node)
builder.add_node("image", image_node)
builder.add_node("audio", audio_node)
builder.add_node("youtube", youtube_node)
builder.add_node("text", text_node)
builder.add_node("code", code_node)
builder.add_node("summarizer", summarizer_node)
builder.add_node("sentiment", sentiment_node)
builder.add_node("intent_router", intent_router_node)
builder.add_node("rag", rag_node)

# creating edges
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_by_task,
    {
        "pdf": "pdf",
        "image": "image",
        "audio": "audio",
        "youtube": "youtube",
        "text": "text",
        "code": "code",
    }
)
 
for node in ["pdf", "image", "audio", "youtube", "text"]:
    builder.add_edge(node, "intent_router")

builder.add_conditional_edges(
    "intent_router",
    lambda state: state["intent"],
    {
        "summarize": "summarizer",
        "qa": "rag",
        "sentiment":"sentiment"
    }
)

builder.add_edge("summarizer", END)
builder.add_edge("rag", END)
builder.add_edge("sentiment", END)
builder.add_edge("code", END)
chatbot = builder.compile()

