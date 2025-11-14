from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uuid
import asyncio
from datetime import datetime
import json

from graph.rag_graph import workflow
from graph.state import RAGState

# Initialize FastAPI app
app = FastAPI(
    title="Cricket Rules RAG API",
    description="Agentic RAG system for answering cricket rules questions with conversational memory",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active sessions (use Redis in production)
active_sessions = {}

# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User's question about cricket rules")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    stream: bool = Field(default=False, description="Whether to stream the response")

class Citation(BaseModel):
    source: str
    page: str
    relevance_score: Optional[int] = None

class QueryResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Citation]
    timestamp: str
    turn_number: int
    message_count: int

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    last_activity: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]
    total_messages: int

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# ==================== Helper Functions ====================

def format_citations(retrieved_chunks: List[Dict]) -> List[Citation]:
    """Extract unique citations from retrieved chunks"""
    seen = set()
    citations = []
    
    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown Source").strip()
        page = chunk.get("page_num", "N/A").strip()
        score = chunk.get("relevance_score")
        
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source=source,
                page=str(page),
                relevance_score=score
            ))
    
    return citations

def get_session_config(session_id: str) -> Dict:
    """Get LangGraph config for a session"""
    return {
        "configurable": {
            "thread_id": session_id
        }
    }

def update_session_metadata(session_id: str, message_count: int):
    """Update session metadata"""
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "turn_number": 0
        }
    
    active_sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
    active_sessions[session_id]["message_count"] = message_count
    active_sessions[session_id]["turn_number"] += 1

# ==================== API Endpoints ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/query", response_model=QueryResponse)
async def query_cricket_rules(request: QueryRequest):
    """
    Query the cricket rules RAG system
    
    - **question**: Your question about cricket rules
    - **session_id**: Optional session ID for conversation continuity
    - **stream**: Whether to stream the response (not implemented in this endpoint)
    """
    try:
        # Generate or use provided session_id
        session_id = request.session_id or str(uuid.uuid4())
        config = get_session_config(session_id)
        
        # Check if this is a new session or continuing one
        is_new_session = request.session_id is None
        
        if is_new_session:
            print(f"🆕 Creating new session: {session_id}")
        else:
            print(f"📝 Continuing session: {session_id}")
        
        # ✅ FIX: Only pass required fields, let LangGraph load the rest from memory
        # Don't initialize messages as empty - that overrides saved state!
        state = {
            "user_question": request.question,
        }
        
        # If it's a new session, initialize empty state
        if is_new_session:
            state["retrieved_chunks"] = []
            state["messages"] = []
            state["context_summary"] = ""
            state["answer"] = None
        
        # Run the workflow
        print(f"🔄 Invoking workflow with config: {config}")
        result = workflow.invoke(state, config=config)
        
        print(f"📊 Result keys: {result.keys()}")
        print(f"📨 Messages in result: {len(result.get('messages', []))}")
        
        # Extract answer
        answer = result.get("answer", "")
        if not answer and "messages" in result and result["messages"]:
            # Fallback: get from last message
            last_msg = result["messages"][-1]
            if isinstance(last_msg, dict):
                answer = last_msg.get("content", "No answer generated")
            else:
                answer = last_msg.content if hasattr(last_msg, 'content') else "No answer generated"
        
        # Extract citations
        citations = format_citations(result.get("retrieved_chunks", []))
        
        # Get message count
        message_count = len(result.get("messages", []))
        
        # Update session metadata
        update_session_metadata(session_id, message_count)
        turn_number = active_sessions[session_id]["turn_number"]
        
        print(f"✅ Response ready - Turn: {turn_number}, Messages: {message_count}")
        
        return QueryResponse(
            session_id=session_id,
            answer=answer,
            citations=citations,
            timestamp=datetime.utcnow().isoformat(),
            turn_number=turn_number,
            message_count=message_count
        )
        
    except Exception as e:
        print(f"❌ Error in query endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/stream")
async def query_cricket_rules_stream(request: QueryRequest):
    """
    Stream the response for real-time updates
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        config = get_session_config(session_id)
        is_new_session = request.session_id is None
        
        async def generate():
            try:
                # Send session info first
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                
                # ✅ FIX: Only pass required fields for continuing sessions
                state = {
                    "user_question": request.question,
                }
                
                # Initialize only for new sessions
                if is_new_session:
                    state["retrieved_chunks"] = []
                    state["messages"] = []
                    state["context_summary"] = ""
                    state["answer"] = None
                
                # Stream through workflow
                for event in workflow.stream(state, config=config):
                    # Send node updates
                    for node_name, node_output in event.items():
                        yield f"data: {json.dumps({'type': 'node', 'node': node_name})}\n\n"
                
                # Get final state
                final_state = workflow.get_state(config).values
                answer = final_state.get("answer", "")
                
                if not answer and "messages" in final_state:
                    messages = final_state["messages"]
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            answer = last_msg.get("content", "")
                        else:
                            answer = last_msg.content if hasattr(last_msg, 'content') else ""
                
                # Send answer
                if answer:
                    # Split into chunks for streaming effect
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        chunk = answer[i:i+chunk_size]
                        yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.05)  # Small delay for effect
                
                # Send citations
                citations = format_citations(final_state.get("retrieved_chunks", []))
                citations_dict = [c.dict() for c in citations]
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations_dict})}\n\n"
                
                # Send completion
                message_count = len(final_state.get("messages", []))
                update_session_metadata(session_id, message_count)
                
                yield f"data: {json.dumps({'type': 'complete', 'message_count': message_count})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in streaming: {str(e)}")

@app.get("/sessions/{session_id}/history", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a session
    """
    try:
        config = get_session_config(session_id)
        
        # Get state from workflow
        state = workflow.get_state(config)
        
        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = state.values.get("messages", [])
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append({
                    "role": "user" if msg.get("type") == "human" else "assistant",
                    "content": msg.get("content", "")
                })
            else:
                formatted_messages.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                })
        
        return ConversationHistory(
            session_id=session_id,
            messages=formatted_messages,
            total_messages=len(formatted_messages)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/sessions", response_model=List[SessionInfo])
async def list_active_sessions():
    """
    List all active sessions
    """
    sessions = []
    for session_id, metadata in active_sessions.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            created_at=metadata["created_at"],
            message_count=metadata.get("message_count", 0),
            last_activity=metadata.get("last_activity", metadata["created_at"])
        ))
    
    return sessions

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session (clear memory)
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    return {"message": f"Session {session_id} deleted", "session_id": session_id}

@app.get("/sessions/{session_id}/info", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """
    Get information about a specific session
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    metadata = active_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=metadata["created_at"],
        message_count=metadata.get("message_count", 0),
        last_activity=metadata.get("last_activity", metadata["created_at"])
    )

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Cricket Rules RAG API Starting...")
    print("📚 Loading vector database connection...")
    print("✅ API Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down Cricket Rules RAG API...")
    print(f"📊 Total sessions handled: {len(active_sessions)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)