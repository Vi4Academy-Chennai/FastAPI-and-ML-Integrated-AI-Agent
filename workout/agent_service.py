from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import os

app = FastAPI(title="Real Estate Agent-as-a-Service")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# In-memory dictionary for session state (Use Redis in production)
sessions = {}

# --- Data Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# --- Agent Tool Schema ---
tools = [{
    "type": "function",
    "function": {
        "name": "get_house_price",
        "description": "Estimate a property price based on house features.",
        "parameters": {
            "type": "object",
            "properties": {
                "sqft": {"type": "integer"},
                "bedrooms": {"type": "integer"}
            },
            "required": ["sqft", "bedrooms"]
        }
    }
}]

def get_house_price(sqft: int, bedrooms: int) -> dict:
    # In a fully integrated system, this calls your actual ML inference endpoint
    return {"estimated_price": (sqft * 150) + (bedrooms * 10000)}

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # 1. Load or initialize session memory based on session_id
    if req.session_id not in sessions:
        sessions[req.session_id] = [
            {"role": "system", "content": "You are a Real Estate API Agent. Extract sqft and bedrooms, call the tool, and provide financial advice."}
        ]
    
    messages = sessions[req.session_id]
    messages.append({"role": "user", "content": req.message})

    # 2. Agent Reasoning (First API Call)
    response = client.chat.completions.create(
        model="gpt-4o", # Replace with qwen/qwen3.8-27b if using Groq
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    ai_msg = response.choices[0].message
    
    # Store the LLM's response (either text or a tool execution request)
    msg_dict = {"role": "assistant", "content": ai_msg.content}
    if ai_msg.tool_calls:
        msg_dict["tool_calls"] = [
            {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
            for t in ai_msg.tool_calls
        ]
    messages.append(msg_dict)

    # 3. Tool Execution & Synthesis
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call.function.name == "get_house_price":
                args = json.loads(tool_call.function.arguments)
                result = get_house_price(args.get("sqft"), args.get("bedrooms"))
                
                # Feed the raw tool data back into the Agent's memory
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result)
                })
        
        # Second API call to generate natural language from the tool data
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        final_text = final_response.choices[0].message.content
        messages.append({"role": "assistant", "content": final_text})
        
        return ChatResponse(reply=final_text, session_id=req.session_id)

    # Return standard text response if no tools were needed
    return ChatResponse(reply=ai_msg.content, session_id=req.session_id)