import streamlit as st
import json
import os
from openai import OpenAI
import requests

# Initialize OpenAI client using the environment variable
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Fetch and print all accessible model IDs for your account
#print("Models available to your API key:")
#for model in client.models.list().data:
#    print(f"- {model.id}")

#- meta-llama/llama-prompt-guard-2-86m
#- groq/compound
#- meta-llama/llama-prompt-guard-2-22m
#- canopylabs/orpheus-arabic-saudi
#- whisper-large-v3
#- openai/gpt-oss-120b
#- whisper-large-v3-turbo
#- openai/gpt-oss-20b
#- groq/compound-mini
#- canopylabs/orpheus-v1-english
#- qwen/qwen3.8-27b
#- openai/gpt-oss-safeguard-20b
#- allam-2-7b

st.set_page_config(page_title="Vi4 Academy AI Agent", layout="centered")
st.title("🏡 Vi4 Academy: Real Estate AI Agent")
st.markdown("Conversational AI powered by Function Calling & ML integration.")

# ------------------------------------------------------------------------------
# 1. TEMPORARY MOCK FUNCTION
# (During the live lecture, this will be replaced with an HTTP request to FastAPI)
# ------------------------------------------------------------------------------
def get_house_price_estimate(sqft, bedrooms, bathrooms, age_of_house, location_score):
    """
    Calls the local FastAPI Machine Learning endpoint.
    """
    url = "http://127.0.0.1:8000/predict-price"

    payload = {
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_of_house": age_of_house,
        "location_score": location_score
    }
    
    # Send the data to the ML model and return the JSON response to the AI Agent
    response = requests.post(url, json=payload)
    return response.json()

# ------------------------------------------------------------------------------
# 2. OPENAI TOOL DEFINITION (JSON SCHEMA)
# ------------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_house_price_estimate",
            "description": "Estimate a property price based on house features. Call this when a user requests a valuation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sqft": {"type": "integer", "description": "Total interior area in square feet"},
                    "bedrooms": {"type": "integer", "description": "Total number of bedrooms"},
                    "bathrooms": {"type": "integer", "description": "Total number of bathrooms"},
                    "age_of_house": {"type": "integer", "description": "Age of the house in years"},
                    "location_score": {"type": "integer", "description": "Location rating scale from 1 (poor) to 10 (prime)"}
                },
                "required": ["sqft", "bedrooms", "bathrooms", "age_of_house", "location_score"]
            }
        }
    }
]

# ------------------------------------------------------------------------------
# 3. SESSION STATE & SYSTEM PROMPT
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert Real Estate & Financial Advisor AI. "
                "When users ask for a price estimate, ensure you have all required details: "
                "sqft, bedrooms, bathrooms, age_of_house, and location_score (1-10). "
                "If any detail is missing, ask follow-up questions politely. "
                "Once all parameters are gathered, execute the get_house_price_estimate tool "
                "and provide strategic financial advice based on the calculated valuation."
            )
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] not in ["system", "tool"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))

# ------------------------------------------------------------------------------
# 4. CHAT INTERACTION & REASONING LOOP
# ------------------------------------------------------------------------------
if prompt := st.chat_input("Ask about home valuations or financial advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initial API call to evaluate user prompt and check for function calls
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b", # <--- Changed from gpt-4o
        messages=st.session_state.messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_msg = response.choices[0].message
    
    # Store assistant payload
    msg_dict = {"role": "assistant", "content": response_msg.content}
    if response_msg.tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": t.id,
                "type": "function",
                "function": {"name": t.function.name, "arguments": t.function.arguments}
            } for t in response_msg.tool_calls
        ]
    st.session_state.messages.append(msg_dict)

    # Execute Tool logic if triggered by LLM
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "get_house_price_estimate":
                args = json.loads(tool_call.function.arguments)
                
                # Call local mock function
                result = get_house_price_estimate(
                    sqft=args.get("sqft"),
                    bedrooms=args.get("bedrooms"),
                    bathrooms=args.get("bathrooms"),
                    age_of_house=args.get("age_of_house"),
                    location_score=args.get("location_score")
                )
                
                # Append tool output to chat context
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "get_house_price_estimate",
                    "content": json.dumps(result)
                })
        for msg in st.session_state.messages:
             print(msg)
        # Second API call to synthesize natural language advice using tool data
        final_response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=st.session_state.messages
        )
        final_content = final_response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": final_content})
        
        with st.chat_message("assistant"):
            st.markdown(final_content)
            
    elif response_msg.content:
        with st.chat_message("assistant"):
            st.markdown(response_msg.content)