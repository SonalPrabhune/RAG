import streamlit as st
import requests

# Define the Flask app URL
API_ENDPOINT = "http://127.0.0.1:5000"

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'is_loading' not in st.session_state:
    st.session_state['is_loading'] = False

# Sidebar configuration panel
st.sidebar.header("Configure Answer Generation")
prompt_template = st.sidebar.text_area("Override Prompt Template", "")
retrieve_count = st.sidebar.number_input("Retrieve this many documents from search:", min_value=1, max_value=50, value=3, step=1)
exclude_category = st.sidebar.text_input("Exclude Category", "")
suggest_followup = st.sidebar.checkbox("Suggest Follow-up Questions", False)

# Function to call the chat API
def make_api_request(question):
    st.session_state['is_loading'] = True
    try:
        history = [{"user": turn[0], "bot": turn[1]} for turn in st.session_state['chat_history']]
        request_payload = {
            "history": history + [{"user": question, "bot": None}],
            "retrievalstrategy": "crs",
            "overrides": {
                "prompt_template": prompt_template or None,
                "exclude_category": exclude_category or None,
                "top": retrieve_count,
                "suggest_followup_questions": suggest_followup
            }
        }
        response = requests.post(f"{API_ENDPOINT}/chat", json=request_payload)
        response.raise_for_status()
        answer = response.json()
        st.session_state['chat_history'].append((question, answer.get('answer', 'No response received')))
        if response.status_code == 200:
            result = answer.get('answer')
            st.write("Response:", result)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        st.session_state['is_loading'] = False

# Main Chat Interface
st.title("Chat with Your Data")

if st.button("Clear Chat"):
    st.session_state['chat_history'] = []

for turn in st.session_state['chat_history']:
    st.write(f"**You:** {turn[0]}")
    st.write(f"**Bot:** {turn[1]}")

new_question = st.text_input("Type a new question:")
if st.button("Send") and new_question:
    make_api_request(new_question)
    
