import streamlit as st
import random
import time
import requests

st.title("Codebase Chatbot")

# Initialize chat history for each codebase (as a dictionary)
if "messages" not in st.session_state:
    st.session_state.messages = {}

# Fetch namespaces and display them as buttons in the sidebar
def fetch_namespaces():
    url = "http://127.0.0.1:8000/get_namespaces"  # Endpoint para obtener namespaces
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("namespaces", [])
    except requests.exceptions.RequestException as e:
        return f"Error llamando al API de namespaces: {e}"



# Send query to the API with the selected codebase
def query_api(query, codebase):
    url = "http://127.0.0.1:8000/perform_rag"  # Cambia si tu API tiene otra URL
    payload = {"query": query, "codebase": codebase}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("response", "No response from API")
    except requests.exceptions.RequestException as e:
        return f"Error llamando a la API: {e}"

# Fetch namespaces and display them in the sidebar
namespaces = fetch_namespaces()

if isinstance(namespaces, list):  # Ensure it's a list
    st.sidebar.title("Codebases")

    
    for ns in namespaces:
        last_item = ns.split('/')[-1]
        
        # Create a button for each namespace
        if st.sidebar.button(f"- {last_item}", help=f"Select {last_item}", use_container_width=True):

            st.session_state.selected_codebase = ns
            st.session_state.selected_codebase_name = last_item

    
    namespace = st.sidebar.text_input(
        "Add a new codebase",
        help="Provide a public GitHub repository link to enable AI-driven code analysis and questions.",
    placeholder="e.g., https://github.com/username/repository"
    )
    
    if st.sidebar.button("+ Add codebase", key="add_codebase_button", type="secondary", icon="➕"):
        # URL
        url = "http://127.0.0.1:8000/create_namespace"

        # Payload con el namespace seleccionado
        payload = {
            "path": namespace
        }

        # Enviar la solicitud POST
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)

            # Manejar la respuesta de la API
            if response.status_code == 200:
                st.sidebar.success(f"Namespace '{namespace}' created successfully!")
                st.rerun()
            else:
                st.sidebar.error(f"Error creating namespace: {response.text}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error calling the API: {e}")
            

# Check if a codebase has been selected
if "selected_codebase" in st.session_state:
    codebase = st.session_state.selected_codebase
    codebase_name = st.session_state.selected_codebase_name

    # Initialize messages for the selected codebase if not already present
    if codebase not in st.session_state.messages:
        st.session_state.messages[codebase] = []

    # Display chat messages for the selected codebase
    for message in st.session_state.messages[codebase]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input for the selected codebase
    if prompt := st.chat_input(f"Ask something about {codebase_name} codebase!"):
        # Display user message in the chat message container
        st.session_state.messages[codebase].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")  # Show temporary message

            # Simulate a response time while calling the API
            time.sleep(1)
            response = query_api(prompt, codebase)
            response_placeholder.markdown(response)
        
        # Add assistant's response to chat history for the selected codebase
        st.session_state.messages[codebase].append({"role": "assistant", "content": response})

else:
    st.warning("Please select a codebase from the sidebar.")
