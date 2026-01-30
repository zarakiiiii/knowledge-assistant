import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Knowledge Assistant", layout="wide")

st.title("Knowledge Assistant")
st.write("Upload a PDF and ask questions grounded in its content.")

#---PDF Upload -----
st.header("Upload Doc")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}  #file name, file object, MIME type eg;pdf/jpeg,etc.
    with st.spinner("Uploading and processing PDF...."):
        res = requests.post(f"{API_URL}/upload",files=files)

    if res.status_code == 200:
        st.success("PDF uploaded and indexed succesfully")
    else:
        st.error("Failed to upload PDF")

st.divider()


#---Query Section ---
st.header("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Ask") and question:
    payload = {"question": question}
    with st.spinner("Searching knowledge base..."):
        res = requests.post(f"{API_URL}/query", json=payload)

    if res.status_code != 200:
        st.error("Error querying backed")
    else:
        data = res.json()

        st.subheader("Answer")
        st.write(data.get("answer", ""))

        sources = data.get("sources",[])
        if sources:
            st.subheader("Sources")
            for i,src in enumerate(sources, 1):  #expandable UI
                with st.expander(f"Source {i} (distance={src['distance']:.3f})"):
                    st.write(src["text"])
        else:
            st.info("No sources returned")

