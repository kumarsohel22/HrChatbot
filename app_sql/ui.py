# import streamlit as st
# import requests

# # Streamlit page setup
# st.set_page_config(page_title="Employee Q&A", page_icon="🤖", layout="centered")

# st.title("Employee Q&A with FastAPI")

# # Input box for the question
# question = st.text_input("Ask a question:", "name the employees who worked in healthcare?")

# # Button to send request
# if st.button("Get Answer"):
#     if question.strip():
#         try:
#             # API endpoint
#             url = "http://127.0.0.1:8000/ask"
#             headers = {"Content-Type": "application/json"}
#             payload = {"question": question}

#             # POST request
#             response = requests.post(url, json=payload, headers=headers)

#             # Display the response
#             if response.status_code == 200:
#                 st.success("Answer:")
#                 st.write(response.json())
#             else:
#                 st.error(f"Error {response.status_code}: {response.text}")
#         except Exception as e:
#             st.error(f"Request failed: {e}")
#     else:
#         st.warning("Please enter a question before submitting.")


import streamlit as st
import requests

# Streamlit page setup
st.set_page_config(page_title="Employee Q&A", page_icon="🤖", layout="centered")

st.title("Employee Q&A with FastAPI")

# Input box for the question
question = st.text_input("Ask a question:", "name the employees who worked in healthcare?")

# Button to send request
if st.button("Get Answer"):
    if question.strip():
        try:
            # API endpoint
            url = "http://127.0.0.1:8000/ask"
            headers = {"Content-Type": "application/json"}
            payload = {"question": question}

            # POST request
            response = requests.post(url, json=payload, headers=headers)

            # Display the response
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found")
                query = data.get("query", "No query available")
                result = data.get("result", "No result available")

                # Show each in its own section
                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### SQL Query")
                st.code(query, language="sql")

                st.markdown("### Result")
                if isinstance(result, list):
                    # Nicely display table-like results if it's a list of rows
                    st.table(result)
                else:
                    st.write(result)

            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter a question before submitting.")


