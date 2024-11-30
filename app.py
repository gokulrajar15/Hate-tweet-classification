import streamlit as st
from main import get_result

def app():
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    st.title("Tweet Classification")
    
    user_input = st.text_input(
        "Enter a Tweet for Classification:", 
        value=st.session_state.user_input, 
        key="input_box"
    )
    st.session_state.user_input = user_input

    st.subheader("Examples:")
    
    example1 = "i am amazing. #i_am #positive #affirmation"
    example2 = "Report: Google A.I. Is ‘Biased’ Against Gay People, Jews  http://www.breitbart.com/tech/2017/10/26/report-go..."
    example3 = "I Hate #$%#$%Jewish%$#@%^^@#"
    example4 = "i get to see my daddy today!! #80days #gettingfed"

    col1, col2 = st.columns(2)
    with col1:
        if st.button(example1[:40] + "..."):
            st.session_state.user_input = example1
    with col2:
        if st.button(example2[:40] + "..."):
            st.session_state.user_input = example2

    st.write("\n") 
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(example3[:40] + "..."):
            st.session_state.user_input = example3
    with col2:
        if st.button(example4[:40] + "..."):
            st.session_state.user_input = example4

    st.write("\n") 
    submit_button = st.button("Submit")
    
    if submit_button:
        if user_input.strip():
            result = get_result(user_input)
            st.write(f"**Predicted Data:** {result}") 
            st.markdown(f'<p style="font-size: 20px; color: #FF6347; font-weight: bold;">Predicted Data: {result}</p>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to classify.")
    
if __name__ == "__main__":
    app()
