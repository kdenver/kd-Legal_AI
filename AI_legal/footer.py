import streamlit as st

def layout(*args):
    # Custom CSS to hide the Streamlit footer and menu
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 40px; }
    </style>
    """
    
    st.markdown(style, unsafe_allow_html=True)

    # Add your custom content to the layout
    for arg in args:
        if isinstance(arg, str):
            st.markdown(arg)
        elif isinstance(arg, HtmlElement):
            st.markdown(str(arg), unsafe_allow_html=True)

def footer():
    # Add footer content
    myargs = [
        "Made with ❤️ by Nikhil, Mihir, Nilay",
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()
