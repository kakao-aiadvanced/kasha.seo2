import streamlit as st
from pprint import pprint
from langgraph.graph import END, StateGraph
from langgraph_flow import retrieve, grade_documents, web_search, decide_to_generate, generate, hallucination_checker, decide_to_answer, GraphState

def print_log(value):
    print("답변:")
    pprint(value["generation"])
    print("출처:")
    source_set = set()
    for doc in value["documents"]:
        source_set.add((doc.metadata["title"], doc.metadata["source"]))

    for title, source in source_set:
        pprint("Document title: " + title)
        pprint("Document URL: " + source)

def run_console(app):
    inputs = {"question": "What is prompt?"}

    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")

    print_log(value)

def run_streamlit(app):
    st.set_page_config(
        page_title="Research Assistant",
        page_icon=":orange_heart:",
    )
    llm_model = st.sidebar.selectbox(
        "Select Model",
        options=[
            "llama3",
        ],
    )
    st.title("Research Assistant powered by OpenAI")

    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Superfast Llama 3 inference on Groq Cloud",
    )

    generate_report = st.button("Generate Report")

    if generate_report:
        with st.spinner("Generating Report"):
            inputs = {"question": input_topic}
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")

            print_log(value)

            final_report = value["generation"]
            st.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()


if __name__ == "__main__":
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("websearch", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_checker", hallucination_checker)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", "hallucination_checker")
    workflow.add_conditional_edges("hallucination_checker",
       decide_to_answer,
       {
           "answer": END,
           "generate": "generate",
       },
    )
    app = workflow.compile()

    # run_console(app)  # 콘솔로만 보고 싶으면 이것을 실행
    run_streamlit(app)  # streamlit으로 보고 싶으면 이것을 실행
