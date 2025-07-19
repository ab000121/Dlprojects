
import streamlit as st
from crewai import Agent, Task, Crew, LLM
import os



st.set_page_config(page_title="Article Generator", layout="centered")

st.title("AI Article Generator using CrewAI")
st.markdown("Create structured blog articles powered by Gemini and CrewAI agents.")

# Gemini API key input
api_key ="your_gemini_api_key_here"  # Replace with your actual API key

topic = st.text_input(" Enter your topic (e.g., Climate Change, AI in Healthcare)")

generate = st.button(" Generate Article")

if generate and api_key and topic:
    os.environ["GEMINI_API_KEY"] =api_key
    gen_llm = LLM(model="gemini/gemini-2.0-flash-exp", api_key=api_key)

    with st.spinner("Planning your article..."):
        #agent 1 
        planner = Agent(
            role="Content Planner",
            goal=f"Plan engaging and factually accurate content of 500 words on {topic}",
            backstory="You're planning a blog article to help readers learn and make informed decisions.",
            llm=gen_llm,
            allow_delegation=False,
            verbose=True,
        )

        #agent 2
        writer = Agent(
            role="Content Writer",
            goal="Write clear and engaging article of 500 words",
            backstory="You write compelling articles from structured plans provided by a planner.",
            llm=gen_llm,
            allow_delegation=False,
            verbose=True,
        )

        task1 = Task(
            description=f"Create a well-structured article plan for the topic: {topic}",
            agent=planner,
            expected_output="A well-written and factual articleof 500 words on {topic}"
        )

        task2 = Task(
            description="Write a detailed blog post using the planner's content plan",
            agent=writer,
            expected_output="A well-written  and factual article of 500 words on {topic}",
            context=[task1]
        )

        crew = Crew(agents=[planner, writer], tasks=[task1, task2], verbose=True)
        result = crew.kickoff(inputs={"topic":topic})

    st.success(" Article generated successfully!")
    st.markdown(" Final Article Output:")
    st.markdown(result)
else:
    if generate:
        st.error("Please enter both the API key ")
