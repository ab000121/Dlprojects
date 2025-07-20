
import streamlit as st
from crewai import Agent, Task, Crew, LLM
import os

st.set_page_config(page_title="Email Generator", layout="centered")

st.title("AI Email Generator using CrewAI")
st.markdown("Create structured email content powered by Gemini and CrewAI agents.")

# Gemini API key input
api_key ="your api key"# Replace with your actual API key

st.write("Fill out the form and let our AI write a professional email for you.")

# User input
with st.form("email_form"):
    purpose = st.text_input("Purpose of the email", "job offer acceptance")
    tone = st.selectbox("Tone of the email", ["Formal", "Friendly", "Persuasive", "Apologetic", "Thankful"])
    recipient = st.text_input("Recipient", "HR Manager")
    key_points = st.text_area("Key Points (one per line)", "Thank them for the opportunity\nExpress interest in the role\nMention availability")
    submitted = st.form_submit_button("Generate Email")

input_text = {
    "Purpose": purpose,
    "Tone": tone,
    "Recipient": recipient,
    "Key Points": key_points.replace('\n', ', ')
}


if submitted and api_key:
        os.environ["GEMINI_API_KEY"] =api_key
        gen_llm = LLM(model="gemini/gemini-2.0-flash-exp", api_key=api_key)
     
        with st.spinner("Planning your email..."):
                # Agent 1: Email Input Agent
                email_planner_agent = Agent(
                                role='Email Planner',
                                goal='Collect purpose, tone, recipient,key points and pla nan email according to it.',
                                backstory='You help user to plan a format for email. ',
                                verbose=True,
                                llm=gen_llm
                        )

        
                # Agent 2: Email Writer Agent
                email_writer_agent = Agent(
                                role='Professional Email Writer',
                                goal='Generate a clean, professional email based on the input',
                                backstory='You are an expert in writing well-structured and grammatically correct emails.',
                                verbose=True,
                                llm=gen_llm
                        )


                # Define tasks for each agent
                # 1. Email Input Task
                planning_task = Task(
                                description=f"Plan an email for the user according to {input_text}",
                                expected_output="You're planning a detailed structure of email.",
                                agent=email_planner_agent 
                        )

                # 2. Email Writing Task
                email_writing_task = Task(
                                description=f"""Write a professional email using the following input:
                                        {input_text}""",
                                expected_output="A full-length email ready to send",
                                agent=email_writer_agent,
                                context=[planning_task]
                        )
        with st.spinner("Generating email content..."):
                crew = Crew(
                        agents=[email_planner_agent,email_writer_agent],
                        tasks=[planning_task,email_writing_task],
                        verbose=True
                )
                result = crew.kickoff(inputs={"input": input_text})
                

        st.success("Email Generated Sucessfully")
        st.markdown(" Final  Output:")
        st.markdown(result)
else:
    if submitted:
        st.error("Please enter  the API key ")
