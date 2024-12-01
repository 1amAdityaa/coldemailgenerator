import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
    def extract_jobs(self, cleaned_text):
    # Define a more detailed prompt template
      prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The following text has been scraped from a website's careers page. 
        Your task is to identify each job posting and organize it into JSON format with these fields:
        
        - `role`: Job title or position.
        - `experience`: Required years or type of experience.
        - `skills`: Key skills or qualifications needed.
        - `description`: Brief summary of job responsibilities and expectations.
        
        Only return valid JSON, without any additional text or explanations. If a specific field is missing for a job posting, exclude that field from the JSON for that entry.

        ### VALID JSON (NO PREAMBLE):
        """
       )
      chain_extract = prompt_extract|self.llm
      res = chain_extract.invoke(input={"page_data": cleaned_text})
      try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
      except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
      return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are [name], [ROLE] from chennai. 
            
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of me
            in fulfilling their needs.
            Also add my portfolio: {link_list}
            Remember you are [name], from [place]. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))