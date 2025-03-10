# from __future__ import annotations
import os
import logging
import datetime
import litellm
import json
import yaml
import datetime
from dotenv import load_dotenv

from typing import Annotated, List
from pydantic import BaseModel, Field
from pydantic_core import from_json
import argparse

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONTEXT_MAX_SPLIT_TOKENS=200
MAX_TOKENS_CONTEXT=500
MAX_TOKENS_QNA=250
MAX_CONTEXT_STRING_LENGTH=1000
MAX_LLM_TOKENS=2048

# Design Notes
# 
# 1. Does the answers for the questions have to come from the actual context in the file or can the context be a summarization of the info that's in the knowledge markdown files
# Every fact should be supported by the context, but the answers do not need to be verbatim.
# 
# 2. The docs say that "Each qna.yaml file needs at least three question and answer pairs per context chunk with a maximum token count of 250 tokens.". Is that 250 tokens per context or per question and answer pair?
# The 250 is an approximate number based on the maximum total size for SDG. The total tokens of Context + 3 Q&A must be less than 750 tokens. To have enough data for a context to answer the questions, an approximate 500 tokens are recommended for context, and the remaining 250 for the 3 Q&A.
# At the end, the Q&A length is no problem as long as the context+3 Q&As remain < 750
# 
# 3. Also from the docs, "Each qna.yaml needs five context blocks and has a maximum token count of 500 tokens." Is that per context or for all contexts?
# This is per context, and the recommended 500 is to ensure there is enough data in the context to answer the questions. It can be less or it can be more, as long as the final lenght of Context + 3 Q&A < 750 tokens.

class ContextContent(BaseModel):
    contexts:List[str]

class QuestionAndAnswer(BaseModel):
    question: str
    answer: str

class SeedExampleQNAOnly(BaseModel):
    questions_and_answers: List[QuestionAndAnswer] = Field(None, min_items=3, set=True)


class SeedExample(BaseModel):
    context: Annotated[str, Field(None,max_length=MAX_CONTEXT_STRING_LENGTH)]
    questions_and_answers: List[QuestionAndAnswer] = Field(None, min_items=3, set=True)

class QNAModel(BaseModel):
    version: Annotated[int,Field(3)]
    created_by: Annotated[str, Field(None)]
    domain: Annotated[str, Field(None)]
    seed_examples: Annotated[List[SeedExample], Field(None, min_items=5, set=True)]

def call_llm(context:str)->str:
    api_key=os.environ.get('API_KEY', None)
    model = os.environ.get('MODEL_NAME', None)
    llm_base = os.environ.get('LLM_URL', None)
    max_tokens = os.environ.get('MAX_LLM_TOKENS', 2048)    
    litellm.api_base=llm_base
    
    context_prompt=f"You are a helpful question and answer writing assistant. Given the following Information generate 1 SeedExample containing 3 question and answer pairs. Ensure that the questions can be answered by the information given. Do not number the pairs.  All output MUST be in valid JSON format.\n\nInformation: {context} \n\nOutput a valid JSON object but do not repeat the schema. This is the JSON schema that must be used: {SeedExampleQNAOnly.model_json_schema()}." 

    messages = [{ "content": context_prompt,"role": "user"}]
       
    response = litellm.completion(messages=messages, 
                                  model=model,
                                  api_key=api_key,
                                  max_tokens=max_tokens,
                                  response_format={ "type": "json_object"})
    extracted_json = response.choices[0].message.content
    tokens_used = response.model_extra["usage"].completion_tokens
    return extracted_json
 
def generate_examples(src_context:str)->SeedExample:   
    valid_output = False
    retry_count = 0
    fin=None

    while not valid_output and retry_count < 3:
        try:
            extracted_json=call_llm(src_context)
            res = SeedExampleQNAOnly.model_validate(from_json(extracted_json,allow_partial=True,cache_strings='keys'))
            fin = SeedExample(context=src_context,questions_and_answers=res.questions_and_answers)
            valid_output = True
        except (Exception) as e:
            _log.error(e,f"Invalid response,count {retry_count}")
            retry_count += 1       

    if fin.questions_and_answers is None:
        raise Exception("Invalid payload, no qna")
    
    return fin
    
def process_context_file(context_file:str,output_file_name:str):
    f=open(context_file,"r")
    context_list = f.read().split("\n\n\n")
    
    # TODO: Need to add error checking here
    
    qna_list=[]
    for cnt in context_list:
        response=generate_examples(cnt)
        qna_list.append(response)
        
    finalqna = QNAModel(version=3,created_by="ai",domain="CHANGE ME",seed_examples=qna_list)
    jsonout = finalqna.model_dump_json()
    
    yaml_string=yaml.dump(json.loads(jsonout))

    # ts =str(datetime.datetime.now().timestamp())


    with open(output_file_name, 'w') as file:
        file.write(yaml_string)   
    
if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig( level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("context_file", help = "Text file containing multiple context section. Each section is deliniated by two blank lines",type=str)
    parser.add_argument("output_file", help = "name of file to write output to",type=str)
    args = parser.parse_args()
    process_context_file(args.context_file,args.output_file)

