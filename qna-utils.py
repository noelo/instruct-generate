# from __future__ import annotations
import os
import logging
import datetime
from typing import Annotated, List
import json
import argparse
import yaml
import requests
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_core import from_json


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

def llm_generate_examples(context:str)->str:
    """Call the llm to generate seed examples"""
    api_key=os.environ.get('GENERATION_API_KEY', None)
    model = os.environ.get('GENERATION_MODEL_NAME', None)
    llm_base = os.environ.get('GENERATION_LLM_URL', None)
    max_tokens = os.environ.get('GENERATION.MAX_LLM_TOKENS', 2048)     
    litellm.api_base=llm_base
    
    context_prompt=f"You are a helpful question and answer writing assistant. Given the following Information generate 1 SeedExample containing 3 question and answer pairs. Ensure that the questions can be answered by the information given. Do not number the pairs.  All output MUST be in valid JSON format.\n\nInformation: {context} \n\nOutput a valid JSON object but do not repeat the schema. This is the JSON schema that must be used: {SeedExampleQNAOnly.model_json_schema()}." 

    messages = [{ "content": context_prompt,"role": "user"}]
    response = litellm.completion(messages=messages,
                                  model=model,
                                  api_key=api_key,
                                  max_tokens=max_tokens,
                                  response_format={ "type": "json_object"})
    extracted_json = response.choices[0].message.content
    
    # to be used at a later stage
    # tokens_used = response.model_extra["usage"].completion_tokens
    return extracted_json

def llm_count_tokens(context:str)->str:
    """Given a string count the number of tokens in it"""
    api_key=os.environ.get('TOKENISATION_API_KEY', None)
    model = os.environ.get('TOKENISATION_MODEL_NAME', None)
    llm_base = os.environ.get('TOKENISATION_LLM_URL', None)

    payload = {"model": f"{model}","prompt": f"{context}","add_special_tokens": "false"}
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(llm_base, headers=headers,json=payload,timeout=120)
    return response.json()
 
def generate_examples(src_context:str)->SeedExample:
    """Generate a list of seed examples given a context"""
    valid_output = False
    retry_count = 0
    fin=None

    while not valid_output and retry_count < 3:
        try:
            extracted_json=llm_generate_examples(src_context)
            res = SeedExampleQNAOnly.model_validate(from_json(extracted_json,allow_partial=True,cache_strings='keys'))
            fin = SeedExample(context=src_context,questions_and_answers=res.questions_and_answers)
            valid_output = True
        except (Exception) as e:
            _log.error(e,f"Invalid response,count {retry_count}")
            retry_count += 1

    if fin.questions_and_answers is None:
        raise Exception("Invalid payload, no qna")
    
    return fin
    
def process_context_file(context_file:str,output_file_name:str) -> None:
    """Open and read the context files and generate examples"""

    f=open(context_file,"r")
    context_list = f.read().split("\n\n\n")
    #TODO: Need to add error checking here

    qna_list=[]
    for cnt in context_list:
        _log.info(f'Context-> {cnt}')
        response=generate_examples(cnt)
        _log.info(f'    Example-> {response}')
        qna_list.append(response)

    finalqna = QNAModel(version=3,created_by="ai",domain="CHANGE ME",seed_examples=qna_list)
    jsonout = finalqna.model_dump_json()
    yaml_string=yaml.dump(json.loads(jsonout))

    with open(output_file_name, 'w') as file:
        file.write(yaml_string)   

def validate_qna(qna_file:str) -> None:
    """Count the tokens in the passed qna file"""

    with open(qna_file, 'r') as file:
        yaml_data = yaml.full_load(file)

    for idx,example in enumerate(yaml_data.get('seed_examples')):
        example_token_count=0
        context_tokens = llm_count_tokens(example['context'])
        example_token_count= context_tokens["count"]
        print(f"Context {example_token_count}")
        
        for idy,qna in enumerate(example['questions_and_answers']):
            qna_tokens_answer = llm_count_tokens(qna['answer'])
            answer_tc = qna_tokens_answer["count"]
            example_token_count = example_token_count + answer_tc
            print(f"Answer {idy+1}: {answer_tc}")
            qna_tokens_question = llm_count_tokens(qna['question'])
            quest_tc = qna_tokens_question["count"]
            example_token_count = example_token_count + quest_tc
            print(f"Question {idy+1}: {quest_tc}")
            
        print(f"Total token count for Example {idx+1}: {example_token_count}")

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig( level=logging.INFO)
    parser = argparse.ArgumentParser()
    time_stamp =str(datetime.datetime.now().timestamp())
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("-v", "--validate", help = "validate the passed file",action="store_true")
    group1.add_argument("-g", "--generate", help = "Generate the samples from the passed context file",action="store_true")

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument("-q","--qna-file", help = "QNA.yaml file to be validated",type=str)
    group2.add_argument("-c","--context", help = "Text file containing multiple context section. Each section is deliniated by two blank lines",type=str)

    parser.add_argument("-o","--output", help = "name of file to write the generated qna text to",default="qna.yaml"+time_stamp,type=str)

    args = parser.parse_args()
    if args.generate:
        process_context_file(args.context,args.output)
    else:
        validate_qna(args.qna_file)