# Function to extract knowledge graphs from paper/ abstract given via input
# 
# Written 2024 by Joshua Sammet, Research assistant at the Chair of medical Knowledge and Decision, University of St. Gallen
#

import torch
import transformers
import argparse
import logging
import json
import os

# Needed for OpenAlex
import requests

"""
Function to reconstruct the inverted index storing of the abstracts
INPUTS:
inverted_text: text in invered index storing

RETURN:
reconstructed_text : string of reconstructed text from input
"""
def reconstruct_text(inverted_index):
    word_index = [] 
    for k,v in inverted_index.items(): 
        for index in v: 
            word_index.append([k,index]) 
    word_index = sorted(word_index,key = lambda x : x[1])
    word_list = []
    for i in range(len(word_index)):
        word_list.append(word_index[i][0]) 
    separator = ' ' 
    reconstructed_text = separator.join(word_list)
    return reconstructed_text

"""
Function to search OpenAlex database
INPUTS:
search_phrase: string of search question sent to OpenAlex
result_count: Number of results returned fom OpenAlex
min_year: Earliest publication date considered in search

RETURN:
res_json["results"]: List of dicts, containing one results per element
abstract_list: List of string that contain the respective abstract for each result
"""
def search_openalex(search_phrase, result_count=10, min_year='2013'):
    # Api touch point
    base_url = "https://api.openalex.org/works"
    # Filters: require abstract and fulltext and earliest day of publish
    filters = [
        "has_abstract:true",
        "has_fulltext:true",
        f"from_publication_date:{min_year}-01-01"
    ]
    # Construct the query parameters
    params = {
    "search": search_phrase,
    "filter": str.join(",", filters),  # Only return works with abstracts
    "per_page": result_count,  # Limit the search
    }
    r = requests.get(base_url, params=params)
    res_json = r.json()
    # Create abstracts
    abstract_list = []
    for i in range(len(res_json["results"])):
        abstract_list.append(reconstruct_text(res_json["results"][i]['abstract_inverted_index']))
        
    return res_json["results"], abstract_list

#--------------Prompts--------------
# These are the three prompts used by the 3 different analysers
PICO_message_abstract="""You are an expert agent specialized in extracting PICO elements on abstracts from scientific publications. 
The PICO elements are population, intervention, comparison and outcome. Your task is to identify the entities and 
relations requested from an abstract of an scientific paper that is given to you in a prompt.
You must generate the output in a JSON containing a list with JSON objects having the following keys: 
"head", "head_type", "relation", "tail", and "tail_type".
The "head" key must contain the text of the extracted entity from the provided user prompt, 
the "head_type" key must contain the type of the extracted head entity which must be one of the PICO elements, the "relation" key must contain the type of relation 
between the "head" and the "tail", the "tail" key must represent the text of an extracted entity which is the tail
of the relation, and the "tail_type" key must contain the type of the tail entity. Attempt to extract around 10 entities and relations.
"""
PICO_bulletpoints_abstract="""You are an expert agent specialized in extracting PICO elements on abstracts from scientific publications. 
The PICO elements are population, intervention, comparison and outcome. Your task is to identify and extract the content from an abstract of an scientific paper that is given to you in a prompt.
You must generate the output in the form of a list of bullet points. The content of each bullet point should summarize an aspect of the abstract with regard to at least one PICO element. Please mention the relevant PICO elements at the beginning of each bullet point.
"""
PICO_stepwise="""You are an expert agent specialized in extracting PICO elements on abstracts from scientific publications. 
The PICO elements are population, intervention, comparison and outcome. Your task is to identify and extract the content from an abstract of an scientific paper that is given to you in a prompt.
Summarize the content of the abstract with regard to at the PICO element.
"""

"""
Function to find publications of interest and analyse them
INPUT:
model_name - name of specified model that should be used
search_phrase - Topic that should be search for (e.g. 'depression treatment random controlled trial'
number_of_abstracts - How many papers should be checked
prompt - Specify which prompt should be used to instruct the model

OUTPUT:
returns json file with knowledge graph for each apper
"""
# Version 1: Giving the system message once
def analyse_initial_sys_message(model_name, search_phrase, number_of_abstracts=10, prompt=None):
    # Get papers from OpenAlex
    results, abstracts = search_openalex(search_phrase, number_of_abstracts)
    
    # Define system prompt message
    if prompt==None:
        system_message = PICO_bulletpoints_abstract
    else:
        system_message = prompt
    
    # Setup LLM
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cuda', attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n <|im_start|> assistant\n "

    inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to("cuda")
    output_ids = model.generate(inputs["input_ids"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    
    for i in range(len(abstracts)):
        print(f"For paper number {i+1}, titled {results[i]['title']}, the knowledge graph of the abstract gives the following information:\n")
        prompt = f"<|im_start|>user\n Extract the knowledge graph fom the following abstract:\n {abstracts[i]}<|im_end|>\n<|im_start|> assistant\n "

        inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to("cuda")
        output_ids = model.generate(inputs["input_ids"],max_new_tokens=200)
        answer = tokenizer.batch_decode(output_ids)[0]
        cut_answer = answer.split("<|im_start|> assistant\n",1)[1]
        print(cut_answer + '\n')

    return None

# Version 2: Giving the system message again for each paper
def analyse_repeated_sys_message(model_name, search_phrase, number_of_abstracts=10, prompt=None):
    # Get papers from OpenAlex
    results, abstracts = search_openalex(search_phrase, number_of_abstracts)
    
    # Define system prompt message
    if prompt==None:
        system_message = PICO_bulletpoints_abstract #PICO_message_abstract
    else:
        system_message = prompt
    
    # Setup LLM
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cuda', attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )
    
    for i in range(len(abstracts)):
        print(f"For paper number {i+1}, titled {results[i]['title']}, the knowledge graph of the abstract gives the following information:\n")
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n Extract the content from the following abstract:\n {abstracts[i]}<|im_end|>\n<|im_start|> assistant\n "

        inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to("cuda")
        output_ids = model.generate(inputs["input_ids"],max_new_tokens=400)
        answer = tokenizer.batch_decode(output_ids)[0]
        cut_answer = answer.split("<|im_start|> assistant\n",1)[1]
        print(cut_answer + '\n')

    return None

# Version 3: Give instructions stepwise. First, ask for a summary (not shown) and then for this summary to be itemized.
def analyse_stepwise(model_name, search_phrase, number_of_abstracts=10, prompt=None):
    # Get papers from OpenAlex
    results, abstracts = search_openalex(search_phrase, number_of_abstracts)
    
    # Define system prompt message
    if prompt==None:
        system_message = PICO_stepwise
    else:
        system_message = prompt
    
    # Setup LLM
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cuda', attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )
    
    for i in range(len(abstracts)):
        print(f"For paper number {i+1}, titled {results[i]['title']}, the knowledge graph of the abstract gives the following information:\n")
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n Extract the content from the following abstract:\n {abstracts[i]}<|im_end|>\n<|im_start|> assistant\n "
        prompt = f"<|im_start|>user\n Please format the summary into 4 bullet points, each bullet point focusing on one PICO elements (population, intervention, comparison and outcome):\n {abstracts[i]}<|im_end|>\n<|im_start|> assistant\n "

        inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=False).to("cuda")
        output_ids = model.generate(inputs["input_ids"],max_new_tokens=400)
        answer = tokenizer.batch_decode(output_ids)[0]
        cut_answer = answer.split("<|im_start|> assistant\n",1)[1]
        print(cut_answer + '\n')

    return None