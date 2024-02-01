import io
from flask import Flask, request, jsonify
import boto3
import uuid
from datetime import datetime
import os 
from jira import JIRA
from PyPDF2 import PdfReader
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from google.cloud import bigtable
from google.cloud.bigtable import column_family


import pandas as pd
import openpyxl
import time
from datetime import datetime
from itertools import zip_longest

openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA55IJCTYL6LUMZSGI'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/GPT/GCPapi/prodapt-nextgen-404505-61ccaa4ed806.json"
project_id = 'prodapt-nextgen-404505'
instance_id = 'prodapt'
output_table_rca = 'rca_output'
feedback_table = 'rca_feedback'
error_table = 'rca_error'
table_name_BRDMeta = 'BRDMeta'


app = Flask(__name__)

# AWS configurations
S3_BUCKET_NAME = 'quesanswithpdf'
table_name = 'quesanswithpdf'
region_name = 'ap-south-1'

table_name_fault = 'faultLens'
table_name_RCA = 'analysisRCA'
table_name_billing = 'billingInquiry3'

dynamodb = boto3.resource('dynamodb', region_name=region_name)
dynamodb_client = boto3.client('dynamodb',region_name=region_name,)

s3 = boto3.client('s3')

# --------------------------------------------------------------billing----------------------------------------------------------------------------------------------
@app.route('/billingenquiry/<accountNumber>', methods=['GET'])
 
def billingenquiry(accountNumber):
 
    try:
 
        params = {
 
            'TableName': table_name_billing,
 
            'FilterExpression': 'accountNumber = :accountNumber',
 
            'ExpressionAttributeValues': {
 
                ':accountNumber': {'S': accountNumber}
 
            }
 
        }
 
 
 
        result = dynamodb_client.scan(**params)
 
        transcript = result['Items'][0]['transcript']['S']
 
        return jsonify(result), 200
 
 
 
    except Exception as error:
 
        print(error)
 
        return 'Internal Server Error', 500
 
 
 
@app.route('/getAnswer/<accountNumber>/<int:i>', methods=['POST'])
 
def get_answer(accountNumber,i):
 
    try:
 
        data = request.get_json()
 
        question = data.get('question')
 
        # Query DynamoDB based on the question
 
        params = {
 
            'TableName': table_name_billing,
 
            'FilterExpression': 'accountNumber = :accountNumber',
 
            'ExpressionAttributeValues': {
 
                ':accountNumber': {'S': accountNumber}
 
            }
 
        }
 
        result = dynamodb_client.scan(**params)
 
        transcript = result['Items'][i]['transcript']['S']
 
        text_splitter = CharacterTextSplitter(
 
            separator= " ",
 
            chunk_size = 1000,
 
            chunk_overlap = 100,
 
            length_function = len,
 
        )
 
        texts = text_splitter.split_text(transcript)
 
        embeddings = OpenAIEmbeddings()
 
        docsearch = FAISS.from_texts(texts, embeddings)
 
        llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.4)
 
        chain = load_qa_chain(llm=llm, chain_type="stuff")
 
        docs = docsearch.similarity_search(question)
 
        answer = chain.run(input_documents=docs, question=question)
 
        print(answer)
 
        if answer:
 
            return jsonify({'answer': answer}), 200
 
        else:
 
            return jsonify({'message': 'Answer not found'}), 404
 
 
 
    except Exception as error:
 
        print(error)
 
        return jsonify({'message': 'Internal Server Error'}), 500
 
 
 
@app.route('/jira', methods=['POST'])
 
def jira():
 
    try:
 
        data = request.get_json()
 
        question = data.get('question')
 
 
 
        uuid_value = uuid.uuid4()
 
        uuid_value = uuid.uuid4()
 
        jira_api_key = "ATATT3xFfGF0dr_AAlzBtc7InCgLdMENOY2vPHlUABVGam1hhhiWEMcO89gyvueWxdLMWg05-c2HOcd33FQNHkyI0Ys9zz5npoJMI7J4TsyVkHnwe4bSqJrG-oA2gnhButAGawGn3nonAaeWw-zn54XitvTHJf9JhoR8qSRY8pr4l4LP8M1F6vE=C4E01230"
 
        jiraOptions = {'server': 'https://aws-callcenter-test.atlassian.net'}
 
        jira = JIRA(options=jiraOptions, basic_auth=('sparshdashtest@gmail.com', jira_api_key))
 
        issue_dict = {
 
            "project":{
 
                "key": "ATT"
 
            },
 
            "summary": f"Issue for questionID: {str(uuid_value)}",
 
            "description": question,
 
            "issuetype": {
 
                "name": "Task"
 
            }
 
        }
 
       
 
        new_issue = jira.create_issue(fields=issue_dict)
 
   
 
        jira_issue_id = new_issue.key
 
       
 
        return jsonify(jira_issue_id), 200
 
 
 
    except Exception as error:
 
        print(error)
 
        return 'Internal Server Error', 500
 
 
@app.route('/prompts/<accountNumber>/<int:i>', methods=['GET'])
 
def prompts(accountNumber,i):
 
    try:
 
        params = {
 
            'TableName': 'billingInquiry3',
 
            'FilterExpression': 'accountNumber = :accountNumber',
 
            'ExpressionAttributeValues': {
 
                ':accountNumber': {'S': accountNumber}
 
            }
 
        }
 
        result = dynamodb_client.scan(**params)
 
        transcript = result['Items'][i]['transcript']['S']
 
        about_me_prompt = f'''
 
        Please give me the three question related to body of text
 
        ques1 :
 
        ques2 :
 
        ques3 :
 
        This is the body of text to extract the information from:
 
        {transcript}
 
       
 
        and these five questions
 
            1. "When's my next payment due, and how much do I owe?"
 
            2. "Confirm if my recent payment was received and applied correctly."
 
            3. "Explain new charges on my bill and their reasons, please."
 
            4. "Why the bill increase? Breakdown of changes causing it?"
 
            5. "Explain how my plan change affects my bill charges."
 
        should not come. every time give me different question dont repeat the question.
 
       
 
        '''
 
        # Getting the response back from ChatGPT (gpt-3.5-turbo)
 
        openai_response = openai.ChatCompletion.create(
 
            model = 'gpt-3.5-turbo-16k',
 
            messages = [{'role': 'user', 'content': about_me_prompt}]
 
        )
 
 
        data = openai_response['choices'][0]['message']['content']
 
        x = data.split('\n')
 
        print(x)
 
        response_data = [
 
            {
 
                "question": x[0]
 
            },
 
            {
 
                "question": x[1]
 
            },
 
            {
 
                "question": x[2]
 
            }
 
        ]
        return jsonify(response_data), 200
 
    except Exception as error:
 
        print(error)
 
        return 'Internal Server Error', 500
 

@app.route('/summary', methods=['GET'])
def summary():
    try:
        response = dynamodb.scan(
        TableName='billingInquiry3',
        ProjectionExpression='transcriptSummary'
        )

        # Extract the column values from the response
        column_values = [item['transcriptSummary'] for item in response['Items']]
        about_me_prompt = f'''
        Please give me the final summary of all the summaries related to body of text.
        This is the body of text to extract the information from:
        {column_values}
        '''
        CSummary = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-16k',
            messages = [{'role': 'user', 'content': about_me_prompt}]
        )
        print(CSummary['choices'][0]['message']['content'])
        
        return jsonify(CSummary['choices'][0]['message']['content']), 200
        
    except Exception as error:
        print(error)
        return 'Internal Server Error', 500




# ------------------------------------------------------------------Contract Summerization-----------------------------------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded file from the request
    uploaded_file = request.files['file']

    # Generate a unique PDF ID
    pdf_id = str(uuid.uuid4())

    # Upload the file to S3
    
    file_key = f'pdfs/{uploaded_file.filename}'
    
    try:
        s3.head_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        return jsonify({'message': 'File already exists'})
    except:
        s3.upload_fileobj(uploaded_file, S3_BUCKET_NAME, file_key)

        # Store reference in DynamoDB
        try:
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'pdfName',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'pdfName',
                        'AttributeType': 'S'
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            # Wait until the table is created
            table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
            print('Table created:', table.table_name)

        except dynamodb.meta.client.exceptions.ResourceInUseException:
            # If the table already exists, just load the existing table
            table = dynamodb.Table(table_name)
            print('Table loaded:', table.table_name)

        # Define the items (data) you want to insert
        items = [
            {
                "pdfName": {
                    "S": uploaded_file.filename
                }
            }
        ]

        # Create a list of item requests for batch writing
        item_requests = []
        for item in items:
            item_requests.append({
                'PutRequest': {
                    'Item': item
                }
            })

        dynamodb_client = boto3.client('dynamodb', region_name=region_name)
        response = dynamodb_client.batch_write_item(RequestItems={table_name: item_requests})

        return jsonify({'message': 'File uploaded successfully'})
    
@app.route('/getSummary/<filename>', methods=['GET'])
def getSummary(filename):
    # Check if the item exists in DynamoDB
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={'pdfName': filename})
    if 'Item' not in response:
        return jsonify({'message': 'File not found in DynamoDB'})

    # Get the object from S3
    file_key = f'pdfs/{filename}'
    try:
        s3_response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        
        file_content = s3_response['Body'].read()
        pdf_file = io.BytesIO(file_content)
        
        reader=PdfReader(pdf_file)
    
        raw_text=' '
        for i,page in enumerate(reader.pages):
            text=page.extract_text()
            if text:
                raw_text+=text
            
        raw_text=raw_text.replace('\n',' ')
        dynamodb_client = boto3.client('dynamodb', region_name=region_name)
        # Update the DynamoDB item with the summary
        dynamodb_client.update_item(
            TableName=table_name,
            Key={'pdfName': {'S': filename}},
            UpdateExpression='SET #na = :val',
            ExpressionAttributeNames={'#na': 'transcript'},
            ExpressionAttributeValues={':val': {'S': raw_text}}
        )
        about_me_prompt = f'''
             Please summerize my text in bullet point.
            This is the body of text to extract the information from:
            {raw_text}
            '''
                
        openai_response = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo-16k',
                messages = [{'role': 'user', 'content': about_me_prompt}]
            )
        
        summary = openai_response['choices'][0]['message']['content']

        dynamodb_client.update_item(
            TableName=table_name,
            Key={'pdfName': {'S': filename}},
            UpdateExpression='SET #na = :val',
            ExpressionAttributeNames={'#na': 'summary'},
            ExpressionAttributeValues={':val': {'S': summary}}
        )

        return jsonify({'Summary': summary})
        
        
    except Exception as e:
        return jsonify({'message': 'Error fetching file', 'error': str(e)})
    
@app.route('/getMetaData/<filename>', methods=['GET'])
def get_MetaData(filename):
    # Check if the item exists in DynamoDB
    table = dynamodb.Table(table_name)
    response = table.get_item(Key={'pdfName': filename})
    if 'Item' not in response:
        return jsonify({'message': 'File not found in DynamoDB'})

    # Get the object from S3
    file_key = f'pdfs/{filename}'
    try:
        s3_response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        
        file_content = s3_response['Body'].read()
        pdf_file = io.BytesIO(file_content)
        
        reader=PdfReader(pdf_file)
    
        raw_text=' '
        for i,page in enumerate(reader.pages):
            text=page.extract_text()
            if text:
                raw_text+=text
            
        raw_text=raw_text.replace('\n',' ')
            
        about_me_prompt = f'''
            Give document metadata summary - 1st Buisness metadata (Purpose: ans in one words, Data Classification : ans in one words, Compliance :ans in one statement). 2nd Operational Metadata (Creation Date : ans in one word, Data Source : ans in one word, Data Quality : ans in one word):
            {raw_text}
            '''
            # Getting the response back from ChatGPT (gpt-3.5-turbo)
        openai_response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-16k',
            messages = [{'role': 'user', 'content': about_me_prompt}]
        )
        metaData = openai_response['choices'][0]['message']['content']
        
        dynamodb_client.update_item(
            TableName=table_name,
            Key={'pdfName': {'S': filename}},
            UpdateExpression='SET #na = :val',
            ExpressionAttributeNames={'#na': 'metaData'},
            ExpressionAttributeValues={':val': {'S': metaData}}
        )
        return jsonify({'meataData': metaData})
        
        
    except Exception as e:
        return jsonify({'message': 'Error fetching file', 'error': str(e)})



@app.route('/getEntireTable', methods=['GET'])
def get_entire_table():
    try:
        # Get all items from DynamoDB table
        table = dynamodb.Table(table_name)
        response = table.scan()

        items = response.get('Items', [])
        
        # If there are more items than what the response contained, continue fetching
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
        
        return jsonify({'TableData': items})
        
    except Exception as e:
        return jsonify({'message': 'Error fetching table data', 'error': str(e)})
    

@app.route('/deletePDF/<filename>', methods=['DELETE'])
def delete_pdf(filename):
    try:
        # Delete the PDF from S3
        file_key = f'pdfs/{filename}'
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=file_key)

        # Delete the PDF record from DynamoDB
        table = dynamodb.Table(table_name)
        table.delete_item(Key={'pdfName': filename})

        return jsonify({'message': 'PDF and related data deleted successfully'})

    except Exception as e:
        return jsonify({'message': 'Error deleting PDF and related data', 'error': str(e)})



@app.route('/getPDFNames', methods=['GET'])
def get_pdf_names():
    try:
        # Query DynamoDB to get all items and their 'pdfName' attribute
        table = dynamodb.Table(table_name)
        response = table.scan(ProjectionExpression='pdfName')

        pdf_names = [item['pdfName'] for item in response.get('Items', [])]

        # If there are more items than what the response contained, continue fetching
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ProjectionExpression='pdfName',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            pdf_names.extend([item['pdfName'] for item in response.get('Items', [])])

        return jsonify({'PDFNames': pdf_names})

    except Exception as e:
        return jsonify({'message': 'Error fetching PDF names', 'error': str(e)})


@app.route('/quesAns/<filename>', methods=['POST'])
def get_Ans(filename):
    
    data = request.get_json()
    question = data.get('question')
    try:
        data = request.get_json()
        question = data.get('question')
        params = {
            'TableName': table_name,
            'FilterExpression': 'pdfName = :pdfName',
            'ExpressionAttributeValues': {
                ':pdfName': {'S': filename}
            }
        }

        result = dynamodb_client.scan(**params)
        transcript = result['Items'][0]['transcript']['S']
        text_splitter = CharacterTextSplitter(
            separator= " ",
            chunk_size = 1000,
            chunk_overlap = 100,
            length_function = len,
        )

        texts = text_splitter.split_text(transcript)

        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)

        llm = OpenAI(model_name="text-davinci-003", temperature=0.4)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        docs = docsearch.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)
        print(answer)
        return jsonify({'answer': answer})
    except Exception as error:
        print(error)
        return jsonify({'message': 'Internal Server Error'}), 500

#------------------------------------------------------------------------------------------------------RCA---------------------------------------------------------------------------------------------------------------


# @app.route('/getLog', methods=['POST'])
# def send_log():
#     try:
#         json_data = request.get_json()
#         if json_data is None:
#             return "No JSON data received", 400

#         log = json_data.get('log')
#         if log is None:
#             return "No log message found in JSON data", 400
        
#         my_custom_function =[
#                     {
#                         'name': 'get_rca_result_from_openai',
#                         'description': 'read the log from the body of input text',
#                         'parameters': {
#                             'type': 'object',
#                             'properties': {
#                                 'classification': {
#                                     'type': 'string',
#                                     'description': 'Classify the error from following points - Incorrect API Permissions, Unsecured Endpoints and Data Access Tokens, Invalid Session Management, Expiring APIs, Bad URLs/HTTP Errors, Overly Complex API Endpoints, Exposed APIs on IPs'
#                                 },
#                                 'analysis': {
#                                     'type': 'string',
#                                     'description': 'Give analysis of the error in 5-WHY principle in generic form eg - Why did the API error occur?, Why is there maintenance or an unexpected issue?, etc. (give 5 questions and their analysis) also add ". \n" after end of each analysis '
#                                 },
#                                 'solution': {
#                                     'type': 'string',
#                                     'description': 'possible solutions considering analysis of the error in 5 points'
#                                 },
#                                 'depth_classification': {
#                                     'type': 'string',
#                                     'description': 'Classify the error in following points - 1. Request URL, 2. Error Type, 3. Error Code, 4. Error Message, 5. Request Method, 6. Category, 7. Probable Cause, 8. Context, 9. Framework version, 10. Exception Location, 11. Impacted systems, 12. Impacted services, 13. Number of lines of logs processed (count yourself), 14. Is password or sensitive information exposed? (Yes/No), 15. Is there any 3rd party system integration challenge? If yes, list them. (write in number Points and type "not available" if not found)'
#                                 },
#                                 'specific_solution': {
#                                     'type': 'string',
#                                     'description': 'possible solutions considering analysis of the error specifically error message and error type in 5 points'
#                                 },
#                             }
#                         }   
#                     }
#                 ]
#         response = openai.ChatCompletion.create(
#                     model="gpt-4",
#                     messages=[{'role': 'user', 'content': log}],
#                     temperature=0.3,
#                     max_tokens=800,
#                     top_p=1,
#                     frequency_penalty=0,
#                     presence_penalty=0,
#                     functions = my_custom_function,
#                     function_call = 'auto'
#                 )
        
#         answer = response['choices'][0]['message']['content']
#         print(answer)
#         pre_analysis = answer['analysis']
#         pre_analysis = pre_analysis.replace('? ','?\n')
#         pre_analysis = pre_analysis.replace('. \n','. \n\n')
#         return jsonify({'answer': answer}) 
#     except Exception as e:
#         return jsonify({'message': 'Error in fetching Error message', 'error': str(e)})
    
    
    
    # try:
    #         table = dynamodb.create_table(
    #             TableName=table_name_RCA,
    #             KeySchema=[
    #                 {
    #                     'AttributeName': 'log',
    #                     'KeyType': 'HASH'
    #                 }
    #             ],
    #             AttributeDefinitions=[
    #                 {
    #                     'AttributeName': 'log',
    #                     'AttributeType': 'S'
    #                 }
    #             ],
    #             ProvisionedThroughput={
    #                 'ReadCapacityUnits': 5,
    #                 'WriteCapacityUnits': 5
    #             }
    #         )
    #         # Wait until the table is created
    #         table.meta.client.get_waiter('table_exists').wait(TableName=table_name_RCA)
    #         print('Table created:', table.table_name_RCA)

    # except dynamodb.meta.client.exceptions.ResourceInUseException:
    #         # If the table already exists, just load the existing table
    #         table = dynamodb.Table(table_name_RCA)
    #         print('Table loaded:', table.table_name_RCA)

    #     # Define the items (data) you want to insert
    # items = [
    #         {
    #             "log": {
    #                 "S": log
    #             },
    #             "analysis": {
    #                 "S": pre_analysis
    #             },
    #             "whyAnalysis": {
    #                 "S": pre_analysis
    #             },
    #             "proposedSolution": {
    #                 "S": pre_analysis
    #             }
    #         }
    #     ]

    #     # Create a list of item requests for batch writing
    # item_requests = []
    # for item in items:
    #         item_requests.append({
    #             'PutRequest': {
    #                 'Item': item
    #             }
    #         })

    # dynamodb_client = boto3.client('dynamodb', region_name=region_name)
    # response = dynamodb_client.batch_write_item(RequestItems={table_name_RCA: item_requests})
        
        
    # print(response)


#----------------------------------------------------------------------------rca GCP api------------------------------------------------------------------------------------------

def fetch_whole_table_data():
    try:
        # Create a Bigtable client and connect to the table
        client = bigtable.Client(project=project_id, admin=True)
        instance = client.instance(instance_id)
        table = instance.table(output_table_rca)

        # Fetch all rows from the table
        rows = table.read_rows()

        # Initialize an empty list to store the table data
        table_data = []

        for row in rows:
            # Initialize an empty dictionary to store row data
            row_data = {
                'row_key': row.row_key.decode('utf-8')
            }

            # Fetch data from the row columns
            for column_family_id, columns in row.cells.items():
                for column, cells in columns.items():
                    # Join cell values if there are multiple cells
                    cell_values = [cell.value.decode('utf-8') for cell in cells]
                    row_data[f'{column_family_id}:{column}'] = '\n'.join(cell_values)

            # Append the row data to the list
            table_data.append(row_data)

        # Return the table data as JSON
        return jsonify(table_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/fetch_whole_table', methods=['GET'])
def fetch_table():
    return fetch_whole_table_data()

#--------------------------------------------------------------------code for storing feedback RCA---------------------------------------------------------------
@app.route('/feedback', methods=['POST'])
def receive_feedback():
    data = request.get_json()
    
    row_key = data.get('row_key')
    feedback = data.get('feedback')
    comments = data.get('comments')

    print(f"rowkey: {row_key}")
    print(f"feedback: {feedback}")
    print(f"Comments: {comments}")
    
    
    client = bigtable.Client(project=project_id, admin=True)
    instance = client.instance(instance_id)
    row_key_feedback = f'{row_key}'
    
    if not instance.exists():
        instance.create()
    else:
        print(f'Instance {instance_id} already exists.')
        
    table = instance.table(feedback_table)

    cf1 = table.column_family('feedback')
    cf2 = table.column_family('comment')

    if not table.exists():
        table.create()
        cf1.create()
        cf2.create()
    else:
        print(f'Table {feedback_table} already exists.')

    row = table.row(row_key_feedback)

    row.set_cell(column_family_id='feedback', column='feedback', value=feedback)
    row.set_cell(column_family_id='comment', column='comment', value=comments)
    
    print(table.mutate_rows([row]))

    response = {'message': 'Feedback received successfully'}
    return jsonify(response)

#-----------------------------------------------------putting error in a table rca----------------------------------------------------------------------------
@app.route('/error', methods=['POST'])
def receive_error():
    data = request.get_json()
    
    error_message = data.get('error')

    print(f"Comments: {error_message}")
    
    
    client = bigtable.Client(project=project_id, admin=True)
    instance = client.instance(instance_id)
    uuid_value = uuid.uuid4()
    row_key = f'{uuid_value}'
    
    row_key_error = f'{row_key}'
    
    if not instance.exists():
        instance.create()
    else:
        print(f'Instance {instance_id} already exists.')
        
    table = instance.table(error_table)

    cf1 = table.column_family('error_message')

    if not table.exists():
        table.create()
        cf1.create()
    else:
        print(f'Table {error_table} already exists.')

    row = table.row(row_key_error)

    row.set_cell(column_family_id='error_message', column='error_message', value=error_message)
    
    print(table.mutate_rows([row]))

    response = {'message': 'error received successfully'}
    return jsonify(response)


#--------------------------------------------------------Auto Test Generator with BRD/PRD (For Meta)------------------------------------------------------------------------------
table_name_BRDMeta = 'BRDMeta'
@app.route('/getEntireTableBRDMeta', methods=['GET'])
def get_entire_table_BRDMeta():
    try:
        dynamodb = boto3.resource('dynamodb', region_name=region_name)
        table = dynamodb.Table(table_name_BRDMeta)
        response = table.scan()

        lastScenerio = datetime.fromisoformat("1980-01-02T12:00:00")
        round_time = datetime.fromisoformat("1980-01-02T12:00:00")
        items = response.get('Items', [])
        print(items)
        
        total_time = 0
        i = 0
        file = 0
        text = 0
        round_no = 0
        for item in items:
            i+=1
            total_time +=int(item['totalTime'])
            if "file" in item['fromWhere']:
                file += 1
            else:
                text += 1
            if datetime.fromisoformat(item['atTime']) > lastScenerio:
                lastScenerio = datetime.fromisoformat(item['atTime'])
            if datetime.fromisoformat(item['atTime']) == round_time:
                round_time = datetime.fromisoformat(item['atTime'])
            else:
                round_time = datetime.fromisoformat(item['atTime'])
                round_no = round_no+1
                
        avg_time = round(total_time/i)

        # If there are more items than what the response contained, continue fetching
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
       
        return jsonify({'TableData': items,"avg_time":avg_time,"no_of_file":file,"no_of_text":text,"lastScenerio": lastScenerio,"round": round_no})
       
    except Exception as e:
        return jsonify({'message': 'Error fetching table data', 'error': str(e)})
    
@app.route('/updateStatus', methods=['POST'])
def post_status_BRDMeta():
    data = request.get_json()
    test_senerio = data.get('test_senerio')
    test_senerio_id = data.get('test_senerio_id')
    
    status = "Not-Transform"
    try:
        response = dynamodb_client.get_item(
            TableName=table_name_BRDMeta,
            Key={
                'IP_Test_Senerio': {'S': test_senerio},
                'IP_Test_Senerio_id': {'S': test_senerio_id}
            }
            )
        if 'Item' in response:
            existing_item = response["Item"]
            existing_item['updateStatus'] = {'S': status}

            jira_api_key = "ATATT3xFfGF0UIhmx_Ode6f3Tc-_ZLHsMmJU4E2jgHuJYPCyxihIJNSW0lWlQtgtQ9HY2qAJdjlI5WfZ4E_uAI9egyuiyY30CC7pD481olKhN7Go0Q0JF_tS_TZiz4Z8hATeXKM6EnPnudKRvwnFan2kkUbjMwEQpnb5fi-M9e04hIJG1OWXt_E=9C6925BC"
            jiraOptions = {'server': 'https://rishutrivedi.atlassian.net'}
            jira = JIRA(options=jiraOptions, basic_auth=('rishutrivedi119@gmail.com', jira_api_key))
            issue_dict = {
                "project": {
                "key": "BRD"
            },
            "summary": f"Issue for Scenario: {test_senerio}",
            "description": f"**Gen_metaData** : \n {existing_item['Gen_metaData']['S']} \n\n **Gen_testData** \n {existing_item['Gen_testData']['S']} \n\n **Gen_testSteps** \n{existing_item['Gen_testSteps']['S']}**Gen_testResult** \n {existing_item['Gen_testResult']['S']}",
            "issuetype": {
                "name": "Task"
            }
            }
            
            new_issue = jira.create_issue(fields=issue_dict)
        
            jira_issue_key = new_issue.key
            
            existing_item['jiraKey'] = {'S': jira_issue_key}
                   
            response = dynamodb_client.put_item(
                TableName=table_name_BRDMeta,
                Item = existing_item
            ) 
            
            return jsonify(jira_issue_key), 200
            
        else:
            return jsonify("Data you are trying to fetch is not in the table"), 200   
         
    except Exception as e:
        return jsonify({'message': 'Error fetching table data', 'error': str(e)})
   
@app.route('/updateTableBRDMeta', methods=['POST'])
def update_table_BRDMeta():
    data = request.get_json()
    test_senerio = data.get('test_senerio')
    test_senerio_id = data.get('test_senerio_id')
    updateMetaData = data.get('metaData')
    updateTestData = data.get('testData')
    updateTestSteps = data.get('testSteps')
    updateTestResult = data.get('testResult')
    
    current_time = datetime.now().strftime("%d-%m-%Y/%H-%M")
    try:
        response = dynamodb_client.get_item(
            TableName=table_name_BRDMeta,
            Key={
                'IP_Test_Senerio': {'S': test_senerio},
                'IP_Test_Senerio_id': {'S': test_senerio_id}
            }
            )
        if 'Item' in response:
            existing_item = response["Item"]
            
            if updateMetaData:
                if 'updateMetaData' in existing_item:
                    existing_item['updateMetaData']['M'][current_time] = {'S': updateMetaData}
                else:
                    existing_item['updateMetaData'] = {'M':{current_time: {'S': updateMetaData}}}
                    
            if updateTestData:
                if 'updateTetaData' in existing_item:
                    existing_item['updateTetaData']['M'][current_time] = {'S': updateTestData}
                else:
                    existing_item['updateTetaData'] = {'M':{current_time: {'S': updateTestData}}}
            
            if updateTestSteps:
                if 'updateTestSteps' in existing_item:
                    existing_item['updateTestSteps']['M'][current_time] = {'S': updateTestSteps}
                else:
                    existing_item['updateTestSteps'] = {'M':{current_time: {'S': updateTestSteps}}}
                    
            if updateTestResult:
                if 'updateTestResult' in existing_item:
                    existing_item['updateTestResult']['M'][current_time] = {'S': updateTestResult}
                else:
                    existing_item['updateTestResult'] = {'M':{current_time: {'S': updateTestResult}}}
            
            existing_item['updateStatus'] = {'S': "Transform"}
                           
        else:
            return jsonify("Data you are trying to fetch is not in the table"), 200

        jira_api_key = "ATATT3xFfGF0UIhmx_Ode6f3Tc-_ZLHsMmJU4E2jgHuJYPCyxihIJNSW0lWlQtgtQ9HY2qAJdjlI5WfZ4E_uAI9egyuiyY30CC7pD481olKhN7Go0Q0JF_tS_TZiz4Z8hATeXKM6EnPnudKRvwnFan2kkUbjMwEQpnb5fi-M9e04hIJG1OWXt_E=9C6925BC"
        jiraOptions = {'server': 'https://rishutrivedi.atlassian.net'}
        jira = JIRA(options=jiraOptions, basic_auth=('rishutrivedi119@gmail.com', jira_api_key))
        issue_dict = {
            "project": {
            "key": "BRD"
        },
        "summary": f"Issue for Scenario: {test_senerio}",
        "description": 
          f"**Gen_metaData** : \n {existing_item['Gen_metaData']['S']} \n\n **Gen_testData** \n {existing_item['Gen_testData']['S']} \n\n **Gen_testSteps** \n{existing_item['Gen_testSteps']['S']}**Gen_testResult** \n {existing_item['Gen_testResult']['S']}\n\n**updataMetaData** : \n {updateMetaData} \n\n **updateTestData** \n {updateTestData}\n\n **update_testSteps** \n{updateTestSteps} **updateTestResult** \n {updateTestResult}" ,
        "issuetype": {
            "name": "Task"
        }
        }
        
        new_issue = jira.create_issue(fields=issue_dict)
    
        jira_issue_key = new_issue.key
        
        existing_item['jiraKey'] = {'S': jira_issue_key}
        
        response = dynamodb_client.put_item(
                TableName=table_name_BRDMeta,
                Item = existing_item
            )
        
        return jsonify(jira_issue_key), 200
    
   
    except Exception as e:
        return jsonify({'message': 'Error fetching table data', 'error': str(e)})


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# def get_functional_test_response_meta(textarea):
#     print("in function")
#     metadata = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo-16k",
#     messages=[
#         {
#         "role": "system",
#         "content": "You are a system. Your task is to generate functional test metadata for cross-application verifications involving the app. Only generate information at the specified points:\n1.Test Case Name:\n2.Test Objective:\n3.Preconditions:\n4.Inputs:\"\n        "
#         },
#         {
#         "role": "user",
#         "content": "how to upload a profile picture on facebook"
#         },
#         {
#         "role": "assistant",
#         "content": "1.Test Case Name: Upload Profile Picture on Facebook\n2.Test Objective: To verify that a user can successfully upload a profile picture on Facebook.\n3.Preconditions:\n   - User must have a valid Facebook account.\n   - User must be logged in to their Facebook account.\n4.Inputs:\n   - Profile picture file to be uploaded."
#         },
#         {
#         "role": "user",
#         "content": "how to see a story in Instagram"
#         },
#         {
#         "role": "assistant",
#         "content": "1.Test Case Name: View Story on Instagram\n2.Test Objective: To verify that a user can successfully view a story on Instagram.\n3.Preconditions:\n   - User must have a valid Instagram account.\n   - User must be logged in to their Instagram account.\n4.Inputs:\n   - None. The test will involve navigating to the Instagram app and accessing the stories section."
#         },
#         {
#         "role": "user",
#         "content": f"{textarea}"  
#         }
#     ],
#     temperature=0.06,
#     max_tokens=5376,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     print("meta metadata calculated")

#     test_result = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-16k",
#         messages=[
#             {
#             "role": "system",
#             "content": "You are a system tasked with generating functional detailed test steps and Expected result for cross-application verifications involving the app on iOS, Android, and the web .Between test steps and expected results, add a hash ('##').\nFor your understanding, I am giving you an \nexample.\n\nTest Steps for iOS:\n1.-----\n2.----\n(so on)\n##\nExpected Result for iOS:\n-----\n##\nTest Steps for Android:\n1.-----\n2.----\n(so on)\n##\nExpected Result for Android:\n-----\n##\nTest Steps for Web:\n1.-----\n2.----\n(so on)\n##\nExpected Result for web:\n-----\n\n\n\n\n-"
#             },
#             {
#             "role": "user",
#             "content": "how to check story in Instagram"
#             },
#             {
#             "role": "assistant",
#             "content": "Test Steps for iOS:\n1. Launch the Instagram app on your iOS device.\n2. Log in to your Instagram account using your credentials.\n3. Tap on the \"Home\" icon at the bottom left corner of the screen to go to the home feed.\n4. Look for the profile pictures with colorful rings around them at the top of the feed. These indicate that there are stories available.\n5. Tap on the profile picture of the user whose story you want to view.\n6. Swipe left or right on the screen to view the next or previous story in the user's story sequence.\n7. Tap on the screen to pause or resume the story.\n8. Swipe down on the screen to exit the story and return to the home feed.\n\n##\nExpected Result for iOS:\n- The Instagram app launches successfully.\n- You are able to log in to your Instagram account.\n- The home feed is displayed.\n- The profile pictures with colorful rings around them are visible at the top of the feed.\n- Tapping on a profile picture opens the user's story.\n- Swiping left or right allows you to view the next or previous story.\n- Tapping on the screen pauses or resumes the story.\n- Swiping down on the screen exits the story and returns to the home feed.\n\n##\nTest Steps for Android:\n1. Open the Instagram app on your Android device.\n2. Log in to your Instagram account using your credentials.\n3. Tap on the \"Home\" icon at the bottom left corner of the screen to go to the home feed.\n4. Look for the profile pictures with colorful rings around them at the top of the feed. These indicate that there are stories available.\n5. Tap on the profile picture of the user whose story you want to view.\n6. Swipe left or right on the screen to view the next or previous story in the user's story sequence.\n7. Tap on the screen to pause or resume the story.\n8. Swipe down on the screen to exit the story and return to the home feed.\n\n##\nExpected Result for Android:\n- The Instagram app opens successfully.\n- You are able to log in to your Instagram account.\n- The home feed is displayed.\n- The profile pictures with colorful rings around them are visible at the top of the feed.\n- Tapping on a profile picture opens the user's story.\n- Swiping left or right allows you to view the next or previous story.\n- Tapping on the screen pauses or resumes the story.\n- Swiping down on the screen exits the story and returns to the home feed.\n\n##\nTest Steps for Web:\n1. Open a web browser on your device and go to the Instagram website (www.instagram.com).\n2. Log in to your Instagram account using your credentials.\n3. Click on the \"Home\" icon at the top left corner of the screen to go to the home feed.\n4. Look for the profile pictures with colorful rings around them at the top of the feed. These indicate that there are stories available.\n5. Click on the profile picture of the user whose story you want to view.\n6. Use the arrow buttons on the screen to navigate through the user's story sequence.\n7. Click on the screen to pause or resume the story.\n8. Click on the \"X\" button at the top right corner of the screen to exit the story and return to the home feed.\n\n##\nExpected Result for Web:\n- The Instagram website loads successfully.\n- You are able to log in to your Instagram account.\n- The home feed is displayed.\n- The profile pictures with colorful rings around them are visible at the top of the feed.\n- Clicking on a profile picture opens the user's story.\n- Using the arrow buttons allows you to navigate through the story.\n- Clicking on the screen pauses or resumes the story.\n- Clicking on the \"X\" button exits the story and returns to the home feed."
#             },
#             {
#             "role": "user",
#             "content": "Create and Verify Check-in Post with Friends Audience"
#             },
#             {
#             "role": "assistant",
#             "content": "Test Steps for iOS:\n1. Launch the Instagram app on your iOS device.\n2. Log in to your Instagram account using your credentials.\n3. Tap on the \"Add Post\" button (represented by a plus icon) at the bottom center of the screen.\n4. Select a photo or video from your device's gallery to include in the check-in post.\n5. Tap on the \"Next\" button at the top right corner of the screen.\n6. Apply filters or edit the photo/video as desired.\n7. Tap on the \"Next\" button.\n8. In the caption field, type the desired text for your check-in post.\n9. Tap on the \"Tag People\" button to tag your friends in the post.\n10. Search for and select the friends you want to tag.\n11. Tap on the \"Done\" button.\n12. Tap on the \"Add Location\" button to add a location to your check-in post.\n13. Search for and select the desired location.\n14. Tap on the \"Share\" button at the top right corner of the screen to publish your check-in post.\n\n##\nExpected Result for iOS:\n- The Instagram app launches successfully.\n- You are able to log in to your Instagram account.\n- The \"Add Post\" button is visible at the bottom center of the screen.\n- You can select a photo or video from your device's gallery.\n- The selected media is displayed on the screen.\n- The \"Next\" button is visible at the top right corner of the screen.\n- Filters and editing options are available for the photo/video.\n- The \"Next\" button is visible after applying filters or editing.\n- The caption field is visible and editable.\n- The \"Tag People\" button is visible and clickable.\n- You can search for and select friends to tag in the post.\n- The \"Done\" button is visible and clickable after tagging friends.\n- The \"Add Location\" button is visible and clickable.\n- You can search for and select a location for the check-in post.\n- The \"Share\" button is visible and clickable to publish the post.\n\n##\nTest Steps for Android:\n1. Open the Instagram app on your Android device.\n2. Log in to your Instagram account using your credentials.\n3. Tap on the \"Add Post\" button (represented by a plus icon) at the bottom center of the screen.\n4. Select a photo or video from your device's gallery to include in the check-in post.\n5. Tap on the \"Next\" button at the top right corner of the screen.\n6. Apply filters or edit the photo/video as desired.\n7. Tap on the \"Next\" button.\n8. In the caption field, type the desired text for your check-in post.\n9. Tap on the \"Tag People\" button to tag your friends in the post.\n10. Search for and select the friends you want to tag.\n11. Tap on the \"Done\" button.\n12. Tap on the \"Add Location\" button to add a location to your check-in post.\n13. Search for and select the desired location.\n14. Tap on the \"Share\" button at the top right corner of the screen to publish your check-in post.\n\n##\nExpected Result for Android:\n- The Instagram app opens successfully.\n- You are able to log in to your Instagram account.\n- The \"Add Post\" button is visible at the bottom center of the screen.\n- You can select a photo or video from your device's gallery.\n- The selected media is displayed on the screen.\n- The \"Next\" button is visible at the top right corner of the screen.\n- Filters and editing options are available for the photo/video.\n- The \"Next\" button is visible after applying filters or editing.\n- The caption field is visible and editable.\n- The \"Tag People\" button is visible and clickable.\n- You can search for and select friends to tag in the post.\n- The \"Done\" button is visible and clickable after tagging friends.\n- The \"Add Location\" button is visible and clickable.\n- You can search for and select a location for the check-in post.\n- The \"Share\" button is visible and clickable to publish the post.\n\n##\nTest Steps for Web:\n1. Open a web browser on your device and go to the Instagram website (www.instagram.com).\n2. Log in to your Instagram account using your credentials.\n3. Click on the \"Add Post\" button (represented by a plus icon) at the top left corner of the screen.\n4. Select a photo or video from your device's gallery to include in the check-in post.\n5. Click on the \"Next\" button at the bottom right corner of the screen.\n6. Apply filters or edit the photo/video as desired.\n7. Click on the \"Next\" button.\n8. In the caption field, type the desired text for your check-in post.\n9. Click on the \"Tag People\" button to tag your friends in the post.\n10. Search for and select the friends you want to tag.\n11. Click on the \"Done\" button.\n12. Click on the \"Add Location\" button to add a location to your check-in post.\n13. Search for and select the desired location.\n14. Click on the \"Share\" button at the bottom right corner of the screen to publish your check-in post.\n\n##\nExpected Result for Web:\n- The Instagram website loads successfully.\n- You are able to log in to your Instagram account.\n- The \"Add Post\" button is visible at the top left corner of the screen.\n- You can select a photo or video from your device's gallery.\n- The selected media is displayed on the screen.\n- The \"Next\" button is visible at the bottom right corner of the screen.\n- Filters and editing options are available for the photo/video.\n- The \"Next\" button is visible after applying filters or editing.\n- The caption field is visible and editable.\n- The \"Tag People\" button is visible and clickable.\n- You can search for and select friends to tag in the post.\n- The \"Done\" button is visible and clickable after tagging friends.\n- The \"Add Location\" button is visible and clickable.\n- You can search for and select a location for the check-in post.\n- The \"Share\" button is visible and clickable to publish the post."
#             },
#             {
#             "role": "user",
#             "content": f"{textarea}"  
#             }
#         ],
#         temperature=0.22,
#         max_tokens=7764,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#         )
#     print("meta test_result calculated")
    
#     test_data = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-16k",
#          messages=[
#             {
#             "role": "system",
#             "content": "You are a system tasked with generating functional test data for cross-application verifications involving the app on iOS, Android, and the web. Generate test data only for the following points. Don't give any extra data. Give data on the given points only.:\n\nUser Details:\na. Username:\nb. Password:\n\nPost Details:\na. Post Type:\nb. Location/Restaurant:\nc. Audience:\nd. Post Content:\n\nPlatforms:\n            "
#             },
#             {
#             "role": "user",
#             "content": "give me the test case for story checking in facebook"
#             },
#             {
#             "role": "assistant",
#             "content": "User Details:\na. Username: testuser1\nb. Password: Test@123\n\nPost Details:\na. Post Type: Story\nb. Location/Restaurant: N/A\nc. Audience: Friends\nd. Post Content: \"Having a great day!\"\n\nPlatforms:\n- iOS\n- Android\n- Web"
#             },
#             {
#             "role": "user",
#             "content": "how to check story in Instagram"
#             },
#             {
#             "role": "assistant",
#             "content": "User Details:\na. Username: testuser1\nb. Password: Test@123\n\nPost Details:\na. Post Type: Story\nb. Location/Restaurant: N/A\nc. Audience: Followers\nd. Post Content: \"Enjoying the sunset\"\n\nPlatforms:\n- iOS\n- Android\n- Web"
#             },
#             {
#             "role": "user",
#             "content": f"{textarea}"
#             }            
#         ],
#         temperature=0.14,
#         max_tokens=5376,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#         )
#     print("meta test_data calculated")
    
#     metadata = metadata['choices'][0]['message']['content']
#     test_result = test_result['choices'][0]['message']['content']
#     test_data = test_data['choices'][0]['message']['content']
    
#     return metadata,test_data,test_result

def get_functional_test_response_meta(textarea):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
            "role": "system",
            "content": "You are a system. Your task is to generate functional test metadata, test data, test steps, and expected results for cross-application verifications involving the app. I am giving you the specific point for each to generate. Always keep in mind to add a hash (##) after metadata, test data, test steps.\n\nMetadata : \n1. Test Case Name:\n2. Test Objective:\n3. Preconditions:\n4. Inputs:\n##\nTest data :\n   User Details:\n      a. Username:\n      b. Password:\n\n   Post Details:\n      a. Post Type:\n      b. Location/Restaurant:\n      c. Audience:\n      d. Post Content:\n\n   Platforms:\n##\nTest steps :\n   a. For iOS:\n   b. For android:\n   c. For web:\n##\nExpected result :\n  a. For iOS:\n  b. For android:\n  c. For web:\n"
            },
            {
            "role": "user",
            "content": "how to upload a photo in Instagram"
            },
            {
            "role": "assistant",
            "content": "Metadata:\n1. Test Case Name: Upload Photo in Instagram\n2. Test Objective: To verify the functionality of uploading a photo in Instagram\n3. Preconditions: User must have a valid Instagram account and be logged in\n4. Inputs: Photo file to be uploaded\n##\nTest data:\nUser Details:\na. Username: testuser\nb. Password: testuser@123\n\nPost Details:\na. Post Type: Photo\nb. Location/Restaurant: [optional]\nc. Audience: Public\nd. Post Content: [optional]\n\nPlatforms: \niOS\nAndroid\nWeb\n##\nTest steps:\na. For iOS:\n1. Launch the Instagram app on the iOS device.\n2. Login with valid credentials.\n3. Tap on the \"+\" button at the bottom center of the screen.\n4. Select \"Photo\" from the options.\n5. Choose a photo from the device's gallery.\n6. Apply filters or edit the photo if desired.\n7. Add a caption or other optional details.\n8. Tap on the \"Next\" button.\n9. Select the desired audience for the post (Public, Followers, Close Friends, etc.).\n10. Add location or tag people if desired.\n11. Tap on the \"Share\" button to upload the photo.\n\nb. For Android:\n1. Open the Instagram app on the Android device.\n2. Log in using valid credentials.\n3. Tap on the \"+\" button at the bottom center of the screen.\n4. Choose \"Photo\" from the options.\n5. Select a photo from the device's gallery.\n6. Apply filters or edit the photo if required.\n7. Add a caption or other optional details.\n8. Tap on the \"Next\" button.\n9. Choose the desired audience for the post (Public, Followers, Close Friends, etc.).\n10. Add location or tag people if desired.\n11. Tap on the \"Share\" button to upload the photo.\n\nc. For web:\n1. Open a web browser and go to the Instagram website.\n2. Log in using valid credentials.\n3. Click on the \"+\" button at the top of the page.\n4. Select \"Upload Photo\" from the options.\n5. Choose a photo from the device's file explorer.\n6. Apply filters or edit the photo if desired.\n7. Add a caption or other optional details.\n8. Click on the \"Next\" button.\n9. Choose the desired audience for the post (Public, Followers, Close Friends, etc.).\n10. Add location or tag people if desired.\n11. Click on the \"Share\" button to upload the photo.\n##\nExpected result:\na. For iOS:\n- The Instagram app should successfully upload the selected photo with the provided caption and other details.\n- The uploaded photo should be visible on the user's profile and in the Instagram feed.\n\nb. For Android:\n- The Instagram app should successfully upload the selected photo with the provided caption and other details.\n- The uploaded photo should be visible on the user's profile and in the Instagram feed.\n\nc. For web:\n- The Instagram website should successfully upload the selected photo with the provided caption and other details.\n- The uploaded photo should be visible on the user's profile and in the Instagram feed."
            },
            {
            "role": "user",
            "content": "Create and Verify Check-in Post with Friends Audience"
            },
            {
            "role": "assistant",
            "content": "Metadata:\n1. Test Case Name: Create and Verify Check-in Post with Friends Audience\n2. Test Objective: To verify the functionality of creating a check-in post with the audience set to Friends\n3. Preconditions: User must have a valid Instagram account and be logged in\n4. Inputs: Location/Restaurant details\n##\nTest data:\nUser Details:\na. Username: testuser\nb. Password: testuser@123\n\nPost Details:\na. Post Type: Check-in\nb. Location/Restaurant: Central Park, New York\nc. Audience: Friends\nd. Post Content: [optional]\n\nPlatforms: \niOS\nAndroid\nWeb\n##\nTest steps:\na. For iOS:\n1. Launch the Instagram app on the iOS device.\n2. Login with valid credentials.\n3. Tap on the \"+\" button at the bottom center of the screen.\n4. Select \"Check-in\" from the options.\n5. Search for \"Central Park, New York\" in the location search bar.\n6. Tap on the correct location from the search results.\n7. Add a caption or other optional details.\n8. Tap on the \"Next\" button.\n9. Select the \"Friends\" audience for the post.\n10. Tap on the \"Share\" button to create the check-in post.\n\nb. For Android:\n1. Open the Instagram app on the Android device.\n2. Log in using valid credentials.\n3. Tap on the \"+\" button at the bottom center of the screen.\n4. Choose \"Check-in\" from the options.\n5. Search for \"Central Park, New York\" in the location search bar.\n6. Tap on the correct location from the search results.\n7. Add a caption or other optional details.\n8. Tap on the \"Next\" button.\n9. Choose the \"Friends\" audience for the post.\n10. Tap on the \"Share\" button to create the check-in post.\n\nc. For web:\n1. Open a web browser and go to the Instagram website.\n2. Log in using valid credentials.\n3. Click on the \"+\" button at the top of the page.\n4. Select \"Check-in\" from the options.\n5. Search for \"Central Park, New York\" in the location search bar.\n6. Click on the correct location from the search results.\n7. Add a caption or other optional details.\n8. Click on the \"Next\" button.\n9. Choose the \"Friends\" audience for the post.\n10. Click on the \"Share\" button to create the check-in post.\n##\nExpected result:\na. For iOS:\n- The Instagram app should successfully create a check-in post with the selected location and the audience set to Friends.\n- The check-in post should be visible on the user's profile and in the Instagram feed.\n- Only the user's friends should be able to see the check-in post.\n\nb. For Android:\n- The Instagram app should successfully create a check-in post with the selected location and the audience set to Friends.\n- The check-in post should be visible on the user's profile and in the Instagram feed.\n- Only the user's friends should be able to see the check-in post.\n\nc. For web:\n- The Instagram website should successfully create a check-in post with the selected location and the audience set to Friends.\n- The check-in post should be visible on the user's profile and in the Instagram feed.\n- Only the user's friends should be able to see the check-in post."
            }
        ],
        temperature=0.17,
        max_tokens=4181,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    return response['choices'][0]['message']['content']
@app.route('/uploadBRDMeta', methods=['POST'])
def upload_file_BRDMeta():
    uploaded_file = request.files.get('file')
    text_data = request.form.get('file')
    
    from_where = ""
    test_senerios = []
    streams = []
    prioritys = []
    usecases = []
    
    if not uploaded_file and not text_data:
        return "No file or text data uploaded or incorrect key used for file."
    
    if uploaded_file:
        key = uploaded_file.filename.split('.')
        if key[1] == 'pdf':
            pass
        
        elif key[1] == 'xlsx':
            workbook = openpyxl.load_workbook(uploaded_file)
            sheet = workbook.active
            data = []
            for row in sheet.iter_rows(min_row=1, values_only=True):
                data.append(row)
            columns = data[0]
            df = pd.DataFrame(data[1:], columns=columns)
            
            test_scenario_column = df["Test Scenario"].dropna()
            test_senerios = test_scenario_column.reset_index(drop=True).tolist()

            stream_column = df["Stream"].dropna()
            streams = stream_column.reset_index(drop=True).tolist()
            
            Priority_column = df["Priority"].dropna()
            prioritys = Priority_column.reset_index(drop=True).tolist()
            
            useCase_column = df["Use Case/Feature"].dropna()
            usecases = useCase_column.reset_index(drop=True).tolist()
            
            from_where = "file"
        elif key[1] == 'txt':
            pass

    if text_data:
        test_senerios = [text_data]
        from_where = "text"
        
    try:
            # Attempt to create the table
        table = dynamodb.create_table(
            TableName=table_name_BRDMeta,
            KeySchema=[
                {
                    'AttributeName': 'IP_Test_Senerio',
                    'KeyType': 'HASH'
                },
                {
                    'AttributeName': 'IP_Test_Senerio_id',
                    'KeyType': 'RANGE' 
                }
                ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'IP_Test_Senerio',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'IP_Test_Senerio_id',
                    'AttributeType': 'S'  
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )

            # Wait until the table is active
        table.meta.client.get_waiter('table_exists').wait(
            TableName=table_name_BRDMeta,
            WaiterConfig={
                'Delay': 1,
                'MaxAttempts': 25
            }
         )
        print('Table created:', table.table_name)

    except dynamodb.meta.client.exceptions.ResourceInUseException:
        table = dynamodb.Table(table_name_BRDMeta)
        print('Table loaded:', table.table_name)
    

    response = table.scan(Select='COUNT')
    item_count = response['Count']
    print(item_count)
    
    atTime = datetime.now()
        
    for test_senerio,stream,priority,usecase in zip_longest(test_senerios,streams,prioritys,usecases):
        try:
            item_count += 1
            start_time = time.time()
            response = get_functional_test_response_meta(test_senerio)
            metadata,test_data,test_steps,expected_result= response.split("##")
            end_time = time.time()
            total_time = round(end_time - start_time)
            print(metadata)
            print(test_data)
            print(test_steps)
            print(expected_result)
        except Exception as e:
            print(str(e))
            result_functional = "some error is occuring while we are stabling connection with openai\n check openai api\check connection"
        

        # Define the items (data) you want to insert
        items = [
            {
        "IP_Test_Senerio": {
            "S": test_senerio
            },
        "IP_Test_Senerio_id": {
            "S": f"TS-{item_count}"
            },
        "IP_stream": {
            "S": str(stream)
            },
        "IP_priority": {
            "S": str(priority)
            },
        "IP_useCase": {
            "S": str(usecase)
            },
        "Gen_metaData": {
            "S": metadata
                },
        "Gen_testData": {
            "S": test_data
        },
        "Gen_testResult": {
            "S": expected_result
        },
        "Gen_testSteps": {
            "S": test_steps
        },
        "totalTime": {
            "S": str(total_time)
        },
        "fromWhere": {
            "S": from_where
        },
        "atTime":{
           "S": str(atTime) 
        }
        }
            ]

        # Create a list of item requests for batch writing
        item_requests = []
        for item in items:
            item_requests.append({
                'PutRequest': {
                    'Item': item
                }
            })
        dynamodb_client = boto3.client('dynamodb', region_name=region_name)
        response = dynamodb_client.batch_write_item(RequestItems={table_name_BRDMeta: item_requests})
        print(response)
        
    return jsonify({'message': 'File uploaded successfully'})


#-----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=8000)



   


