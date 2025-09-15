from pathlib import Path
import subprocess
import json 
from python.datahubclient import DataHubClient 
import os

READ_FROM_ENV = True


if not READ_FROM_ENV:
    DATAHUB_HOST = 'https://api-gate.efra.maize.io/datahub'
    DATAHUB_API_KEY = 'cnr-YCfdx54Fv58HWVOOkyh7LRUESUezesOC4UWiJPsrFzC12fSkTQ6Pc6x5dqpQ'
else:
    DATAHUB_API_KEY = os.getenv('DATAHUB_API_KEY')
    DATAHUB_HOST = os.getenv('DATAHUB_HOST')




DOCSSTORE_FILE_NAME = "docsstore.json"
QUERIESSTORE_FILE_NAME = "queriesstore.json"

def get_os_vars_input():
    if not READ_FROM_ENV:
        INPUT_DATASET_FULL_NAME="CNR/msmarco-passage_trec-dl-2019_judged-small/1"
    else:
        INPUT_DATASET_FULL_NAME = os.getenv('INPUT_DATASET')
        
    print("INPUT_DATASET_FULL_NAME", INPUT_DATASET_FULL_NAME)
    INPUT_NAMESPACE = INPUT_DATASET_FULL_NAME.split('/')[0]
    INPUT_DATASET = INPUT_DATASET_FULL_NAME.split('/')[1]
    INPUT_VERSION = INPUT_DATASET_FULL_NAME.split('/')[2]
    
    return INPUT_NAMESPACE, INPUT_DATASET, INPUT_VERSION

def get_os_vars_output():
    if not READ_FROM_ENV:
        OUTPUT_DATASET_FULL_NAME="CNR/emvb/1.0"
    else:
        OUTPUT_DATASET_FULL_NAME = os.getenv('OUTPUT_DATASET')
    OUTPUT_NAMESPACE = OUTPUT_DATASET_FULL_NAME.split('/')[0]
    OUTPUT_DATASET = OUTPUT_DATASET_FULL_NAME.split('/')[1]
    OUTPUT_VERSION = OUTPUT_DATASET_FULL_NAME.split('/')[2]
    if not READ_FROM_ENV:
        OUTPUT_DESC = "emvb test"
        OUTPUT_TAGS = ["emvb", "test"]
    else:
        OUTPUT_DESC = os.getenv('OUTPUT_DESC')
        OUTPUT_TAGS = ["emvb", "retrieval"]
    return OUTPUT_NAMESPACE, OUTPUT_DATASET, OUTPUT_VERSION, OUTPUT_DESC, OUTPUT_TAGS



def main():
    
    INPUT_NAMESPACE, INPUT_DATASET, INPUT_VERSION = get_os_vars_input()    
    print("INPUT_NAMESPACE:", INPUT_NAMESPACE)
    print("INPUT_DATASET:", INPUT_DATASET)
    print("INPUT_VERSION:", INPUT_VERSION)
    
    MODEL_NAMESPACE = os.getenv('MODEL_NAMESPACE')
    MODEL_NAME = os.getenv('MODEL_NAME')
    MODEL_VERSION = os.getenv('MODEL_VERSION')
    
    dh_client = DataHubClient(DATAHUB_HOST, DATAHUB_API_KEY)
    
    metadata = dh_client.get_dataset_data(INPUT_NAMESPACE, INPUT_DATASET, INPUT_VERSION)
    
    print("Loading metadata for compatibilty...")   
    print("Metadata contains:")
    print(metadata)
    
    
    OUTPUT_PATH_ELAPSED_TIME="./elapsed_times.json"
    OUTPUT_PATH_MAIN="./main.json"
    OUTPUT_PATH_RUN="./run.json"
    
    exec_path = "./build/perf_emvb"
    print("Executing:", exec_path)

    process = subprocess.Popen(exec_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    stdout, _ = process.communicate()

    if stdout:
        print(stdout.decode())  
    if process.returncode != 0:
        print("Error: process returned non-zero exit code")
        exit(1)


    try:
        with open(OUTPUT_PATH_RUN, "r") as f:
            run = json.load(f)
        with open(OUTPUT_PATH_ELAPSED_TIME, "r") as f:
            elapsed_times = json.load(f)
        
        #print(run)
        #print(elapsed_times)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error reading JSON files:", e)
        exit(1)
        
    
    OUTPUT_NAMESPACE, OUTPUT_DATASET, OUTPUT_VERSION, OUTPUT_DESC, OUTPUT_TAGS = get_os_vars_output()
    
    print("OUTPUT_NAMESPACE:", OUTPUT_NAMESPACE)
    print("OUTPUT_DATASET:", OUTPUT_DATASET)
    print("OUTPUT_VERSION:", OUTPUT_VERSION)
    print("OUTPUT_DESC:", OUTPUT_DESC)
    print("OUTPUT_TAGS:", OUTPUT_TAGS)
    
    with open(OUTPUT_PATH_MAIN, 'w') as f:
        to_upload_main = [
            {
                "id": 0,
                "original_dataset": Path(OUTPUT_PATH_RUN).name
            },
            {
                "id": 1,
                "elapsed_times": Path(OUTPUT_PATH_ELAPSED_TIME).name
            }
        ]
        json.dump(to_upload_main, f)
    
    upload_status = dh_client.create_dataset(
        namespace=OUTPUT_NAMESPACE,
        name=OUTPUT_DATASET,
        version=OUTPUT_VERSION,
        description=OUTPUT_DESC,
        file=OUTPUT_PATH_MAIN,
        tags=["indexing", "emvb"]
        )
    
   
    print(upload_status)

if __name__ == "__main__":
    main()
