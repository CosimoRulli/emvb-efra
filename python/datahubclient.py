import sys
import requests
import logging
import os

SLEEP_TIME: int = 120  # seconds sleeptime between api calls for testing data export
MAX_RETRIES: int = 15  # max number of api calls for testing data export

log = logging.getLogger(__name__)

class DataHubClient:
    def __init__(self, host: str, api_key: str):
        self.host = host
        self.auth_header = {'X-API-KEY': api_key}
        self.list_url = host + '/api/v1/datasets/list'

    def dataset_url(self, namespace: str, name: str, version: str):
        return self.host + f'/api/v1/datasets/{namespace}/{name}/{version}'

    def list_datasets(self, start: int = 0, size: int = 10, namespace: str = None,
                      name: str = None, version: str = None):
        params = {'start': start, 'rows': size}
        if namespace is not None and len(namespace) > 0:
            params['namespace'] = namespace
        if name is not None and len(name) > 0:
            params['name'] = name
        if version is not None and len(version) > 0:
            params['version'] = version
        response = requests.get(self.list_url, params=params, headers=self.auth_header)
        if response.ok:
            ddict = response.json()
            return ddict['count'], ddict['results']
        raise Exception("Error contacting EFRA DataHub, got: " + str(response.status_code))

    def get_dataset_data(self, namespace: str, name: str, version: str) -> str:
        url = self.dataset_url(namespace, name, version) + '/data'
        response = requests.get(url, headers=self.auth_header)
        if response.ok:
            return response.text
        raise Exception("Error contacting EFRA DataHub, got: " + str(response.status_code))

    def get_dataset_object_file(self, namespace: str, name: str, version: str, file: str) -> str:
        url = self.dataset_url(namespace, name, version) + f'/{file}/file'
        response = requests.get(url, headers=self.auth_header)
        if response.ok:
            return response.text
        raise Exception("Error contacting EFRA DataHub, got: " + str(response.status_code))

    def create_dataset(self, namespace: str, name: str, version: str, file: str, description: str,
                       tags: list[str] = []) -> bool:
        url = self.dataset_url(namespace, name, version)
        parameters = {'description': description}
        if len(tags) > 0:
            parameters['tags[]'] = tags
        files = {
            'file': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')
        }
        response = requests.post(url, data=parameters, files=files, headers=self.auth_header)
        print(response)
        if response.ok:
            return True
        return False

    # def create_dataset_with_objects(self, namespace: str, name: str, version: str, file: str, objects: list[str],
    #                                 description: str, tags: list[str] = []) -> bool:
    #     url = self.dataset_url(namespace, name, version)
    #     parameters = {'description': description}
    #     if len(tags) > 0:
    #         parameters['tags[]'] = tags
    #     files = {
    #         'file': None
    #     }
    #     response = requests.post(url, data=parameters, files=files, headers=self.auth_header)
    #     if response.ok:
    #         for obj in objects:
    #             id = os.path.basename(obj)
    #             obj_files = {
    #                 'file': (id, open(obj, 'rb'), 'application/octet-stream')
    #             }
    #             obj_url = url + f'/{id}'
    #             obj_response = requests.post(obj_url, data={}, files=obj_files, headers=self.auth_header)
    #             if not obj_response.ok:
    #                 self.delete_dataset(namespace, name, version)
    #                 return False
    #         files = {
    #             'file': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')
    #         }
    #         finalize_url = url + '/finalize'
    #         response = requests.post(finalize_url, data={}, files=files, headers=self.auth_header)
    #         if response.ok:
    #             return True
    #     return False

    def create_dataset_with_objects(self, namespace: str, name: str, version: str, file: str, objects: list[str], description: str, tags: list[str] = []) -> bool:
        url = self.dataset_url(namespace, name, version)
        print(f"[DEBUG] Creating dataset with objects at URL: {url}")

        parameters = {'description': description}
        if len(tags) > 0:
            parameters['tags[]'] = tags
        print(f"[DEBUG] Parameters for dataset creation: {parameters}")

        files = {
            'file': None
        }
        response = requests.post(url, data=parameters, files=files, headers=self.auth_header)
        print(f"[DEBUG] Initial dataset creation response status: {response.status_code}")

        if response.ok:
            for obj in objects:
                id = os.path.basename(obj)
                obj_files = {
                    'file': (id, open(obj, 'rb'), 'application/octet-stream')
                }
                obj_url = url + f'/{id}'

                try:
                    obj_response = requests.post(obj_url, data={}, files=obj_files, headers=self.auth_header, timeout=120)
                except requests.exceptions.Timeout as e:
                    print(f"[ERROR] Timeout uploading object '{id}': {e}")
                    continue 


            files = {
                'file': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')
            }
            finalize_url = url + '/finalize'
            print(f"[DEBUG] Finalizing dataset creation at URL: {finalize_url}")
            response = requests.post(finalize_url, data={}, files=files, headers=self.auth_header)
            print(f"[DEBUG] Finalization response status: {response.status_code}")

            if response.ok:
                print("[DEBUG] Dataset with objects created and finalized successfully.")
                return True

        print("[ERROR] Failed to create dataset with objects.")
        return False
    
    def delete_dataset(self, namespace: str, name: str, version: str) -> bool:
        response = requests.delete(self.dataset_url(namespace, name, version), headers=self.auth_header)
        if response.ok:
            return True
        return False
