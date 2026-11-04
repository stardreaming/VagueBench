"""Optional image/form transport retained for custom model servers."""
import os
import requests


class CustomModel:
    def __init__(self, model_path, language_only):
        self.model_path = model_path
        self.language_only = language_only
        self.model_type = 'custom'
        self.server_url = os.environ.get('server_url')
        if not self.server_url:
            raise ValueError('Set server_url for the custom model backend')

    def respond(self, prompt, obs=None):
        with open(obs, 'rb') as image:
            response = requests.post(self.server_url, files={'image': image},
                                     data={'sentence': prompt}, timeout=120)
        response.raise_for_status()
        return response.json()['response']

