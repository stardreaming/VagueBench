"""Model transports; credentials are read from the environment at runtime."""
import os

from vaguebench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from vaguebench.planner.planner_utils import convert_format_2claude


class RemoteModel:
    def __init__(self, model_name, model_type='remote', language_only=False, tp=1,
                 provider='auto', base_url=None, api_key_env=None,
                 response_format='json_schema', temperature=0, max_tokens=2048):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.response_format = response_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        if response_format not in {'json_schema', 'json_object', 'text'}:
            raise ValueError('Unknown response format')
        if model_type == 'local':
            from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
            self.generation_config = GenerationConfig
            self.model = pipeline(model_name, backend_config=PytorchEngineConfig(
                session_len=12000, dtype='float16', tp=tp))
            return
        if model_type != 'remote':
            raise ValueError('RemoteModel supports remote or local execution')
        if provider == 'auto':
            name = model_name.lower()
            provider = 'anthropic' if 'claude' in name else 'gemini' if 'gemini' in name else 'openai'
        if provider not in {'openai', 'anthropic', 'gemini', 'dashscope'}:
            raise ValueError('Unknown provider')
        self.provider = provider
        default_keys = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY',
                        'gemini': 'GEMINI_API_KEY', 'dashscope': 'DASHSCOPE_API_KEY'}
        key_name = api_key_env or default_keys[provider]
        key = os.environ.get(key_name)
        if not key:
            raise ValueError(f'Set the {key_name} environment variable before evaluation')
        endpoint = base_url or os.environ.get('VAGUEBENCH_BASE_URL')
        options = {'api_key': key}
        if provider == 'anthropic':
            from anthropic import Anthropic
            if endpoint:
                options['base_url'] = endpoint
            self.model = Anthropic(**options, timeout=120, max_retries=2)
        else:
            from openai import OpenAI
            defaults = {'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/',
                        'dashscope': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}
            endpoint = endpoint or defaults.get(provider)
            if endpoint:
                options['base_url'] = endpoint
            self.model = OpenAI(**options, timeout=120, max_retries=2)

    def respond(self, message_history):
        schema = llm_generation_guide if self.language_only else vlm_generation_guide
        if self.model_type == 'local':
            config = self.generation_config(
                temperature=self.temperature, max_new_tokens=self.max_tokens,
                response_format={'type': 'json_schema', 'json_schema': {
                    'name': 'embodied_planning', 'schema': schema}})
            return self.model(message_history, gen_config=config).text
        if self.provider == 'anthropic':
            response = self.model.messages.create(
                model=self.model_name, messages=convert_format_2claude(message_history),
                temperature=self.temperature, max_tokens=self.max_tokens)
            return '\n'.join(block.text for block in response.content if getattr(block, 'type', None) == 'text')
        options = {}
        if self.response_format == 'json_schema':
            options['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'embodied_planning', 'schema': schema}}
        elif self.response_format == 'json_object':
            options['response_format'] = {'type': 'json_object'}
        response = self.model.chat.completions.create(
            model=self.model_name, messages=message_history, temperature=self.temperature,
            max_tokens=self.max_tokens, **options)
        content = response.choices[0].message.content
        if not content:
            raise ValueError('Model returned no text')
        return content
