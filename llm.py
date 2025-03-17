from qwen_agent.llm import get_chat_model

llm_cfg = {
            # Use the model service provided by DashScope:
            # 'model_type': 'qwen_dashscope',
            # 'model': 'qwen-max',
            # 'model_server': 'dashscope',
            
            # 使用Ollama部署的本地服务
            'model': 'qwen2.5:32b',
            'model_server': 'http://127.0.0.1:11434/v1',
            'api_key': 'EMPTY',
            'generate_cfg': {
                'top_p': 0.8
            }
          }
llm = get_chat_model(llm_cfg)
messages = [{
    'role': 'user',
    'content': "What's the weather like in San Francisco?"
}]
functions = [{
    'name': 'get_current_weather',
    'description': 'Get the current weather in a given location',
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description':
                'The city and state, e.g. San Francisco, CA',
            },
            'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit']
            },
        },
        'required': ['location'],
    },
}]

# 此处演示流式输出效果
responses = []
for responses in llm.chat(messages=messages,
                          functions=functions,
                          stream=True):
    print(responses)