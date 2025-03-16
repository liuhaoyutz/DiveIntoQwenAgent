import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

"""
定义工具`my_image_gen`。

@register_tool修饰符定义在qwen_agent/tools/base.py文件中，它主要做2件事情：
1、将MyImageGen.name设置为'my_image_gen'
2、将MyImageGent注册到一个全局字典TOOL_REGISTRY中，即TOOL_REGISTRY[my_image_gent]=MyImageGen。这样系统就知道有这样一个tool了。

MyImageGen继承自抽象基类BaseTool，该抽象基类要求其所有子类必须实现call方法，即该Tool的工作流程。
MyImageGen从抽象基类BaseTool继承了name, description, parameters三个属性。
"""
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # `description`告诉Agent该tool的功能。
    description = 'AI painting (image generation) service, input text description, and return the image URL drawn based on text information.'
    # `parameters`告诉Agent该tool需要的输入参数。
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Detailed description of the desired image content, in English',
        'required': True
    }]

    # 抽象基类BaseTool要求所有子类都必须实现call方法。
    def call(self, params: str, **kwargs) -> str:
        # `params`是由LLM生成的tool的输入参数。
        prompt = json5.loads(params)['prompt']  # params 是一个JSON格式的字符串，使用 json5.loads() 将其转换成Python对象，并从中提取名为 prompt 的值。
        prompt = urllib.parse.quote(prompt)  # 使用urllib.parse.quote对prompt进行URL编码。因为prompt可能包含特殊字符，如果不进行编码，可能会导致生成的URL无效。

        # json5.dumps将包含URL的字典序列化为JSON格式的字符串。
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


# 配置Agent要使用的LLM
llm_cfg = {
    # 使用通义千问DashScope API
    #'model': 'qwen-max',
    #'model_server': 'dashscope',
    #'api_key': 'YOUR_DASHSCOPE_API_KEY',

    # 使用Ollama部署的本地服务
    'model': 'qwen2.5:32b',
    'model_server': 'http://127.0.0.1:11434/v1',
    'api_key': 'EMPTY',

    # (Optional) LLM超参数
    'generate_cfg': {
        'top_p': 0.8
    }
}

# 定义系统提示词，用英文是因为https://pollinations.ai/网站对英文处理的更好。
system_instruction = '''You are a helpful assistant.
After receiving the user's request, you should:
- first draw an image and obtain the image url,
- then run code `request.get(image_url)` to download the image,
- and finally select an image operation from the given document to process the image.
Please show the image using `plt.show()`.'''

# 定义工具列表，my_image_gen是我们自定义的工具，用于通过https://pollinations.ai/网站生成图片。code_interpreter是一个预定义工具，用于执行python代码。
tools = ['my_image_gen', 'code_interpreter']  # `code_interpreter` is a built-in tool for executing code.

# 定义要传递的文件。该文件是一个教程，包含怎样利用Python代码从网站下载图片，并对图片进行一些处理（如旋转）。该文件的内容将被向量化并存入向量数据库，以供RAG检索。
files = ['./resource/doc.pdf']  # Give the bot a PDF file to read.

"""
基于Assistant类创建能够生成图片并对图片进行处理的Agent。

Assistant类定义在qwen_agent/agents/assistant.py文件中，它继承自FnCallAgent类，进而继承Agent类。Agent是一个ABC抽象基类。

Agent类定义在qwen_agent/agent.py文件中，其作用是：
A base class for Agent.
An agent can receive messages and provide response by LLM or Tools.
Different agents have distinct workflows for processing messages and generating responses in the `_run` method.

Agent类实现了run方法，该方法会调用要求各个子类实现的_run方法，所以：
FnCallAgent类实现了_run方法，调用LLM（_call_llm），进而调用tool（_call_tool）。
Assistant类实现了_run方法，该方法从RAG检索出相关knowledge（初始化时已经从传递的文件中读取knowledge，保存到了向量数据库中），然后调用其父类FnCallAgent的_run方法，进而调用LLM和tool。
"""
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)

# 在命令行模式下使用Agent
for i in range(2):
    messages = []  # This stores the chat history.
    
    # 输入用户提示词，例如："draw a dog and rotate it 90 degrees."
    query = input('user query: ')
    
    # 将用户提示词加入到消息列表中。
    messages.append({'role': 'user', 'content': query})
    
    # 执行Agent生成图片并进行处理。
    response = []
    for response in bot.run(messages=messages):
        # Streaming output.
        print('bot response:')
        pprint.pprint(response, indent=2)  # pprint是专门用来美化打印Python数据结构的工具，indent=2参数指定了缩进级别为2个空格，提高（如嵌套的字典或列表）可读性。
    
    # Append the bot responses to the chat history.
    messages.extend(response)

# 通过WebUI使用Agent
from qwen_agent.gui import WebUI

WebUI(bot).run()