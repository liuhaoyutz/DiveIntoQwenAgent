from qwen_agent.agents import GroupChat
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import Message

# 定义multi-agent配置文件。CFGS中定义了3个Agent，一个代表真正玩家，一个代表NPC玩家，一个代表棋盘。
NPC_NAME = '小明'
USER_NAME = '小塘'
CFGS = {
    'background':
        f'一个五子棋群组，棋盘为5*5，黑棋玩家和白棋玩家交替下棋，每次玩家下棋后，棋盘进行更新并展示。{NPC_NAME}下白棋，{USER_NAME}下黑棋。',
    'agents': [
        {
            'name':
                '棋盘',
            'description':
                '负责更新棋盘',
            'instructions':
                '你扮演一个五子棋棋盘，你可以根据原始棋盘和玩家下棋的位置坐标，把新的棋盘用矩阵展示出来。棋盘中用0代表无棋子、用1表示黑棋、用-1表示白棋。用坐标<i,j>表示位置，i代表行，j代表列，棋盘左上角位置为<0,0>。',
            'selected_tools': ['code_interpreter'],
        },
        {
            'name':
                NPC_NAME,
            'description':
                '白棋玩家',
            'instructions':
                '你扮演一个玩五子棋的高手，你下白棋。棋盘中用0代表无棋子、用1黑棋、用-1白棋。用坐标<i,j>表示位置，i代表行，j代表列，棋盘左上角位置为<0,0>，请决定你要下在哪里，你可以随意下到一个位置，不要说你是AI助手不会下！返回格式为坐标：\n<i,j>\n除了这个坐标，不要返回其他任何内容',
        },
        {
            'name': USER_NAME,
            'description': '黑棋玩家',
            'is_human': True
        },
    ],
}

# 定义LLM配置
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

"""
GroupChat类定义在qwen_agent/agents/group_chat.py文件中，其作用是：
This is an agent for multi-agent management.
This agent can accept a list of agents, manage their speaking order, and output the response of each agent.

可以看到，GroupChat类也是一个Agent，它用于管理一组Agent的发言顺序，并输出每个Agent的response。
GroupChat类继承自Agent和MultiAgentHub两个类。

MultiAgentHub类是一个抽象基类，它只定义了3个属性：agents列表，agent_names列表，nonuser_agents列表。

Agent类是一个抽象基类，定义在qwen_agent/agent.py文件中，其作用是：
A base class for Agent.
An agent can receive messages and provide response by LLM or Tools.
Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
Agent类实现了run方法，该方法会调用要求各个子类实现的_run方法。

GroupChat类的__init__方法中，调用_init_agents_from_config方法初始化了CFGS中指定的3个Agent，GroupChat对象bot也是一个Agent，所以一共有4个Agent，bot用来管理其他3个Agent。
GroupChat类的_run方法中，bot与其他3个Agent说话，并返回Agent的回答。
可以看出，GroupChat Agent bot是整个游戏的管理者，负责协调其他3个Agent该做什么。
"""

def test(query: str = '<1,1>'):
    bot = GroupChat(agents=CFGS, llm=llm_cfg)

    messages = [Message('user', query, name=USER_NAME)]
    for response in bot.run(messages=messages):
        print('bot response:', response)

def app_tui():
    bot = GroupChat(agents=CFGS, llm=llm_cfg)

    messages = []
    while True:
        query = input('user question: ')
        messages.append(Message('user', query, name=USER_NAME))
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)

def app_gui():
    bot = GroupChat(agents=CFGS, llm=llm_cfg)

    chatbot_config = {
        'user.name': '小塘',
        'prompt.suggestions': [
            '开始！我先手，落子 <1,1>',
            '我后手，请小明先开始',
            '新开一盘，我先开始',
        ],
        'verbose': True
    }

    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()

if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
