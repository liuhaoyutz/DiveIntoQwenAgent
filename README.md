# DiveIntoQwenAgent
前置条件：  
安装QwenAgent：pip install -U "qwen-agent[gui,rag,code_interpreter,python_executor]"

image_gen是一个能根据用户prompt生成图片，并对图片进行处理（如旋转）的Agent，生成图片和处理的过程包括通过RAG取得执行用户指令背景知识，调用自定义生成图片工具，调用Python解释器工具。  

执行：  
python image_gen.py  

chess_game是一个五子棋游戏，它演示了多Agent的情况。棋盘，用户，NPC是3个Agent，又通过一个GroupChat Agent来协调它们3个Agent的通话顺序。  

执行：  
python chess_game.py