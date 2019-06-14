# -*- coding: UTF-8 -*-

"""
将 GPT-2 的 pretrained 的模型封装成基于 msg 的 GS Passive Agent。

做这事情的目的：
1) 设计如何对接 download 的 agent
2) 如何封装 agent

一些资料收集和总结体会，将写在这里
SEE : https://docs.google.com/document/d/1JLDHsmXoeHBzkwcJt5TCZLbU-oYpo_wM7pbBVlzxKms/edit#

!!! GPT2 并没有提供 pip install 的方式，暂时先用 copy 代码的方式引入进来，src 目录中的文件来自于
SEE : https://github.com/openai/gpt-2/tree/master/src


决定先试一个 tensorflow 版本的 GPT2 (因为官方版本为 tensorflow )
SEE : https://github.com/minimaxir/gpt-2-simple

"""


from gs_research_workflow.nlp.agents.gpt2 import gpt_2_simple as gpt2

model_name = "117M"
# gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/117M/

sess = gpt2.start_tf_sess()

gpt2.load_gpt2(sess)

single_text = gpt2.generate(sess, return_as_list=True)[0]
print(single_text)
