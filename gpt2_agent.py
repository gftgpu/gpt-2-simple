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
SEE : https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce

本代码主要改编自： https://github.com/minimaxir/gpt-2-simple/blob/master/gpt_2_simple/gpt_2.py

"""
import json
import os
import logging

import tensorflow as tf
import numpy as np
from tensorflow import Session
from typing import Dict

from gs_research_workflow.nlp.agents.gpt2.gpt_2_simple.src.encoder import Encoder
from gs_research_workflow.nlp.agents.gpt2.gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gs_research_workflow.nlp.agents.gpt2.gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gs_research_workflow.nlp.agents.gpt2.gpt_2_simple.src.accumulate import AccumulatingOptimizer


from gs_research_workflow.nlp.agents.gpt2 import gpt_2_simple as gpt2
from tensorflow.core.protobuf import rewriter_config_pb2

logger = logging.getLogger(__name__)

# ---- 先忽略这部分注释的代码 -----
# model_name = "117M"
# gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/117M/

# sess1 = gpt2.start_tf_sess()

# gpt2.load_gpt2(sess1)
# single_text = gpt2.generate(sess, prefix=raw_text,return_as_list=True)[0]
# gpt2.generate_text(sess1, raw_text)
# print("*"*30)
# print(single_text)


class GPT2Agent:
    def __init__(self, model_name: str, prediction_mode: bool):
        self.model_name: str = model_name
        """模型名称，可以用来区分不同的 fine_tuning 版本，以及不同的参数级别版本"""

        self.prediction_mode: bool = prediction_mode
        """是否在 prediction_mode 是一个 prediction 的参数项"""

        self._tf_sess: Session = self._start_tf_sess()
        self._encoder: Encoder = None
        self._checkpoint_path: str = None

        self._sample_seq_fetches: Dict = None
        """??? 具体含义不了解
        参考 https://github.com/openai/gpt-2/blob/master/src/sample.py#L25 的返回值 """

        self._x_placeholder = None

        # 这里是一些没有放在 'hparams.json' 中的 hyper_parameters，在 prediction step 可以修改
        # SEE: https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py#L11

        self._seed: int = None
        """Integer seed for random number generators, fix seed to reproduce results"""

        self._nsamples: int = 1
        """Number of samples to return total"""

        self._batch_size: int = 1
        """Number of batches (only affects speed/memory).  Must divide nsamples."""

        self._temperature: float = 1
        """Float value controlling randomness in boltzmann
            distribution. Lower temperature results in less random completions. As the
            temperature approaches zero, the model will become deterministic and
            repetitive. Higher temperature results in more random completions."""

        self._top_k: int = 0
        """Integer value controlling diversity. 1 means only 1 word is
            considered for each step (token), resulting in deterministic completions,
            while 40 means 40 words are considered at each step. 0 (default) is a
            special setting meaning no restrictions. 40 generally is a good value."""

        self._generated_token_length: int = None
        """
        Number of tokens in generated text, if None (default), is determined by model hyperparameters
        """

    def _start_tf_sess(self, threads=-1) -> Session:
        """
        Returns a tf.Session w/ config

        考虑以后通用化之后，可以放在 tensorflow 版本的 Agent 基类
        Copy From https://github.com/minimaxir/gpt-2-simple/blob/master/gpt_2_simple/gpt_2.py#L57
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
        if threads > 0:
            config.intra_op_parallelism_threads = threads
            config.inter_op_parallelism_threads = threads

        return tf.Session(config=config)

    def prepare_checkpoint_path(self):
        """
        准备好 model 的 checkpoint 文件内容

        暂时先 HardCode ， 以后会从一个Http(Env) 下载到本地临时文件夹进行缓存
        """
        self._checkpoint_path = "/tmp/laigen/gs_research_workflow/nlp/agents/gpt2/models/117M"

    def load_model(self):
        assert self._checkpoint_path is not None
        assert os.path.exists(self._checkpoint_path)

        hparams = model.default_hparams()
        with open(os.path.join(self._checkpoint_path, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        context = tf.placeholder(tf.int32, [1, None])
        output = model.model(hparams=hparams, X=context)

        ckpt = tf.train.latest_checkpoint(self._checkpoint_path)
        saver = tf.train.Saver(allow_empty=True)
        self._tf_sess.run(tf.global_variables_initializer())

        logger.debug(f"Loading checkpoint:{ckpt}")
        saver.restore(self._tf_sess, ckpt)

        # 准备 encoder 对象等
        assert self._nsamples % self._batch_size == 0

        self._encoder = encoder.get_encoder(self._checkpoint_path)

        if self._generated_token_length is None:
            self._generated_token_length = hparams.n_ctx // 2
        elif self._generated_token_length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
        self._x_placeholder = tf.placeholder(tf.int32, [self._batch_size, None])

        np.random.seed(self._seed)
        tf.set_random_seed(self._seed)

        self._sample_seq_fetches = sample.sample_sequence(
            hparams=hparams, length=self._generated_token_length,
            context=self._x_placeholder,
            batch_size=self._batch_size,
            temperature=self._temperature, top_k=self._top_k)

    def prediction(self, raw_text: str) -> str:
        context_tokens = self._encoder.encode(raw_text)
        generated = 0
        ret_text = ""
        for _ in range(self._nsamples // self._batch_size):
            out = self._tf_sess.run(self._sample_seq_fetches, feed_dict={
                self._x_placeholder: [context_tokens for _ in range(self._batch_size)]
            })[:, len(context_tokens):]
            for i in range(self._batch_size):
                generated += 1
                text = self._encoder.decode(out[i])
                # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                ret_text += text
        return ret_text



if __name__ == "__main__":
    gpt2_agent = GPT2Agent("117M", True)
    # 这里模拟了 agent 的 init 的步骤,以后可以通过 internal msg 的方式进行 trigger
    gpt2_agent.prepare_checkpoint_path()
    gpt2_agent.load_model()

    raw_texts = ["China and the United States have been engaged in a trade war through increasing tariffs and other measures since 2018.",
                 "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."]
    for txt in raw_texts:
        next_txt = gpt2_agent.prediction(txt)
        print("="*40)
        print(f"[{txt}]")
        print(next_txt)

