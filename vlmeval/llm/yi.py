import os.path as osp
import sys
import warnings

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlmeval.smp import get_cache_path, splitlen

from .utils import rank0_print

"""
You can perform inference of Yi-VL through the following steps:
1. clone the repo https://github.com/01-ai/Yi to path-to-Yi
2. set up the environment and install the required packages in path-to-Yi/VL/requirements.txt
3. set Yi_ROOT in vlmeval/config.py 
    Yi_ROOT = path-to-Yi

You are all set now! To run a demo for Yi-VL:
```python
from vlmeval import *
model = supported_VLM['Yi_VL_6B']()
model.generate('apple.jpg', 'What is in this image?')
```
To run evaluation for Yi-VL, use `python run.py --model Yi_VL_6B --data {dataset_list}`
"""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class Yi:

    INSTALL_REQ = True

    def __init__(self,
                 model_path='01-ai/Yi-34B',
                 root=None,
                 **kwargs):
        self.is_llm = True
        self.root = osp.join(root, 'VL')
        sys.path.append(self.root)

        if splitlen(model_path, '/') == 2 and not osp.exists(model_path):
            if get_cache_path(model_path) is None:
                snapshot_download(repo_id=model_path)

        disable_torch_init()

        load_kwargs = {"device_map": "cpu"}
        load_kwargs["torch_dtype"] = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **load_kwargs
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.eval().cuda()

        kwargs_default = dict(
            do_sample=kwargs.get('do_sample', False),
            temperature=kwargs.get('temperature', 0),
            num_beams=kwargs.get('num_beams', 1),
            conv_mode="text_default",
            top_p=kwargs.get('num_beams', None),
            max_new_tokens=kwargs.get('max_new_tokens', 1024))
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        from llava.conversation import conv_templates

        from .utils import KeywordsStoppingCriteria

        qs = prompt
        conv = conv_templates[self.kwargs['conv_mode']].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer(prompt).input_ids
        input_ids = (
            torch.tensor(input_ids, dtype=torch.long)
            .unsqueeze(0)
            .cuda()
        )

        # stop_str = conv.sep
        stop_str = '<|im_end|>'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids)
        self.model = self.model.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                do_sample=self.kwargs['do_sample'],
                temperature=self.kwargs['temperature'],
                top_p=self.kwargs['top_p'],
                num_beams=self.kwargs['num_beams'],
                stopping_criteria=[stopping_criteria],
                max_new_tokens=self.kwargs['max_new_tokens'],
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
