import os
import os.path as osp
import sys
from abc import abstractproperty

import torch
from PIL import Image

from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt


class LLaVA(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self,
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        sys.path.append('./thirdparty/LLaVA')
        try:
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
        except:
            warnings.warn("Please install llava before using LLaVA")
            sys.exit(-1)

        assert osp.exists(model_pth) or splitlen(model_pth) == 2

        if model_pth == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_pth == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_pth)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_pth,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except Exception as e:
            print(e)
            if 'ShareGPT4V' in model_pth:
                import llava
                warnings.warn(
                    f'Please manually remove the encoder type check in {llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        kwargs_default = dict(do_sample=True, temperature=0.2,
                              max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if (
            'hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += "\n请直接回答选项字母。" if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += "\n请直接回答问题。" if cn_string(
                prompt) else "\nAnswer the question directly."

        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
        from llava.constants import (DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN,
                                     DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (KeywordsStoppingCriteria, process_images,
                                    tokenizer_image_token)
        image = Image.open(image_path).convert('RGB')
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to(
            'cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[
                                             stopping_criteria], **self.kwargs)
        output = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:]).strip().split("</s>")[0]
        return output
