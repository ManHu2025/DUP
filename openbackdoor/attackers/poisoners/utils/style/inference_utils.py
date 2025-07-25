import pickle
import torch

import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset_config import BASE_CONFIG
from .data_utils import update_config, Instance, get_label_dict

from .utils import init_gpt2_model


class GPT2Generator(object):
    def __init__(self, model_path, upper_length="same_10", beam_size=1, top_p=0.0, data_dir=None):
        self.model_path = model_path
        self.args = torch.load("{}/training_args.bin".format(self.model_path))
        self.modify_args(upper_length, beam_size, top_p)
        self.config = BASE_CONFIG
        update_config(self.args, self.config)

        if self.args.global_dense_feature_list != "none":

            self.label_dict, self.reverse_label_dict = get_label_dict(data_dir)

            self.global_dense_features = []
            for gdf in self.args.global_dense_feature_list.split(","):
                with open(
                    "{}/{}_dense_vectors.pickle".format(data_dir, gdf), "rb"
                ) as f:
                    vector_data = pickle.load(f)

                final_vectors = {}
                for k, v in vector_data.items():
                    final_vectors[self.label_dict[k]] = v["sum"] / v["total"]

                self.global_dense_features.append((gdf, final_vectors))

        self.gpt2_model, self.tokenizer = init_gpt2_model(checkpoint_dir=model_path,
                                                          args=self.args,
                                                          model_class=GPT2LMHeadModel,
                                                          tokenizer_class=GPT2Tokenizer)

    def modify_args(self, upper_length, beam_size, top_p):
        args = self.args
        args.upper_length = upper_length
        args.stop_token = "eos" if upper_length == "eos" else None
        args.beam_size = beam_size
        args.num_samples = 1
        args.temperature = 0
        args.top_p = top_p
        args.top_k = 1
        if torch.cuda.is_available():
            args.device = torch.cuda.current_device()
        else:
            args.device = 'cpu'

    def modify_p(self, top_p):
        self.args.top_p = top_p

    def generate_batch(self, contexts, global_dense_features=None, get_scores=False,
                       interpolation=None, top_p=None):
        args = self.args
        tokenizer = self.tokenizer
        instances = []

        if global_dense_features is None:
            global_dense_features = [None for _ in contexts]

        for context, gdf in zip(contexts, global_dense_features):
            if not isinstance(context, str):
                context = str(context)  # 将 text 转换为字符串
            context_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))

            # NOTE - For model_110, use the older version of the code
            # The following code is only compatible with the newer models
            instance = Instance(
                self.args, self.config,
                {"sent1_tokens": context_ids, "sent2_tokens": context_ids}
            )
            instance.preprocess(tokenizer)

            if gdf is not None and self.args.global_dense_feature_list != "none":
                if self.global_dense_features:
                    global_dense_vectors = np.array(
                        [x[1][gdf] for x in self.global_dense_features],
                        dtype=np.float32,
                    )
                else:
                    global_dense_vectors = np.zeros((2, 20), dtype=np.float32)
                    global_dense_vectors[0, gdf["f1_bucket"]] = 1
                    global_dense_vectors[1, gdf["ed_bucket"] + 10] = 1
            else:
                global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

            instance.gdv = global_dense_vectors
            instances.append(instance)

        output, _, scores = self.gpt2_model.generate(
            gpt2_sentences=torch.tensor([inst.sentence for inst in instances]).to(args.device),
            segments=torch.tensor([inst.segment for inst in instances]).to(args.device),
            global_dense_vectors=torch.tensor([inst.gdv for inst in instances]).to(args.device),
            init_context_size=instances[0].init_context_size,
            eos_token_id=tokenizer.eos_token_id,
            get_scores=get_scores,
            interpolation=interpolation,
            top_p=top_p
        )

        all_output = []
        for out_num in range(len(output)):
            instance = instances[out_num]
            curr_out = output[out_num, instance.init_context_size:].tolist()

            if tokenizer.eos_token_id in curr_out:
                curr_out = curr_out[:curr_out.index(tokenizer.eos_token_id)]

            if self.args.upper_length.startswith("same"):
                extra = int(self.args.upper_length.split("_")[-1])
                curr_out = curr_out[:len(instance.sent1_tokens) + extra]

            all_output.append(
                tokenizer.decode(curr_out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            )

        return all_output, scores

    def generate(self, context, global_dense_features=None, get_scores=False,
                 interpolation=None, top_p=None):
        return self.generate_batch([context],
                                   [global_dense_features] if global_dense_features is not None else None,
                                   get_scores=get_scores,
                                   interpolation=interpolation,
                                   top_p=top_p)[0][0]


class QwenGenerator(object):
    def __init__(self, model_path):
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'

        self.model_path = model_path

        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 为 pad_token 设置一个单独的标记
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token_id + 1  # 或者使用一个模型未占用的标记

        # 使用 tokenizer.pad_token 更新模型配置
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, contexts):

        all_output = []
        for prompt in contexts:
            if not isinstance(prompt, str):
                prompt = str(prompt)  # 将text转换为字符串

            messages = [
                {"role": "system", "content": "你是一名中文语法与句式转换专家，请将输入句子转换为双重否定句，确保没有改变句意且符合语法规范，同时保持表达自然流畅。最后，如果不是肯定句，可以添加一些句子相关的描述，并将其改为否定句。"},  # 系统角色消息
                {"role": "user", "content": "请将给出的语句改为双重否定句。"+prompt}  # 用户角色消息
            ]

            # 使用分词器的apply_chat_template方法来格式化消息
            text = self.tokenizer.apply_chat_template(
                messages,  # 要格式化的消息
                tokenize=False,  # 不进行分词
                add_generation_prompt=True  # 添加生成提示
            )

            # 将格式化后的文本转换为模型输入，并转换为PyTorch张量，然后移动到指定的设备
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # 清除past_key_values（模型状态缓存）
            model_inputs['past_key_values'] = None  # 每次都清空缓存，确保从零开始生成

            # 使用model.generate()方法直接生成文本
            # 通过设置max_new_tokens参数控制输出的最大长度
            generated_ids = self.model.generate(
                model_inputs.input_ids,  # 模型输入的input_ids
                max_new_tokens=512,  # 最大新生成的token数量
                attention_mask = model_inputs['attention_mask'],
                past_key_values = None
            )

            # 从生成的ID中提取新生成的ID部分
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # 使用分词器的batch_decode方法将生成的ID解码回文本，并跳过特殊token
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            all_output.append(response)
            # print("prompt:",prompt)
            # print("response:",response)

        return all_output