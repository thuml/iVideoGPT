import os
import torch
from accelerate.utils import set_seed
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model

import sys
sys.path.append("/dev/null/iVideogpt/")  # TODO

from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction


def get_tokenizer(tokenizer_init_config):
    assert tokenizer_init_config['vqgan_type'] == 'ctx_vqgan', "we only have CompressiveVQModel now"
    vq_model = CompressiveVQModel.from_pretrained(
        tokenizer_init_config['pretrained_model_name_or_path'], subfolder=None, revision=None, variant=None, use_safetensor=True,
        low_cpu_mem_usage=False, device_map=None,
    )
    if tokenizer_init_config['context_length'] != vq_model.context_length:
        print(
            f"[Warning] pretrained context length of vq_model mismatch, change from {vq_model.context_length} to {tokenizer_init_config['context_length']}")
        vq_model.set_context_length(tokenizer_init_config['context_length'])
    vocab_size = vq_model.num_vq_embeddings + vq_model.num_dyn_embeddings
    if tokenizer_init_config['special_token']:
        vocab_size += 2
    return vq_model, vocab_size


def load_models(model_init_config, tokenizer_init_config):
    tokenizer, vocab_size = get_tokenizer(tokenizer_init_config)

    if model_init_config['config_name']:
        config = AutoConfig.from_pretrained(
            model_init_config['config_name'],
            trust_remote_code=model_init_config['trust_remote_code'],
        )
    else:
        assert False
    config.vocab_size = vocab_size
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_init_config['trust_remote_code'])

    if model_init_config['action_conditioned']:
        # TODO: magic number
        perlude_tokens_num = (256 + 1) * model_init_config['context_length'] - 1
        tokens_per_dyna = 16
        model = HeadModelWithAction(model, action_dim=model_init_config['action_dim'], prelude_tokens_num=perlude_tokens_num,
                                    tokens_num_per_dyna=tokens_per_dyna, context=model_init_config['context_length'],
                                    segment_length=model_init_config['segment_length'], model_type=model_init_config['model_type'],
                                    action_recon=model_init_config['action_recon'], use_context_action=True, all_seq_action=True)

    if model_init_config['lora']:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=model_init_config['lora_r'],
                                 lora_alpha=model_init_config['lora_alpha'],
                                 lora_dropout=model_init_config['lora_dropout'],
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                 "down_proj", "embed_tokens", "lm_head"],  # ! only for llama
                                 )
        model.llm = get_peft_model(model.llm, peft_config)

    state_dict = load_file(os.path.join(model_init_config['pretrained_transformer_path'], 'model.safetensors'))
    model.load_state_dict(state_dict, strict=True)

    return model, tokenizer


class iVideoGPTPredictor:
    def __init__(self,
                 config_name,
                 seed,
                 vqgan_type,
                 pretrained_vqgan_name_or_path,
                 pretrained_transformer_path,
                 action_dim,
                 generate_max_batchsize,
                 decode_max_batchsize,
                 action_recon,
                 lora,
                 lora_r,
                 lora_alpha,
                 lora_dropout,
                 epoch=None):

        model_type = config_name.split('/')[-2]
        if model_type not in ['gpt2', 'llama']:
            assert False, f"model_type {model_type} is not supported."

        tokenizer_init_config = {
            "vqgan_type": vqgan_type,
            "pretrained_model_name_or_path": pretrained_vqgan_name_or_path,
            "context_length": 2,
            "special_token": True,
        }

        model_init_config = {
            "config_name": config_name,
            "trust_remote_code": False,
            "action_conditioned": True,
            "context_length": 2,
            "action_dim": action_dim,
            "segment_length": 12,
            "model_type": model_type,
            "action_recon": action_recon,
            "lora": lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "pretrained_transformer_path": pretrained_transformer_path,
        }

        self.video_predictor_config = {
            "context_length": 2,
            "segment_length": 12,
            "generate_max_batchsize": generate_max_batchsize,
            "decode_max_batchsize": decode_max_batchsize,
        }

        # If passed along, set the training seed now.
        if seed is not None:
            set_seed(seed)

        # needed by vp2
        self.num_context = 2
        self.base_prediction_modality = "rgb"

        # Load models
        self.model, self.tokenizer = load_models(model_init_config, tokenizer_init_config)
        self.model = self.model.to('cuda')
        self.tokenizer = self.tokenizer.to('cuda')

    def close(self):
        pass

    @torch.no_grad()
    def __call__(self, batch):
        context_frames = batch["video"]
        action_seq = batch["actions"]

        context_frames = torch.Tensor(context_frames).to('cuda')
        action_seq = torch.Tensor(action_seq).to('cuda')
        context_frames = context_frames.permute(0, 1, 4, 2, 3)  # change to B,T,C,H,W

        # Input: 2 context frames & T actions
        # Output: Predictions for T future frames
        if self.video_predictor_config['context_length'] != 2 or self.video_predictor_config['segment_length'] != 12:
            assert False, "Only support context_length=2 and segment_length=12."

        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if context_frames.shape[0] > 200:
            tokens_list = []
            for i in range(0, context_frames.shape[0], 200):
                tokens, labels = self.tokenizer.tokenize(
                    torch.cat((context_frames[i:i + 200], torch.zeros_like(context_frames[i:i + 200, 1:])), dim=1),
                    self.video_predictor_config['context_length'])
                tokens_list.append(tokens)
            tokens = torch.cat(tokens_list)

        else:
            tokens, labels = self.tokenizer.tokenize(
                torch.cat((context_frames, torch.zeros_like(context_frames[:, 1:])), dim=1),
                self.video_predictor_config['context_length'])

        gen_input = tokens[:, :self.video_predictor_config['context_length'] * (256 + 1)]  # TODO: magic number
        max_new_tokens = (1 + 16) * (
            self.video_predictor_config['segment_length'] - self.video_predictor_config['context_length']) - 1
        gen_kwargs = {
            # 'do_sample': False,
            'do_sample': True,
            'temperature': 1.0,
            'top_k': 100,
            'max_new_tokens': max_new_tokens,
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_tokens_list = []
            max_batch = self.video_predictor_config['generate_max_batchsize']
            for i in range(0, gen_input.shape[0], max_batch):
                generated_tokens = self.model.generate(
                    gen_input[i:i + max_batch],
                    **gen_kwargs,
                    **({'action': action_seq[i:i + max_batch]}),
                    pad_token_id=50256,  # this is meaningless but supressing warning
                )

                generated_tokens_list.append(generated_tokens)

            all_generated_tokens = torch.cat(generated_tokens_list)

            model_output_list = []
            max_batch = self.video_predictor_config['decode_max_batchsize']
            for i in range(0, gen_input.shape[0], max_batch):
                model_output = self.tokenizer.detokenize(
                    all_generated_tokens[i:i + max_batch], self.video_predictor_config['context_length']).clamp(
                    0.0, 1.0)
                # shape is B,12,3,64,64，will change to B,11,64,64,3 in return statement
                model_output_list.append(model_output)

        output = torch.cat(model_output_list)[:, 1:].permute(0, 1, 3, 4, 2).to(
            torch.float).cpu().numpy()  # change to numpy of B,T,H,W,C
        return {"rgb": output}
