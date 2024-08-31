import os

import time
import numpy as np
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed

from safetensors.torch import load_file
import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
from accelerate import Accelerator
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from ivideogpt.vq_model import CompressiveVQModel
from ivideogpt.transformer import HeadModelWithAction
from ivideogpt.vq_model import LPIPS


def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()


def plot_img(img, postfix=''):
    cv2.imwrite(f'tmp-img{postfix}.png', img.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)


def get_tokenizer(args):
    if args.vqgan_type == 'ctx_vqgan':
        if not args.load_pretrained_model:
            vq_model = CompressiveVQModel.from_config(args.pretrained_model_name_or_path)
        else:
            vq_model = CompressiveVQModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, use_safetensor=True,
                low_cpu_mem_usage=False, device_map=None,
            )
        if args.context_length != vq_model.context_length:
            print(
                f"[Warning] pretrained context length of vq_model mismatch, change from {vq_model.context_length} to {args.context_length}")
            vq_model.set_context_length(args.context_length)
        vocab_size = vq_model.num_vq_embeddings + vq_model.num_dyn_embeddings
        vocab_size += 2  # special token
    else:
        raise NotImplementedError
    return vq_model, vocab_size


def load_models(args):
    tokenizer, vocab_size = get_tokenizer(args)

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=False,
        )
        if "llama" in args.config_name and args.llama_attn_drop is not None:
            config.attention_dropout = args.llama_attn_drop
    else:
        assert False
    config.vocab_size = vocab_size
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    perlude_tokens_num = (256 + 1) * args.context_length - 1
    tokens_per_dyna = 16
    model = HeadModelWithAction(model, action_dim=args.action_dim, prelude_tokens_num=perlude_tokens_num,
                                tokens_num_per_dyna=tokens_per_dyna, context=args.context_length,
                                segment_length=args.segment_length, model_type=args.config_name.split('/')[-2],
                                reward_prediction=True)
    if args.load_pretrained_model:
        state_dict = load_file(os.path.join(args.pretrained_transformer_path, 'model.safetensors'))
        if args.load_internal_llm:
            model.llm.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

    return model, tokenizer


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class VideoPredictor(nn.Module):
    def __init__(self, device, args) -> None:
        super(VideoPredictor, self).__init__()

        self.args = args
        self.device = device
        self.model, self.tokenizer = load_models(args)
        self.model = self.model.to(device)
        self.tokenizer = self.tokenizer.to(device)

        # prepare for tokenizer training
        self.lpips = LPIPS().to(device).eval()
        if args.selected_params:
            # frozon codebook params
            params = [parameter for name, parameter in self.tokenizer.named_parameters() if 'quantize' not in name]
        else:
            params = list(self.tokenizer.parameters())
        self.tok_optimizer = torch.optim.AdamW(
            params,
            lr=args.tok_lr,
            betas=(args.tok_beta1, args.tok_beta2),
            weight_decay=args.tok_wd,
            eps=1e-8,
        )
        self.tok_scaler = torch.cuda.amp.GradScaler()

        # prepare for model training
        no_decay = []
        if args.embed_no_wd:
            for mn, m in self.model.named_modules():
                for pn, p in m.named_parameters():
                    if pn.endswith('bias') or \
                        (pn.endswith('weight') and isinstance(m, torch.nn.Embedding)) or \
                        (pn.endswith('weight') and isinstance(m, torch.nn.LayerNorm)) or \
                            (pn.endswith('weight') and isinstance(m, LlamaRMSNorm)):
                        fpn = '%s.%s' % (mn, pn) if mn else pn
                        no_decay.append(fpn)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.model_wd,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.model_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.model_lr)
        self.model_scaler = torch.cuda.amp.GradScaler()

    def train(self, batch, update_tokenizer=True, update_model=True):
        start = time.time()
        metrics = {}
        obs, action, reward = batch
        obs = obs.to(self.device) / 255.
        action = action.to(self.device)
        reward = reward.to(self.device)
        if self.args.symlog:
            reward = symlog(reward)

        if update_tokenizer:
            metrics.update(self.update_tokenizer(self.args, obs))
        if update_model:
            metrics.update(self.update_model(self.args, obs, action, reward))
        metrics.update({'model_update_time': time.time() - start})
        return metrics

    def update_tokenizer(self, args, obs):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            self.tok_optimizer.zero_grad()
            with torch.no_grad():
                B, T, C, H, W = obs.shape
                frame_pixel_values = obs
                num_target_frames = min((T - args.context_length), args.max_target_frames)
                if num_target_frames == (T - args.context_length):
                    target = frame_pixel_values[:, args.context_length:].reshape(
                        B * (T - args.context_length), C, H, W)  # B*(T-t), C, H, W
                else:
                    target = frame_pixel_values[:, args.context_length:]
                    # random select frames, to handle OOM
                    target = target[:, torch.randperm(target.size(1))[:num_target_frames]]
                    target = target.reshape(B * num_target_frames, C, H, W)  # B*(T-t), C, H, W

                if args.context_length > 1:
                    reference_single = frame_pixel_values[:, :args.context_length].reshape(-1, C, H, W)  # B*t, C, H, W
                    reference = None
                else:
                    reference = frame_pixel_values[:, args.context_length - 1:args.context_length].repeat(
                        1, args.segment_length - args.context_length, 1, 1, 1).reshape(B * num_target_frames, C, H, W)  # B*(T-t), C, H, W
                    reference_single = frame_pixel_values[:, args.context_length - 1]
                pixel_values = target

            fmap, fmap_ref, commit_loss, dyna_commit_loss = self.tokenizer(sample=reference_single,
                                                                           dyn_sample=target,
                                                                           return_dict=False,
                                                                           return_loss=True,
                                                                           segment_len=num_target_frames)

            recon_loss = F.l1_loss(pixel_values, fmap)
            ref_recon_loss = F.l1_loss(reference_single, fmap_ref)
            perceptual_loss = self.lpips(pixel_values.contiguous() * 2 - 1.0,
                                         fmap.contiguous() * 2 - 1.0, weight=None).mean()
            ref_perceptual_loss = self.lpips(reference_single.contiguous() * 2 - 1.0,
                                             fmap_ref.contiguous() * 2 - 1.0, weight=None).mean()
            loss = recon_loss + ref_recon_loss + perceptual_loss + ref_perceptual_loss + commit_loss + dyna_commit_loss

        # loss.backward()
        self.tok_scaler.scale(loss).backward()
        if args.max_grad_norm is not None:
            self.tok_scaler.unscale_(self.tok_optimizer)
            torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), args.max_grad_norm)
        # self.tok_optimizer.step()
        self.tok_scaler.step(self.tok_optimizer)
        self.tok_scaler.update()

        return {
            'tokenizer_loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'ref_recon_loss': ref_recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'ref_perceptual_loss': ref_perceptual_loss.item(),
            'commit_loss': commit_loss.item(),
            'dyna_commit_loss': dyna_commit_loss.item(),
        }

    def update_model(self, args, obs, action, reward):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            self.model_optimizer.zero_grad()
            pixel_values, actions, rewards = obs.to(self.device), action.to(self.device), reward.to(self.device)

            with torch.no_grad():
                tokens, labels = self.tokenizer.tokenize(pixel_values, args.context_length)

                model_input = {
                    'input_ids': tokens,
                    'labels': labels,
                    'action': actions,
                }

            outputs, reward_pred = self.model(**model_input)
            ce_loss = outputs.loss
            reward_loss = F.mse_loss(reward_pred, rewards[:, args.context_length:])
            loss = ce_loss + args.reward_weight * reward_loss

        # loss.backward()
        self.model_scaler.scale(loss).backward()
        grad_norm, grad_norms = 0, {}
        if args.max_grad_norm is not None:
            self.model_scaler.unscale_(self.model_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            grad_norm = grad_norm.item()
        # self.model_optimizer.step()
        self.model_scaler.step(self.model_optimizer)
        self.model_scaler.update()

        return {
            'ce_loss': ce_loss.item(),
            'reward_loss': reward_loss.item(),
            'model_loss': loss.item(),
            'model_train/reward_mean': rewards[:, args.context_length:].mean().item(),
            'model_train/reward_pred_mean': reward_pred.mean().item(),
            'model_train/grad_norm': grad_norm,
            **grad_norms,
        }

    @torch.no_grad()
    def rollout(self, obs, policy, horizon):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            B = obs.shape[0]
            args = self.args

            obs = obs.to(self.device) / 255.
            init_obs = obs
            current_frames = list(torch.chunk(obs, 3, dim=1))  # TODO: magic number (frame_stack=3)

            tokens_per_ctx = 256
            tokens_per_dyn = 16

            context_frames = torch.stack(current_frames[-args.context_length:], dim=1)
            tokens, _ = self.tokenizer.tokenize(torch.cat((context_frames, torch.zeros_like(
                context_frames)), dim=1), args.context_length)
            tokens = tokens[:, :args.context_length * (tokens_per_ctx + 1)]
            init_tokens = tokens
            embeds = self.model.get_input_embeddings(tokens)
            cache = None

            obss = []
            actions = []
            rewards = []

            obs = init_obs
            for t in range(horizon):
                action = policy(obs, t)
                action_embeds = self.model.action_linear(action)
                embeds[:, -1] += action_embeds

                result = self.model.llm.generate(
                    inputs_embeds=embeds,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=50256,
                    top_k=100,
                    use_cache=True,
                    max_new_tokens=tokens_per_dyn + 1,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )

                predicted_token = result.sequences[:, :-1]
                last_layer_hidden_states = result.hidden_states[-1]
                last_token_states = last_layer_hidden_states[-1]
                reward = self.model.reward_linear(last_token_states).squeeze(-2)

                cat_predicted_token = (torch.concat([predicted_token, (torch.ones(B) * self.model.token_for_sdf).unsqueeze(1).to(self.device)], dim=1)
                                       .to(predicted_token.dtype))
                embeds = torch.concat([embeds, self.model.get_input_embeddings(cat_predicted_token)], dim=1)
                # tokens = torch.concat([tokens, predicted_token], dim=1)

                fmap, cache = self.tokenizer.detokenize(torch.concat(
                    [init_tokens, predicted_token], dim=1), args.context_length, cache=cache, return_cache=True)
                fmap = fmap.clamp(0.0, 1.0)
                current_frames.append(fmap[:, -1])
                current_frames.pop(0)
                obs = torch.cat(current_frames, dim=1)

                obss.append(obs)
                actions.append(action)
                rewards.append(reward)

        # dummy step
        obss = [init_obs] + obss
        actions = [torch.zeros_like(actions[0])] + actions
        rewards = [torch.zeros_like(rewards[0])] + rewards

        if self.args.symlog:
            rewards = [symexp(reward) for reward in rewards]

        return torch.stack(obss, 1).float(), torch.stack(actions, 1).float(), torch.stack(rewards, 1).float()

    def save_snapshot(self, workdir, suffix=''):
        torch.save(self.model.state_dict(), os.path.join(workdir, f'model{suffix}.pt'))
        torch.save(self.tokenizer.state_dict(), os.path.join(workdir, f'tokenizer{suffix}.pt'))

    def load_snapshot(self, workdir, suffix=''):
        self.model.load_state_dict(torch.load(os.path.join(workdir, f'model{suffix}.pt')))
        self.tokenizer.load_state_dict(torch.load(os.path.join(workdir, f'tokenizer{suffix}.pt')))
