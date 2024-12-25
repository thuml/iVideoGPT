from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class HeadModelWithAction(nn.Module):
    def __init__(self, llm, action_dim, prelude_tokens_num, tokens_num_per_dyna, context, segment_length, model_type='llama',
                 reward_prediction=False, action_recon=None, **kwargs):
        #              prelude                          dyna
        #                 |                              |
        # ([0:255] scf [0:255])         sdf [0:15] sdf [0:15]
        # action[0]     action[1]        action[2]     action[3]
        # reward[0]     reward[1]        reward[2]     reward[3]
        #                                reward_pred[0] reward_pred[1]
        super().__init__()

        self.llm = llm
        self.action_dim = action_dim
        self.prelude_tokens_num = prelude_tokens_num
        self.tokens_num_per_dyna = tokens_num_per_dyna
        self.context = context
        self.segment_length = segment_length
        self.model_type = model_type
        self.token_for_sdf = llm.config.vocab_size - 1  # the last token is used for sdf
        self.reward_prediction = reward_prediction
        self.action_recon = action_recon

        if self.model_type == 'llama':
            embed_dim = llm.config.hidden_size
        elif self.model_type == 'gpt2':
            embed_dim = llm.config.n_embd
        else:
            raise ValueError(f"model_type {self.model_type} is not supported.")
        self.action_linear = nn.Linear(action_dim, embed_dim)

        nn.init.zeros_(self.action_linear.weight)
        nn.init.zeros_(self.action_linear.bias)

        if self.reward_prediction:
            self.reward_linear = nn.Linear(embed_dim, 1)

        if self.action_recon:
            self.action_recon_linear = nn.Linear(embed_dim, action_dim)

    def get_input_embeddings(self, input_ids):
        if self.model_type == 'llama':
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        elif self.model_type == 'gpt2':
            inputs_embeds = self.llm.wte(input_ids)
        else:
            raise ValueError(f"model_type {self.model_type} is not supported.")
        return inputs_embeds

    @torch.no_grad()
    def generate(
        self,
        inputs_token,
        do_sample=True,
        temperature=1.0,
        top_k=100,
        max_new_tokens=None,
        pad_token_id=50256,
        action: Optional[torch.FloatTensor] = None,  # B,T,D
    ):
        #                   prelude                                                 prediction
        #                    |                                                          |
        # ([0:255] scf     [0:255]) sdf             [0:15] sdf     [0:15] sdf   [0:15] sdf    [0:15] sdf(last one discarded)\
        # action[0]        action[1]              action[2]     action[3]
        device = inputs_token.device
        token_per_dyna = ((max_new_tokens + 1) // (self.segment_length - self.context)) - 1
        B, T = inputs_token.size()
        action_embeds = self.action_linear(action)
        inputs_embeds = self.get_input_embeddings(inputs_token)
        rewards = []

        for i in range(self.segment_length - self.context):
            # equivalent to cut off context - 1 actions
            inputs_embeds[:, self.prelude_tokens_num + i *
                          (self.tokens_num_per_dyna + 1), :] += action_embeds[:, i + self.context - 1, :]

            if self.reward_prediction:
                # !TODO: bug fix, currently not used
                # !See line 298-313 in mbrl/video_predictor.py for correct reward prediction
                result = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                    top_k=top_k,
                    use_cache=True,
                    max_new_tokens=token_per_dyna,
                    return_dict_in_generate=True
                )
                predicted_token = result.sequences
                last_token_hidden_states = result.hidden_states[-1]
                last_layer_states = last_token_hidden_states[-1]
                rewards.append(self.reward_linear(last_layer_states).squeeze(-1))
            else:
                predicted_token = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                    top_k=top_k,
                    use_cache=True,
                    max_new_tokens=token_per_dyna,
                    return_dict_in_generate=False
                )
            predicted_token = (torch.concat([predicted_token, (torch.ones(B) * self.token_for_sdf).unsqueeze(1).to(device)], dim=1)
                               .to(predicted_token.dtype))
            inputs_embeds = torch.concat([inputs_embeds, self.get_input_embeddings(predicted_token)], dim=1)
            inputs_token = torch.concat([inputs_token, predicted_token], dim=1)

        assert inputs_token.size(1) == T + max_new_tokens + 1  # +1 for the last token
        assert len(rewards) == 0 or len(rewards) == self.segment_length - self.context
        if self.reward_prediction:
            reward = torch.stack(rewards, dim=1).squeeze(-1)
            return inputs_token[:, :-1], reward  # B, segment-context
        return inputs_token[:, :-1]  # the last token(sdf) is not used

    @torch.no_grad()
    def generate_without_action(
        self,
        inputs_token,
        do_sample=True,
        temperature=1.0,
        top_k=100,
        max_new_tokens=None,
    ):
        #                   prelude                                                 prediction
        #                    |                                                          |
        # ([0:255] scf [0:255] scf [0:255]) sdf             [0:15] sdf     [0:15] sdf   [0:15] sdf    [0:15] sdf(last one discarded)
        device = inputs_token.device
        token_per_dyna = ((max_new_tokens + 1) // (self.segment_length - self.context)) - 1
        B, T = inputs_token.size()
        for i in range(self.segment_length - self.context):
            inputs_embeds = self.get_input_embeddings(inputs_token)
            predicted_token = self.llm.generate(
                inputs_embeds=inputs_embeds,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=token_per_dyna
            )
            inputs_token = torch.concat([inputs_token, predicted_token,
                                         (torch.ones(B) * self.token_for_sdf).unsqueeze(1).to(device)], dim=1).to(torch.int64)

        assert inputs_token.size(1) == T + max_new_tokens + 1  # +1 for the last token
        return inputs_token[:, :-1]  # the last token(sdf) is not used

    # input: tokens + action(B,T,D)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        action: Optional[torch.FloatTensor] = None,  # B,T,D
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        #              prelude                                                      dyna
        #                 |                                                          |
        # ([0:255] scf [0:255] scf [0:255])     (sdf+action[0]->[0:15]) (sdf+action[1]->[0:15]) (sdf+action[2]->[0:15])
        # [0:255]  -> [0:255]
        #  +
        # action
        #                                                            r0                    r1                       r2

        inputs_embeds = self.get_input_embeddings(input_ids)
        action_embeds = self.action_linear(action)
        action_embeds = action_embeds[:, self.context - 1: -1, :]
        new_inputs_embeds = inputs_embeds.clone()
        start_index = self.prelude_tokens_num + \
            torch.arange(self.segment_length - self.context) * (self.tokens_num_per_dyna + 1)
        new_inputs_embeds[:, start_index, :] += action_embeds

        x = self.llm(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=new_inputs_embeds,
            labels=labels,
            output_hidden_states=self.reward_prediction or self.action_recon,
        )

        if self.action_recon:
            hidden_states = x.hidden_states[-1]
            hidden_action_recon_states = hidden_states[:, self.prelude_tokens_num:]
            action_recon = self.action_recon_linear(
                hidden_action_recon_states).reshape(-1, self.segment_length - self.context, (self.tokens_num_per_dyna + 1), self.action_dim)
            action_recon_loss = nn.functional.mse_loss(
                action_recon, action[:, self.context - 1:-1].unsqueeze(-2).repeat(1, 1, self.tokens_num_per_dyna + 1, 1))
            self.action_recon_loss = action_recon_loss
            x.loss += self.action_recon * action_recon_loss

        if self.reward_prediction:
            hidden_states = x.hidden_states[-1]
            # see diagram above
            reward_start_index = start_index + self.tokens_num_per_dyna
            hidden_reward_states = hidden_states[:, reward_start_index, :]
            reward_pred = self.reward_linear(hidden_reward_states)
            return x, reward_pred  # B, segment-context
        return x
