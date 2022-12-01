from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration, MT5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

from torch import nn
from torch.autograd import Function
import torch

from typing import Optional, Tuple, Union


class StopGradient(Function):
    @staticmethod
    def forward(ctx, i):
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return 0 * grad_output


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, i):
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return -1 * grad_output


class AlignedMT5ForConditionalGeneration(MT5ForConditionalGeneration):
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.adversary = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.frozen = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.adversary_weight_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def parallelize(self, device_map):
        super().parallelize(device_map)
        first_device = "cuda:" + str(min(self.device_map.keys()))
        last_device = "cuda:" + str(max(self.device_map.keys()))
        self.adversary_weight_param.data = self.adversary_weight_param.to(first_device)
        self.adversary = self.adversary.to(last_device)

    @torch.no_grad()
    def copy_to_frozen(self):
        for i, layer in enumerate(self.frozen):
            if i == 0 or i == 2:
                layer.weight.data = self.adversary[i].weight.data.clone().detach()
                layer.weight.data.requires_grad = False
                layer.bias.data = self.adversary[i].bias.data.clone().detach()
                layer.bias.data.requires_grad = False

    @torch.no_grad()
    def non_negative_weight(self):
        self.adversary_weight_param.data = (self.adversary_weight_param).clamp(min=0)

    def adversary_loss(
        self,
        hidden_states: torch.FloatTensor,
        adversarial_label: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        frozen=False,
    ):
        reversed_grad_states = GradientReversal.apply(hidden_states)
        if frozen:
            self.copy_to_frozen()
            adversary = self.frozen
        else:
            adversary = self.adversary
        logits = (
            adversary(reversed_grad_states).squeeze(dim=2).to(adversarial_label.device)
        )
        loss_func = nn.BCEWithLogitsLoss(reduction="none")
        adversarial_label = GradientReversal.apply(
            adversarial_label[:, None].expand(logits.shape)
        )
        adv_loss = loss_func(logits.float(), adversarial_label.float())

        adv_loss = torch.mean(adv_loss * attention_mask)
        return adv_loss

    def constrained_adversary_loss(
        self,
        hidden_states: torch.FloatTensor,
        adversarial_label: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        eps=0.3,
    ):
        self.non_negative_weight()
        true_adv_loss = self.adversary_loss(
            StopGradient.apply(hidden_states.clone()),
            adversarial_label.clone(),
            attention_mask.clone(),
        )
        neg_adv_loss = self.adversary_loss(
            hidden_states, adversarial_label, attention_mask, frozen=True
        )
        constrained_term = (neg_adv_loss - eps) * self.adversary_weight_param
        return constrained_term + true_adv_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        adversarial_data: Optional[torch.LongTensor] = None,
        adversarial_mask: Optional[torch.FloatTensor] = None,
        adversarial_label: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if (
            adversarial_data != None
            and adversarial_label != None
            and adversarial_mask != None
        ):
            adv_outputs = self.encoder.forward(
                input_ids=adversarial_data,
                attention_mask=adversarial_mask,
                return_dict=return_dict,
            )
            encoder_hidden_states = adv_outputs.last_hidden_state
            adv_loss = self.constrained_adversary_loss(
                encoder_hidden_states, adversarial_label, adversarial_mask
            )
            outputs.loss = (outputs.loss + adv_loss).squeeze()
        return outputs
