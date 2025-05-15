#!/usr/bin/env python
# Enhanced Adaptive SPIN Trainer with Robust Scaling Dynamics
import logging
import sys
import warnings
from collections import defaultdict
from typing import Optional, Dict, List, Union, Literal, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    modeling_outputs
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from transformers.utils import is_peft_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

class TransitionNet(nn.Module):
    """Neural network for modeling state transitions in scaling dynamics."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*2, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim),
            nn.LayerNorm(dim))
        
    def forward(self, s_prev, h_prev):
        return self.net(torch.cat([s_prev, h_prev], dim=-1))

class LowRankAdapter(nn.Module):
    """Efficient low-rank adapter for scale updates."""
    def __init__(self, dim, rank=64):
        super().__init__()
        self.down_proj = nn.Linear(dim*2, rank, bias=False)
        self.up_proj = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Linear(dim*2, rank)
        
    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        return self.up_proj(F.silu(self.gate(x)) * self.down_proj(x))

class StateVerifier(nn.Module):
    """Learned verifier for scale state quality."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.SiLU(),
            nn.Linear(dim*2, 1),
            nn.Sigmoid())
        
    def forward(self, states):
        return self.net(states).squeeze(-1)

class AdaptiveSPINModel(PreTrainedModel):
    def __init__(self, base_model, config=None):
        if config is None:
            config = base_model.config
        super().__init__(config)
        self.base_model = base_model
        self.dim = config.hidden_size
        
        # Scaling components
        self.transition_net = TransitionNet(self.dim)
        self.adapter = LowRankAdapter(self.dim)
        self.verifier = StateVerifier(self.dim)
        
        # Scaling parameters
        self.s0 = nn.Parameter(torch.ones(self.dim))
        self.Q = nn.Parameter(torch.ones(self.dim))
        self.R = nn.Parameter(torch.ones(self.dim))
        
        # State buffers
        self.register_buffer('current_scale', self.s0.clone())
        self.register_buffer('best_scale', self.s0.clone())

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hs = outputs.hidden_states[-1]
        batch_size = hs.size(0)
        s = self.current_scale.unsqueeze(0).expand(batch_size, -1).to(hs.device)
        scaled_states = []
        
        for t in range(hs.size(1)):
            h_t = hs[:, t, :]
            scaled_h = h_t * s
            scaled_states.append(scaled_h)
            
            if t < hs.size(1) - 1:
                s_prior = self.transition_net(s, h_t)
                s = self._update_scale(h_t, s_prior)
        
        self.current_scale = s.mean(0).detach()
        outputs.logits = self.base_model.lm_head(torch.stack(scaled_states, dim=1))
        return modeling_outputs.CausalLMOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )

    def _update_scale(self, h, s_prior):
        candidates = self._generate_candidates(h, s_prior)
        scores = self.verifier(candidates)
        y_obs = (scores.softmax(-1).unsqueeze(-1) * candidates).sum(dim=0)
        delta = self.adapter(h, s_prior)
        return s_prior + delta

    def _generate_candidates(self, h, s_prior, K=5):
        candidates = []
        with torch.no_grad():
            for _ in range(K):
                logits = self.base_model.lm_head(h * s_prior)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                next_output = self.base_model(next_id, output_hidden_states=True)
                next_h = next_output.hidden_states[-1][:, -1, :]
                next_s = self.transition_net(s_prior, h)
                candidates.append(next_s)
        return torch.stack(candidates)

    def save_pretrained(self, save_directory, **kwargs):
        self.base_model.save_pretrained(save_directory)
        state = {
            'transition_net': self.transition_net.state_dict(),
            'adapter': self.adapter.state_dict(),
            'verifier': self.verifier.state_dict(),
            's0': self.s0,
            'Q': self.Q,
            'R': self.R,
            'current_scale': self.current_scale,
            'best_scale': self.best_scale
        }
        torch.save(state, f"{save_directory}/scaling_components.bin")
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(base_model)
        try:
            state = torch.load(f"{pretrained_model_name_or_path}/scaling_components.bin", map_location='cpu')
            model.transition_net.load_state_dict(state['transition_net'])
            model.adapter.load_state_dict(state['adapter'])
            model.verifier.load_state_dict(state['verifier'])
            model.s0 = nn.Parameter(state['s0'])
            model.Q = nn.Parameter(state['Q'])
            model.R = nn.Parameter(state['R'])
            model.current_scale = state['current_scale']
            model.best_scale = state['best_scale']
        except FileNotFoundError:
            logger.warning("No scaling components found, initialized with defaults")
        return model

class AdaptiveSPINTrainer(Trainer):
    def __init__(
        self,
        model: Union[AdaptiveSPINModel, str] = None,
        ref_model: Optional[Union[PreTrainedModel, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        spinup_steps: int = 1000,
        candidate_samples: int = 5,
        **kwargs
    ):
        if isinstance(model, str):
            model = AdaptiveSPINModel.from_pretrained(model)
            
        self.beta = beta
        self.loss_type = loss_type
        self.spinup_steps = spinup_steps
        self.candidate_samples = candidate_samples
        self.current_step = 0
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(model=model, args=args, **kwargs)

    def train_spinup(self, dataset):
        logger.info(f"Starting spin-up phase for {self.spinup_steps} steps")
        spinup_optimizer = torch.optim.Adam([
            {'params': self.model.s0}, 
            {'params': self.model.Q},
            {'params': self.model.R},
            {'params': self.model.transition_net.parameters()},
            {'params': self.model.adapter.parameters()}
        ], lr=1e-3)
        
        self.model.base_model.eval()
        for step in range(self.spinup_steps):
            total_loss = 0.0
            for batch in self.get_train_dataloader():
                inputs = self._prepare_inputs(batch)
                with torch.no_grad():
                    outputs = self.model.base_model(**inputs, output_hidden_states=True)
                    hs = outputs.hidden_states[-1]
                
                h_seq = hs[:, :-1].flatten(0, 1)
                h_next = hs[:, 1:].flatten(0, 1)
                
                # Scale sequence processing
                s_seq = [self.model.s0.unsqueeze(0).expand(h_seq.size(0), -1)]
                for t in range(1, h_seq.size(0)):
                    s_t = self.model.transition_net(s_seq[t-1], h_seq[t-1])
                    s_seq.append(s_t)
                s_seq = torch.stack(s_seq)
                
                # Loss calculation
                scaled_h = h_seq * s_seq
                state_loss = ((scaled_h - h_next).pow(2) / self.model.R.exp()).mean()
                dyn_loss = ((s_seq[1:] - self.model.transition_net(s_seq[:-1], h_seq[:-1])).pow(2) / self.model.Q.exp()).mean()
                
                loss = state_loss + dyn_loss
                loss.backward()
                spinup_optimizer.step()
                spinup_optimizer.zero_grad()
                total_loss += loss.item()
            
            if step % 100 == 0:
                logger.info(f"Spin-up step {step}: loss={total_loss/len(dataset):.4f}")
                
        logger.info("Spin-up phase completed")

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )
        
        with torch.no_grad():
            ref_outputs = self.model.base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask")
            )
        
        policy_logps = F.log_softmax(outputs.logits, dim=-1).gather(-1, inputs["labels"].unsqueeze(-1)).squeeze(-1)
        ref_logps = F.log_softmax(ref_outputs.logits, dim=-1).gather(-1, inputs["labels"].unsqueeze(-1)).squeeze(-1)
        
        log_ratio = policy_logps - ref_logps
        if self.loss_type == "sigmoid":
            loss = -F.logsigmoid(self.beta * log_ratio).mean()
        elif self.loss_type == "hinge":
            loss = F.relu(1 - self.beta * log_ratio).mean()
        
        # Logging
        if self.current_step % self.args.logging_steps == 0:
            self.log({
                "loss": loss.item(),
                "scale_norm": self.model.current_scale.norm().item(),
                "log_ratio": log_ratio.mean().item()
            })
            
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        if self.current_step < self.spinup_steps and not hasattr(self, '_spinup_complete'):
            self.train_spinup(self.train_dataset)
            self._spinup_complete = True
            
        loss = super().training_step(model, inputs)
        self.current_step += 1
        return loss

if __name__ == "__main__":
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = AdaptiveSPINModel(base_model)
    
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-5,
        logging_steps=5
    )
    
    trainer = AdaptiveSPINTrainer(
        model=model,
        args=args,
        beta=0.1,
        spinup_steps=10
    )
    
    # After training
    model.save_pretrained("./trained_spin_model")