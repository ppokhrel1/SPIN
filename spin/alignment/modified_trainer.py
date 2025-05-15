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
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from transformers.utils import is_peft_available

from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length
from transformers.integrations import is_mlflow_available, is_wandb_available


from transformers.trainer_utils import EvalLoopOutput

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

class AdaptiveSPINTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "logistic"] = "sigmoid",
        args: TrainingArguments = None,
        spinup_steps: int = 1000,
        scaling_rank: int = 64,
        candidate_samples: int = 5,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        **kwargs
    ):
        
        self.model_init_kwargs = model_init_kwargs or {}
        self.ref_model_init_kwargs = ref_model_init_kwargs or {}

        # Training parameters
        self.beta = beta
        self.loss_type = loss_type
        self.spinup_steps = spinup_steps
        self.candidate_samples = candidate_samples
        self.current_step = 0
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        # Model initialization
        self._initialize_models(model, ref_model, kwargs.get('peft_config'))
        self._setup_optimizer()

        # Initialize scaling components
        self._setup_scaling_system()
        

        self.super().__init__(model=self.model, args=args, **kwargs)

    def _setup_scaling_system(self):
        """Initialize all components for adaptive scaling."""
        self.dim = self.model.config.hidden_size
        self.transition_net = TransitionNet(self.dim).to(self.model.device)
        self.adapter = LowRankAdapter(self.dim, self.scaling_rank).to(self.model.device)
        self.verifier = StateVerifier(self.dim).to(self.model.device)
        
        # Initialize scaling parameters with reasonable defaults
        self.s0 = nn.Parameter(torch.ones(self.dim, device=self.model.device))
        self.Q = nn.Parameter(torch.ones(self.dim, device=self.model.device))
        self.R = nn.Parameter(torch.ones(self.dim, device=self.model.device))
        
        # State tracking
        self.register_buffer('current_scale', self.s0.clone())
        self.register_buffer('best_scale', self.s0.clone())

    def _initialize_models(self, model, ref_model, peft_config=None):
        """Handle model initialization with proper validation."""
        # Model initialization logic
        if isinstance(model, str):
            logger.info(f"Loading model from {model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model, 
                **self.model_init_kwargs
            ).to(self.args.device)
        
        # Reference model handling
        if ref_model is not None:
            if isinstance(ref_model, str):
                logger.info(f"Loading reference model from {ref_model}")
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    ref_model,
                    **self.ref_model_init_kwargs
                ).to(self.args.device)
            else:
                self.ref_model = ref_model.to(self.args.device)
        
        # PEFT configuration
        if peft_config is not None:
            self._configure_peft(peft_config)

    def _configure_peft(self, peft_config):
        """Handle PEFT model configuration."""
        if not is_peft_available():
            raise ImportError("PEFT is required but not installed. Use `pip install peft`.")
        
        from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model
        
        if isinstance(self.model, PeftModel):
            logger.info("Merging and unloading existing PEFT model")
            self.model = self.model.merge_and_unload()
        
        # Prepare for 4/8-bit training if needed
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            logger.info("Preparing model for k-bit training")
            prepare_kwargs = {"use_gradient_checkpointing": self.args.gradient_checkpointing}
            if hasattr(self.args, "gradient_checkpointing_kwargs"):
                prepare_kwargs.update(self.args.gradient_checkpointing_kwargs)
            self.model = prepare_model_for_kbit_training(self.model, **prepare_kwargs)
        
        # Convert to PEFT model
        logger.info("Applying PEFT configuration")
        self.model = get_peft_model(self.model, peft_config)
        self.is_peft_model = True

    def _setup_optimizer(self):
        """Configure optimizer with scaling parameters."""
        if self.optimizer is None:
            return
            
        scaling_params = [
            {'params': [self.s0, self.Q, self.R]},
            {'params': self.transition_net.parameters(), 'lr': 1e-4},
            {'params': self.adapter.parameters(), 'lr': 1e-4},
            {'params': self.verifier.parameters(), 'lr': 1e-4}
        ]
        
        # Add to existing optimizer or create new one
        if hasattr(self.optimizer, 'param_groups'):
            self.optimizer.add_param_group(scaling_params)
        else:
            self.optimizer = torch.optim.AdamW(scaling_params)

    def train_spinup(self, dataset):
        """Initialize scaling dynamics before main training."""
        logger.info(f"Starting spin-up phase for {self.spinup_steps} steps")
        
        spinup_optimizer = torch.optim.Adam([
            {'params': [self.s0, self.Q, self.R], 'lr': 1e-3},
            {'params': self.transition_net.parameters(), 'lr': 1e-4},
            {'params': self.adapter.parameters(), 'lr': 1e-4}
        ])
        
        self.model.eval()
        self.transition_net.train()
        self.adapter.train()
        
        for step in range(self.spinup_steps):
            total_loss = 0.0
            for batch in self.get_train_dataloader():
                inputs = self._prepare_inputs(batch)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hs = outputs.hidden_states[-1]
                
                # Process hidden states
                h_seq = hs[:, :-1].flatten(0, 1)  # (seq_len-1)*bsz x dim
                h_next = hs[:, 1:].flatten(0, 1)
                
                # Initialize scale sequence
                s_seq = [self.s0.unsqueeze(0).expand(h_seq.size(0), -1)]
                for t in range(1, h_seq.size(0)):
                    s_t = self.transition_net(s_seq[t-1], h_seq[t-1])
                    s_seq.append(s_t)
                s_seq = torch.stack(s_seq)
                
                # Compute losses
                scaled_h = h_seq * s_seq
                state_loss = ((scaled_h - h_next).pow(2) / self.R.exp()).mean()
                dyn_loss = ((s_seq[1:] - self.transition_net(s_seq[:-1], h_seq[:-1])).pow(2) / self.Q.exp()).mean()
                
                loss = state_loss + dyn_loss
                loss.backward()
                spinup_optimizer.step()
                spinup_optimizer.zero_grad()
                total_loss += loss.item()
            
            if step % 100 == 0:
                logger.info(f"Spin-up step {step}: loss={total_loss/len(dataset):.4f}")
        
        logger.info("Spin-up phase completed")
        self.current_scale = self.s0.clone()
        self.best_scale = self.s0.clone()

    def scaled_forward(self, input_ids, attention_mask=None):
        """Forward pass with adaptive scaling dynamics."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hs = outputs.hidden_states[-1]
        
        # Initialize scaling
        batch_size = hs.size(0)
        s = self.current_scale.unsqueeze(0).expand(batch_size, -1)
        scaled_states = []
        
        # Process each timestep
        for t in range(hs.size(1)):
            h_t = hs[:, t, :]
            
            # Apply current scaling
            scaled_h = h_t * s
            scaled_states.append(scaled_h)
            
            # Update scale for next step
            if t < hs.size(1) - 1:
                s_prior = self.transition_net(s, h_t)
                s = self._update_scale(h_t, s_prior)
        
        # Update class state
        self.current_scale = s.mean(0).detach()
        
        # Compute final logits
        outputs.logits = self.model.lm_head(torch.stack(scaled_states, dim=1))
        return outputs

    def _update_scale(self, h, s_prior):
        """Update scale state using verification and adaptation."""
        # Generate candidate states
        candidates = self._generate_candidates(h, s_prior)
        
        # Score candidates
        with torch.no_grad():
            scores = self.verifier(candidates)
        
        # Weighted combination
        y_obs = scores.softmax(-1).unsqueeze(-1) * candidates
        y_obs = y_obs.sum(dim=0)
        
        # Apply adapter
        delta = self.adapter(h, s_prior)
        return s_prior + delta

    def _generate_candidates(self, h, s_prior, K=None):
        """Generate candidate scale states using model predictions."""
        K = K or self.candidate_samples
        candidates = []
        
        with torch.no_grad():
            for _ in range(K):
                # Generate next token
                logits = self.model.lm_head(h * s_prior)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                
                # Get next hidden state
                next_output = self.model(
                    next_id,
                    output_hidden_states=True
                )
                next_h = next_output.hidden_states[-1][:, -1, :]
                
                # Predict next scale
                next_s = self.transition_net(s_prior, h)
                candidates.append(next_s)
        
        return torch.stack(candidates)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute SPIN loss with adaptive scaling."""
        # Forward pass with scaling
        outputs = self.scaled_forward(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )
        
        # Calculate policy and reference logps
        policy_logps = self._get_logps(outputs.logits, inputs["labels"])
        with torch.no_grad():
            ref_outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask")
            )
            ref_logps = self._get_logps(ref_outputs.logits, inputs["labels"])
        
        # SPIN loss calculation
        log_ratio = policy_logps - ref_logps
        if self.loss_type == "sigmoid":
            loss = -F.logsigmoid(self.beta * log_ratio).mean()
        elif self.loss_type == "hinge":
            loss = F.relu(1 - self.beta * log_ratio).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Log metrics
        metrics = {
            "loss": loss.item(),
            "scale_norm": self.current_scale.norm().item(),
            "log_ratio": log_ratio.mean().item()
        }
        self._log_metrics(metrics)
        
        return (loss, outputs) if return_outputs else loss

    def _get_logps(self, logits, labels):
        """Get log probabilities for labels."""
        logps = F.log_softmax(logits, dim=-1)
        return logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def _log_metrics(self, metrics):
        """Store and log training metrics."""
        for k, v in metrics.items():
            self._stored_metrics[k][self.current_step].append(v)
        
        if self.current_step % self.args.logging_steps == 0:
            avg_metrics = {
                k: sum(v)/len(v) 
                for k, v in self._stored_metrics.items()
            }
            logger.info(f"Step {self.current_step}: {avg_metrics}")
            self._stored_metrics.clear()

    def training_step(self, model, inputs):
        """Override training step to include spin-up if needed."""
        if self.current_step < self.spinup_steps and not hasattr(self, '_spinup_complete'):
            self.train_spinup(self.train_dataset)
            self._spinup_complete = True
        
        loss = super().training_step(model, inputs)
        self.current_step += 1
        return loss

if __name__ == "__main__":
    # Example usage
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    args = TrainingArguments(
        output_dir="./spin_results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-5,
        logging_steps=10,
        gradient_checkpointing=True
    )
    
    trainer = AdaptiveSPINTrainer(
        model=model,
        args=args,
        beta=0.1,
        spinup_steps=500,
        scaling_rank=64
    )
    
    # Load dataset here (implementation specific)
    # train_dataset = ...
    
    # Start training
    trainer.train()
