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
from transformers.integrations import is_mlflow_available, is_wandb_available



from .utils import DataCollatorWithPadding

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
    base_model_prefix = "wrapped_model"
    def __init__(self, base_model, config=None):
        if config is None:
            config = base_model.config
        super().__init__(config)
        if isinstance(base_model, AdaptiveSPINModel):
            self.wrapped_model = base_model.base_model  # Unwrap if nested
        else:
            self.wrapped_model = base_model
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

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=None):
        outputs = self.wrapped_model(
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
        outputs.logits = self.wrapped_model.lm_head(torch.stack(scaled_states, dim=1))
        return modeling_outputs.CausalLMOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            return_dict=return_dict if return_dict is not None else True,
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
                logits = self.wrapped_model.lm_head(h * s_prior)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                next_output = self.wrapped_model(next_id, output_hidden_states=True)
                next_h = next_output.hidden_states[-1][:, -1, :]
                next_s = self.transition_net(s_prior, h)
                candidates.append(next_s)
        return torch.stack(candidates)

    def save_pretrained(self, save_directory, **kwargs):
        self.wrapped_model.save_pretrained(save_directory)
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
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "logistic"] = "sigmoid",
        args: TrainingArguments = None,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        label_pad_token_id: int = -100,
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
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollator]     = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        **kwargs
    ):
        
        self.model_init_kwargs = model_init_kwargs or {}
        self.ref_model_init_kwargs = ref_model_init_kwargs or {}

        self.scaling_rank = scaling_rank
        self.model = model
        self.ref_model = ref_model
        self.max_length = max_length

        # Training parameters
        self.beta = beta
        self.loss_type = loss_type
        self.spinup_steps = spinup_steps
        self.candidate_samples = candidate_samples
        self.current_step = 0
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        self.tokenizer     = tokenizer
        self.data_collator = data_collator
        # Model initialization
        self._initialize_models(model, ref_model, kwargs.get('peft_config'))
        self._setup_optimizer()

        # Initialize scaling components
        # self._setup_scaling_system()
        if model is not None:
            self.is_encoder_decoder = self.model.wrapped_model.config.is_encoder_decoder
        else:
            self.is_encoder_decoder = is_encoder_decoder

        data_collator = DataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
                max_target_length=max_target_length,
                )
        super().__init__(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics = kwargs.get("preprocess_logits_for_metrics"),

            )
    # def _setup_scaling_system(self):
    #     """Initialize all components for adaptive scaling."""
    #     self.dim = self.model.config.hidden_size
    #     self.transition_net = TransitionNet(self.dim).to(self.model.device)
    #     self.adapter = LowRankAdapter(self.dim, self.scaling_rank).to(self.model.device)
    #     self.verifier = StateVerifier(self.dim).to(self.model.device)
        
    #     # Initialize scaling parameters with reasonable defaults
    #     self.s0 = nn.Parameter(torch.ones(self.dim, device=self.model.device))
    #     self.Q = nn.Parameter(torch.ones(self.dim, device=self.model.device))
    #     self.R = nn.Parameter(torch.ones(self.dim, device=self.model.device))
        
        # State tracking
        #self.register_buffer('current_scale', self.s0.clone())
        #self.register_buffer('best_scale', self.s0.clone())

    def _initialize_models(self, model, ref_model, peft_config=None):
        """Handle model initialization with proper validation."""
        # Model initialization logic
        if isinstance(model, str):
            logger.info(f"Loading model from {model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model, 
                **self.model_init_kwargs
            )
        
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
        
        if isinstance(self.model.wrapped_model, PeftModel):
            logger.info("Merging and unloading existing PEFT model")
            self.model.wrapped_model = self.model.wrapped_model.merge_and_unload()
        
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
        pass

    def train_spinup(self, dataset):
        """Initialize scaling dynamics before main training."""
        logger.info(f"Starting spin-up phase for {self.spinup_steps} steps")
        
        spinup_optimizer = torch.optim.Adam([
            {'params': [self.model.s0, self.model.Q, self.model.R], 'lr': 1e-3},
            {'params': self.model.transition_net.parameters(), 'lr': 1e-4},
            {'params': self.model.adapter.parameters(), 'lr': 1e-4}
        ])
        
        self.model.wrapped_model.eval()
        self.model.transition_net.train()
        self.model.adapter.train()
        
        for step in range(self.spinup_steps):
            total_loss = 0.0
            for batch in self.get_train_dataloader():
                inputs = self._prepare_inputs(batch)
                
                input_ids      = inputs.get("real_input_ids")
                attention_mask = inputs.get("real_attention_mask", None)
                print("spin‑up inputs:", inputs.keys())
                with torch.no_grad():
                    #outputs = self.model(**inputs, output_hidden_states=True)
                    outputs = self.model(
                        input_ids      = input_ids,
                        attention_mask = attention_mask,
                        output_hidden_states = True,
                        return_dict=True,
                    )
                hs = outputs.hidden_states[-1]
                    
                del outputs.hidden_states
                # Process hidden states
                device = self.model.s0.device
                h_seq = hs[:, :-1].flatten(0, 1).to(device)  # (seq_len-1)*bsz x dim
                h_next = hs[:, 1:].flatten(0, 1).to(device)
                #hs = hs.to(device)
                #print(hs.shape)
                #_, T, D = hs.shape
                # Initialize scale sequence
                sequences = []
                
                s_seq = self.model.s0.unsqueeze(0).expand(h_seq.size(0), -1).to(device)
                
                s_prev = s_seq[0]
                sequences.append(s_prev)
                
                # 2) iterate over the flattened hidden states
                for t in range(1, h_seq.size(0)):
                    h_prev = h_seq[t-1]                         # [D]
                    # transition_net lives on the same device, so this yields [D]
                    s_t = self.model.transition_net(s_prev, h_prev)
                    sequences.append(s_t)                  # [D]
                    s_prev = s_t       
                s_seq = torch.stack(sequences)
                
                # Compute losses
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
        self.current_scale = self.model.s0.clone()
        self.best_scale = self.model.s0.clone()

    def scaled_forward(self, input_ids, attention_mask=None):
        """Forward pass with adaptive scaling dynamics."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        #input_embeds
        #hs = outputs.hidden_states[-1]
        
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
                s_prior = self.model.transition_net(s, h_t)
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
                next_s = self.model.transition_net(s_prior, h)
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

    def training_step(self, model, inputs, num_items_in_batch=None):
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
