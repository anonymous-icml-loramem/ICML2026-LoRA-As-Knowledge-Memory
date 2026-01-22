# src/trainers/dcd_trainer.py

import os
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from src.utils import log_to_wandb
# Simple Dataset class for feeding training data (list of dicts) into DataLoader
class SimpleDictDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DCDTrainer:
    """
    Trainer for Deep Context Distillation (DCD/PnP) training.
    """
    def __init__(self, config, teacher_model, student_model, tokenizer, dataset, optimizer):
        self.config = config
        self.training_config = config['training']
        self.dcd_params = self.training_config.get('dcd_params', {
        'distillation_temp': 2.0,
        'l1_loss_weight'  : 0.1,
        'loss_mode'       : 'all',
    })
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.output_dir = config.get('output_dir',
                                 os.path.join('outputs', config['experiment']['name']))
        self._eos_token = getattr(self.tokenizer, "eos_token", None) or getattr(self.tokenizer, "pad_token", "")
        self.max_length = self.training_config.get("max_length", 4096)
        self._autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._use_autocast = torch.cuda.is_available()

        # Device configuration
        self.teacher_device = teacher_model.device
        self.student_device = student_model.device
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Create DataLoader
        train_dataset = SimpleDictDataset(dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        num_steps = self.training_config['num_train_steps']
        warmup_ratio = self.training_config.get('warmup_ratio', 0.0)
        num_warmup_steps = int(num_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )

    def _format_chat(self, messages, response_text):
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": response_text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
        except TypeError:
            # Fallback for tokenizers that don't accept enable_thinking argument (e.g., Llama)
            formatted = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": response_text}],
                tokenize=False,
                add_generation_prompt=False
            )
        if self._eos_token and not formatted.endswith(self._eos_token):
            formatted += self._eos_token
        return formatted

    def _format_prompt(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def _encode_with_labels(self, full_text: str, prefix_text: str):
        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        prefix = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        prefix_len = min(len(prefix["input_ids"]), len(input_ids))
        labels = list(input_ids)
        for i in range(prefix_len):
            labels[i] = -100
        return input_ids, attention_mask, labels

    def _pad_batch(self, sequences, pad_value):
        max_len = max(len(seq) for seq in sequences) if sequences else 0
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _collate_fn(self, batch):
        """Tokenizes batch data for model input."""
        teacher_input_ids = []
        teacher_attention = []
        teacher_labels = []
        student_input_ids = []
        student_attention = []
        student_labels = []

        for item in batch:
            if 'teacher_messages' in item and 'student_messages' in item and 'assistant_response' in item:
                response_text = item['assistant_response']
                teacher_prompt = self._format_prompt(item['teacher_messages'])
                student_prompt = self._format_prompt(item['student_messages'])
                teacher_full = self._format_chat(item['teacher_messages'], response_text)
                student_full = self._format_chat(item['student_messages'], response_text)
            else:
                context = item.get('context', '')
                synthetic_sample = item.get('synthetic_sample', '')
                teacher_prompt = context
                student_prompt = ""
                teacher_full = context + synthetic_sample
                student_full = synthetic_sample

            t_ids, t_attn, t_labels = self._encode_with_labels(teacher_full, teacher_prompt)
            s_ids, s_attn, s_labels = self._encode_with_labels(student_full, student_prompt)
            teacher_input_ids.append(t_ids)
            teacher_attention.append(t_attn)
            teacher_labels.append(t_labels)
            student_input_ids.append(s_ids)
            student_attention.append(s_attn)
            student_labels.append(s_labels)

        teacher_tokens = {
            "input_ids": self._pad_batch(teacher_input_ids, self.tokenizer.pad_token_id),
            "attention_mask": self._pad_batch(teacher_attention, 0),
            "labels": self._pad_batch(teacher_labels, -100),
        }
        student_tokens = {
            "input_ids": self._pad_batch(student_input_ids, self.tokenizer.pad_token_id),
            "attention_mask": self._pad_batch(student_attention, 0),
            "labels": self._pad_batch(student_labels, -100),
        }
        return teacher_tokens, student_tokens

    def _calculate_dcd_loss(self, teacher_outputs, student_outputs, teacher_labels, student_labels):
        """Calculates DCD loss (KL-Divergence + L1-Regularization)."""
        # --- Modified section start ---
        loss_mode = self.dcd_params.get('loss_mode', 'all')
        
        valid_idx = teacher_labels != -100
        nc_valid_idx = student_labels != -100
        if valid_idx.sum() == 0 or nc_valid_idx.sum() == 0:
            return student_outputs.logits.sum() * 0.0, 0.0, 0.0

        teacher_logits = teacher_outputs.logits[valid_idx].to(self.student_device)
        student_logits = student_outputs.logits[nc_valid_idx]
        teacher_hidden_states = [
            hidden_state[valid_idx].to(self.student_device)
            for hidden_state in teacher_outputs.hidden_states
        ]
        student_hidden_states = [
            hidden_state[nc_valid_idx] for hidden_state in student_outputs.hidden_states
        ]
        temp = self.dcd_params['distillation_temp']
        
        kl_loss = torch.tensor(0.0, device=self.student_device)
        loss_l1 = torch.tensor(0.0, device=self.student_device)

        # 1. Calculate KL-Divergence Loss (Logits Distillation)
        if loss_mode in ['all', 'kl_only']:
            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
            
            kl_loss_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
            kl_loss = kl_loss_per_token.mean()

        # 2. Calculate Hidden State L1 Loss
        if loss_mode in ['all', 'l1_only']:
            layer_losses = []
            for student_hs, teacher_hs in zip(student_hidden_states, teacher_hidden_states):
                if student_hs.size(0) != teacher_hs.size(0):
                    logging.warning("Skipping batch due to mismatch in hidden state shapes.")
                    return student_outputs.logits.sum() * 0.0, 0.0, 0.0
                diff = torch.abs(student_hs - teacher_hs).mean()
                teacher_norm = torch.abs(teacher_hs).mean()
                layer_losses.append(diff / (teacher_norm + 1e-8))
            if layer_losses:
                loss_l1 = torch.mean(torch.stack(layer_losses))

        # 3. Combine final loss
        if loss_mode == 'kl_only':
            total_loss = kl_loss
        elif loss_mode == 'l1_only':
            total_loss = loss_l1 * self.dcd_params['l1_loss_weight']
        else: # 'all'
            total_loss = kl_loss + loss_l1 * self.dcd_params['l1_loss_weight']
            
        return total_loss, kl_loss.item(), loss_l1.item()
        # --- Modified section end ---

    def _save_checkpoint(self, step_or_name: str):
        """Saves the student model's LoRA weights."""
        checkpoint_dir = os.path.join(self.output_dir, str(step_or_name))
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.student_model.save_pretrained(checkpoint_dir)
        logging.info(f"Checkpoint saved: {checkpoint_dir}")

    def train(self):
        """Runs the main training loop."""
        self.student_model.train()
        num_steps = self.training_config['num_train_steps']
        strategy_cfg = self.training_config.get('checkpoint_strategy',
                                            {'policy': 'none', 'value': 0})

        
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(range(num_steps), desc="DCD Training")

        for step in progress_bar:
            try:
                teacher_tokens, student_tokens = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                teacher_tokens, student_tokens = next(data_iterator)

            teacher_inputs = {
                "input_ids": teacher_tokens["input_ids"].to(self.teacher_device),
                "attention_mask": teacher_tokens["attention_mask"].to(self.teacher_device),
            }
            student_inputs = {
                "input_ids": student_tokens["input_ids"].to(self.student_device),
                "attention_mask": student_tokens["attention_mask"].to(self.student_device),
            }
            teacher_labels = teacher_tokens["labels"].to(self.teacher_device)
            student_labels = student_tokens["labels"].to(self.student_device)

            with torch.autocast(device_type="cuda", dtype=self._autocast_dtype, enabled=self._use_autocast):
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**teacher_inputs, output_hidden_states=True)

            with torch.autocast(device_type="cuda", dtype=self._autocast_dtype, enabled=self._use_autocast):
                student_outputs = self.student_model(**student_inputs, output_hidden_states=True)

            loss, kl_loss, l1_loss = self._calculate_dcd_loss(
                teacher_outputs, student_outputs, teacher_labels, student_labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # WandB logging
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/kl_loss": kl_loss,
                "train/l1_loss": l1_loss,
                "train/learning_rate": self.scheduler.get_last_lr()[0],
                "train/step": step + 1
            }
            log_to_wandb(wandb_metrics, step=step + 1)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "KL": f"{kl_loss:.4f}", 
                "L1": f"{l1_loss:.4f}"
            })

            # Save intermediate checkpoint
            if strategy_cfg.get('policy') == "steps" and strategy_cfg.get('value', 0) > 0 and (step + 1) % strategy_cfg['value'] == 0:
                self._save_checkpoint(f"step_{step+1}")
        
        # Save final model
        self._save_checkpoint("final")
        logging.info("Final LoRA saved successfully.")