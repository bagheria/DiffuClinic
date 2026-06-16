
import math
import os

import torch
from torch.amp import GradScaler, autocast


def train(model, train_loader, compute_loss, config: dict, tokenizer) -> None:
    training = config["training"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = training["num_epochs"]
    batch_size = training["batch_size"]
    grad_accum_steps = training["grad_accum_steps"]
    warmup_steps = training["warmup_steps"]
    grad_clip_norm = training["grad_clip_norm"]
    log_every = training["log_every"]
    save_every = training["save_every"]

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training["learning_rate"], weight_decay=training["weight_decay"])

    total_steps = math.ceil(len(train_loader.dataset) / batch_size) * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    primary_device = next(model.parameters()).device

    use_amp = bool(training.get("fp16", False))
    scaler = GradScaler("cuda") if use_amp else None

    model.train()
    global_step = 0
    running_loss = 0.0

    print(f"Training - {num_epochs} epochs, {total_steps} total steps")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")
    print("-" * 60)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {key: value.to(primary_device) for key, value in batch.items()}

            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    loss = compute_loss(model, batch, config)
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                loss = compute_loss(model, batch, config)
                (loss / grad_accum_steps).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = running_loss / (log_every * grad_accum_steps)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1} | Step {global_step:>5} | Loss {avg_loss:.4f} | LR {lr_now:.2e}")
                    running_loss = 0.0

                if global_step % save_every == 0:
                    ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  Checkpoint saved -> {ckpt}")

    print("\nTraining complete.")
