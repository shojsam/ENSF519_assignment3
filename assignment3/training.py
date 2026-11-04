"""Loss, training, and evaluation for Assignment 3."""

import math

import torch
from torch.nn import functional as F


def compute_loss(logits, targets):
    """Return the mean cross-entropy loss of the next-token predictions.

    args:
        logits: shape (batch_size, T, vocab_size)
        targets: shape (batch_size, T), torch.long, the next token at every position
    outputs:
        loss: a scalar
    """
    loss = None
    ####################################
    # Your implementation starts here. #
    ####################################
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    ##################################
    # Your implementation ends here. #
    ##################################
    return loss


def evaluate(model, data_loader, device, max_batches=None):
    """Return the mean loss and the perplexity of the model on a dataset.

    args:
        model: the model to evaluate
        data_loader: torch.utils.data.DataLoader that returns batches (x, y)
        device: torch.device, the device to evaluate on
        max_batches: if not None, only evaluate the first max_batches batches
    outputs:
        mean_loss: float, the loss averaged over all evaluated windows
        perplexity: float, exp(mean_loss)
    """
    mean_loss, perplexity = None, None
    ####################################
    # Your implementation starts here. #
    ####################################
    was_training = model.training
    total_loss = 0.0
    num_windows = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(data_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                loss = compute_loss(model(x), y)
                batch_size = x.shape[0]
                total_loss += loss.item() * batch_size
                num_windows += batch_size
    finally:
        model.train(was_training)

    if num_windows == 0:
        raise ValueError("Evaluation requires at least one window.")
    mean_loss = total_loss / num_windows
    perplexity = math.exp(mean_loss)
    ##################################
    # Your implementation ends here. #
    ##################################
    return mean_loss, perplexity


def train_model(model, train_loader, val_loader, num_epochs, device, eval_every=500, **kwargs):
    """Train the model and record the training and validation losses.

    args:
        model: GPT, the model to train
        train_loader: torch.utils.data.DataLoader for the training set
        val_loader: torch.utils.data.DataLoader for the validation set
        num_epochs: int, the number of epochs to train
        device: torch.device, the device to train on
        eval_every: int, evaluate on the first 50 validation batches every eval_every steps
        **kwargs: include any optional keyword argument that may be helpful, which can include:
            - optimizer_kwargs: arguments for the optimizer
            - scheduler_kwargs: arguments for the learning rate scheduler
    outputs:
        history: a dictionary with
            "train_loss": a list of Python floats, the training loss of every step,
            "val_step": a list of the steps at which the model was evaluated,
            "val_loss": a list of Python floats, the validation loss at these steps
    """
    history = {"train_loss": [], "val_step": [], "val_loss": []}
    ####################################
    # Your implementation starts here. #
    ####################################
    total_steps = num_epochs * len(train_loader)
    if total_steps == 0:
        return history
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")

    optimizer_kwargs = {"lr": 1e-3, "weight_decay": 0.1}
    optimizer_kwargs.update(kwargs.get("optimizer_kwargs", {}))
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler_kwargs = {"max_lr": 1e-3, "total_steps": total_steps, "pct_start": 0.05}
    scheduler_kwargs.update(kwargs.get("scheduler_kwargs", {}))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)

    model.to(device)
    model.train()
    step = 0
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            history["train_loss"].append(loss.item())
            if step % eval_every == 0:
                val_loss, _ = evaluate(model, val_loader, device, max_batches=50)
                history["val_step"].append(step)
                history["val_loss"].append(val_loss)
                if kwargs.get("verbose", True):
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, step {step}/{total_steps}: "
                        f"train loss {loss.item():.3f}, val loss {val_loss:.3f}",
                        flush=True,
                    )
    ##################################
    # Your implementation ends here. #
    ##################################
    return history
