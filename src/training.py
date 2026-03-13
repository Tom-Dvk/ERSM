import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Compute top-1 accuracy (%) on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_engine(model, train_loader, test_loader, optimizer, scheduler,
                 device, epochs, name="Model"):
    """Standard training loop with optional ERSM energy regularisation.

    Returns a dict with keys ``train_loss``, ``test_acc``, ``mean_mask``.
    """
    history = {"train_loss": [], "test_acc": [], "mean_mask": []}
    print(f"--- Starting Training: {name} ---")

    for ep in range(epochs):
        model.train()
        t_loss = 0
        mask_sum = 0
        batches = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, aux = model(imgs)
            loss = F.cross_entropy(logits, labels)

            if aux is not None:
                reg = (aux["keep_mask"] * aux["energy"]).mean()
                loss += reg
                mask_sum += aux["keep_prob"].mean().item()

            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            batches += 1

        scheduler.step()

        avg_loss = t_loss / max(1, batches)
        avg_mask = mask_sum / max(1, batches) if (batches > 0 and aux is not None) else 0.0
        test_acc = evaluate_accuracy(model, test_loader, device)

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(test_acc)
        history["mean_mask"].append(avg_mask)

        print(
            f"[{name}] Ep {ep + 1:02d} | Loss: {avg_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | Mean Mask: {avg_mask:.2f}"
        )

    return history
