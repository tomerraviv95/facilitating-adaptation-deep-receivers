import torch


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    return 1 - bits_acc
