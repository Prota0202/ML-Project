import pandas as pd
import torch
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def perm_importance(pipe, X_val, y_val, n_repeats=5, seed=42):
    pi = permutation_importance(
        pipe, X_val, y_val,
        n_repeats=n_repeats, random_state=seed, scoring="r2", n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": list(X_val.columns),
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def pdp_plot(pipe, X_train, features, out=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    PartialDependenceDisplay.from_estimator(pipe, X_train, features=list(features), ax=ax)
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
    return fig


def saliency_map(model, x, target, device=None):
    """|x ⊙ ∂logit_target/∂x|. Approx rapide d'Integrated Gradients."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = x.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    logits = model(x)
    logits[0, target].backward()
    return (x.detach() * x.grad.detach()).abs().squeeze().cpu().numpy()
