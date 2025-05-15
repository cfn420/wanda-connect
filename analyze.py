import torch, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

layers = range(32)
types = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"]
indices = range(0, 128, 8)
os.makedirs("tempfig2", exist_ok=True)

for percentile_ratio in range(60, 95, 5):
    for layer in layers:
        for module_type in types:
            data = torch.empty((4096 if module_type != "mlp.down_proj" else 11008, 4096 * len(indices)), device=0)
            for index in indices:
                fname = f"Layer{layer}_{module_type}_inp{index}"
                data[:, 512 * index: 512 * index + 4096] = torch.load(f"Tensors/{fname}.pt", weights_only=True).to(device=0)

            stds = torch.std(input=data, dim=1).detach().cpu()
            percentile = np.percentile(stds, percentile_ratio)
            percentile_mask = stds < percentile
            os.makedirs(os.path.dirname(f"percentiles/Layer{layer}/{module_type}/{percentile_ratio}th_mask.pt"), exist_ok=True)
            torch.save(percentile_mask, f"percentiles/Layer{layer}/{module_type}/{percentile_ratio}th_mask.pt")
            # hist = plt.hist(stds, bins=150)
            # plt.title(f"Layer {layer} | {module_type} | {index}th input")
            # plt.savefig(f"tempfig2/Layer{layer}_{module_type}.png")
            # plt.clf()
            del data
            gc.collect()
            torch.cuda.empty_cache()
        