import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import json

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, prune_wanda_connect, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "wanda_connect"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--modify_layer', nargs="+", type=int, default=[], help='Layer to add bias term / 0 - 31')
    parser.add_argument('--modify_type', nargs="+", type=str, default=[], help='Module type to add bias / q, k, v, o, up, gate, down')
    parser.add_argument('--hetero', action="store_true", help="Whether to modify heterogeneous layers")
    parser.add_argument('--hetero_q', nargs="+", type=int, default=[], help="Layers to modify q_proj")
    parser.add_argument('--hetero_k', nargs="+", type=int, default=[], help="Layers to modify k_proj")
    parser.add_argument('--hetero_v', nargs="+", type=int, default=[], help="Layers to modify v_proj")
    parser.add_argument('--hetero_o', nargs="+", type=int, default=[], help="Layers to modify o_proj")
    parser.add_argument('--hetero_up', nargs="+", type=int, default=[], help="Layers to modify up_proj")
    parser.add_argument('--hetero_gate', nargs="+", type=int, default=[], help="Layers to modify gate_proj")
    parser.add_argument('--hetero_down', nargs="+", type=int, default=[], help="Layers to modify down_proj")
    parser.add_argument('--percentile', type=int, default=None, help='Percentile for bias vector')


    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    # Turn on bias vector
    for module in model.modules():
        if any(isinstance(child, torch.nn.Linear) for child in module.children()):
            for childname, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    new_layer = torch.nn.Linear(in_features=child.in_features, 
                                                out_features=child.out_features, 
                                                bias=True,
                                                dtype=child.weight.dtype,
                                                device=child.weight.device)
                    new_layer.weight.data = child.weight.data.clone()
                    new_layer.bias.data.zero_()
                    del child
                    setattr(module, childname, new_layer)
        torch.cuda.empty_cache()

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda_connect":
            prune_wanda_connect(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")
    if not args.hetero:
        if args.percentile == None:
            os.makedirs(os.path.dirname(f"results/{model_name}/Layer{args.modify_layer}/{args.modify_type}/ppl.txt"), exist_ok=True)
            with open(f"results/{model_name}/Layer{args.modify_layer}/{args.modify_type}/ppl.txt", "w") as f:
                f.write(f"{ppl_test}")
        else:
            os.makedirs(os.path.dirname(f"results/{model_name}/Layer{args.modify_layer}/{args.modify_type}/{args.percentile}th_ppl.txt"), exist_ok=True)
            with open(f"results/{model_name}/Layer{args.modify_layer}/{args.modify_type}/{args.percentile}th_ppl.txt", "w") as f:
                f.write(f"{ppl_test}")
    else:
        str_q = "".join(map(str, args.hetero_q))
        str_k = "".join(map(str, args.hetero_k))
        str_v = "".join(map(str, args.hetero_v))
        str_o = "".join(map(str, args.hetero_o))
        str_up = "".join(map(str, args.hetero_up))
        str_gate = "".join(map(str, args.hetero_gate))
        str_down = "".join(map(str, args.hetero_down))
        filepath = f"results/{model_name}/Hetero/Q{str_q}K{str_k}V{str_v}O{str_o}Up{str_up}Gate{str_gate}Down{str_down}/ppl.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(f"{ppl_test}")
        

    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    # save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    # with open(save_filepath, "w") as f:
    #     print("method\tactual_sparsity\tppl_test", file=f, flush=True)
    #     print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        # task_list = ["hellaswag"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        if not args.hetero:
            if args.percentile == None:
                json.dump(results, open(f"results/{model_name}/Layer{args.modify_layer}/{args.modify_type}/tasks.json", "w"),
                            indent=4)
            else:
                json.dump(results, open(f"results/{model_name}/Layer{args.modify_layer}/{args.modify_type}/{args.percentile}th_tasks.json", "w"),
                            indent=4)
        else:
            str_q = "".join(map(str, args.hetero_q))
            str_k = "".join(map(str, args.hetero_k))
            str_v = "".join(map(str, args.hetero_v))
            str_o = "".join(map(str, args.hetero_o))
            str_up = "".join(map(str, args.hetero_up))
            str_gate = "".join(map(str, args.hetero_gate))
            str_down = "".join(map(str, args.hetero_down))
            filepath = f"results/{model_name}/Hetero/Q{str_q}K{str_k}V{str_v}O{str_o}Up{str_up}Gate{str_gate}Down{str_down}/tasks.json"
            json.dump(results, open(filepath, "w"), indent=4)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)
        

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()