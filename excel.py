import pandas as pd
import os, json

filename = "results.xlsx"
model_name = "Llama-2-7b-hf"
cols = ["ppl", "boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
rows = [f"Layer {i}" for i in range(32)]
rows.insert(0, "Default")
excelwriter = pd.ExcelWriter(f"excel/results.xlsx", engine="xlsxwriter")

settings = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
for setting in settings:
    df = pd.DataFrame(columns=cols, index=rows)
    
    # Record Default Wanda Data
    default_dirname = f"results/{model_name}/Layer[]/[]"
    with open(f"{default_dirname}/ppl.txt", "r") as f:
        df.at["Default", "ppl"] = float(f.readline())
    with open(f"{default_dirname}/tasks.json") as f:
        data = json.load(f)
        for key in data['results'].keys():
            df.at["Default", key] = data['results'][key]["acc"]

    # Record Experimental Data
    for layer in range(32):
        dirname = f"results/{model_name}/Layer[{layer}]/['{setting}']"
        with open(f"{dirname}/ppl.txt", "r") as f:
            df.at[f"Layer {layer}", "ppl"] = float(f.readline())
        with open(f"{dirname}/tasks.json") as f:
            data = json.load(f)
            for key in data['results'].keys():
                df.at[f"Layer {layer}", key] = data['results'][key]["acc"]
    df.to_excel(excel_writer=excelwriter, sheet_name=setting, float_format="%.16f")

excelwriter.close()

        
