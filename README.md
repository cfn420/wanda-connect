# Wanda-CoNNect: Connectivity-Enhanced Wanda Pruning for LLMs
This repo contains a connectivity integration in **Wanda**, a pruning method for LLMs, as presented in the paper:

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI  <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

This repository is a modified version of the original [Wanda](https://github.com/locuslab/wanda) project, which is licensed under the MIT License. While the code is available for reuse under the same license, I kindly request that the modifications introduced in Wanda-CoNNect not be used in academic publications without prior consent. Please contact me if you're interested in collaborating or citing this work: c.p.c.franssen [at] vu.nl.

This work builds on the outstanding foundation laid by the authors of **Wanda**, and I gratefully acknowledge their generosity in open-sourcing their code.

--- 

## 🔧 Modifications by Christian Franssen
Our integration exclusively modifies the pruning critera of the up-projection in the Llama MLP:
```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```
We simply changed the original pruning criteria for the up-projection from 'S_{ij}=|W_{ij}|\cdot\|X_{j}\|_{2}' (original Wanda) to 

`S_{ij}=|W_{ij}|\cdot\sqrt{\sum_{n=1}^{\text{calsamples}}\sum_{k=1}^{\text{seqlength}}|X_{jk}^{n}|\cdot|X_{ik}^{n}|}.`

Here, `X_{jk}^{n}` is the input activation for the `k`th token in the $n$th sample of the calibration set, and `X_{ik}^{n}` is the *output activation of the gate-projection* for the `k`th token in the `n`th sample of the calibration set. This score encourages retention of weights where input and output channels **co-activate**, improving the functional connectivity of the pruned network.

We show the results (for 3 runs) in the following tables for various calibration set sizes and see that even with a modest integration of connectivity, we are able to improve upon Wanda.

**Mean (STD) perplexity on Llama-7b.**
|Pruning|Method|1|4|16|64|128|
|:-|:-|:-|:-|:-|:-|:-|
|2:4|Wanda|12.46 (0.23)|11.98 (0.20)|11.71 (0.33)|11.62 (0.08)|11.56 (0.06)|
|2:4|CoNNect|**12.22 (0.24)**|**11.68 (0.14)**|**11.36 (0.24)**|**11.43 (0.13)**|**11.23 (0.05)**|
|4:8|Wanda|9.05 (0.16)|8.81 (0.04)|8.74 (0.06)|**8.69 (0.00)**| **8.63 (0.04)**|
|4:8|CoNNect|**8.91 (0.14)**|**8.67 (0.02)**|**8.64 (0.04)**|8.71 (0.02)|**8.63 (0.03)**|

**Mean (STD) perplexity on Llama-2-7b.**
|Pruning|Method|1|2|8|16|32|
|:-|:-|:-|:-|:-|:-|:-|
|2:4|Wanda|12.48 (0.46)|11.99 (0.32)|11.73 (0.32)|11.55 (0.14)|11.48 (0.11)|
|2:4|CoNNect | **11.66 (0.33)**| **11.40 (0.30)**| **11.28 (0.25)**|**11.22 (0.11)**|**11.09 (0.06)**|
|4:8|Wanda|8.41 (0.13)|8.29 (0.12)|8.22 (0.11)|8.17 (0.06)|8.14 (0.04)|
|4:8|CoNNect|**8.18 (0.12)**|**8.05 (0.10)**|**8.04 (0.09)**|**8.06 (0.05)**|**8.01 (0.03)**|

## ⚙️ Setup
Installation instructions are equivalent to Wanda instructions and can be found in [INSTALL.md](INSTALL.md).

## 🚀 Usage
Run [`scripts/llama_7b.sh`](scripts/llama_7b.sh) to prune the LLaMA-7B model using Wanda-CoNNect.

## 🙏 Acknowledgement
This repository is built upon the [Wanda](https://github.com/locuslab/wanda) repository.

## 🛡️ License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## 🤝 Contact
For questions or contributions, please email c.p.c.franssen [at] vu.nl or open an issue.