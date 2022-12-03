# DAMP: Doubly Aligned Multilingual Parser for Task-Oriented Dialogue
This repository contains code to evaluate HuggingFace Seq2Seq models on the TOP family of datasets along with Constrained Adversarial Alignment.

Feel free to contact [William Held](https://williamheld.com/) with any questions at wheld3 [@] gatech edu.

For a description of methods, as well as reported results combining AMBER with pre-training alignment - see the [[Paper]](https://openreview.net/forum?id=77DPrCbYZdHl)

## Pre-Requisites
`pip install -r requirements.txt

## Reproducing MT5 Results with Pretrained Models
### mT5 Small 
Non-Adversarial: `bash eval_t5.sh`
Adverserial: `bash eval_t5_adv.sh`
#### mT5 Base (Requires 2 GPUs)
Non-Adversarial: `bash eval_t5_base.sh`
Non-Adversarial: `bash eval_t5_base_adv.sh`

## Training Models From Scratch
### mT5 Small 
Non-Adversarial: `bash train_t5.sh`
Adverserial: `bash train_t5_adv.sh`
#### mT5 Base (Requires 2 GPUs w/ 12GB RAM Each)
Non-Adversarial: `bash train_t5_base.sh`
Non-Adversarial: `bash train_t5_base_adv.sh`

## Constrained Adversarial Alignment Implementation
`alignment_mixin.py`
