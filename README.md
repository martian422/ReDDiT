# [Rehashing Noise For Discrete Visual Generation](http://arxiv.org/abs/2505.19656)

We introduce *ReDDiT*, a **Re**hashing noise Framework for **D**iscrete **Di**ffusion **T**ransformer.

In this repo, we release:
* **The ReDDiT framework.**
  1. Multi-index corruption and derived low-discrepancy rehash sampler (in **diffusion.py**). 
  2. Modified objective with RepA for discrete diffusion processes.
  3. 2D-RoPE, min-SNR-5 modification for modern transformers.
* **Sampler Extension.**
  1. Predict-remask sampler for MaskGIT-style methods (as the baseline).
  2. Discrete Flow Matching sampler (requires more steps for optimal performance).

If you find our rehashing noise or proposed sampler helpful, please consider cite our work, thanks!

### Acknowledgements
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm), with min-SNR reference to [UniDisc](https://github.com/alexanderswerdlow/unidisc), representation alignment from [REPA](https://github.com/sihyun-yu/REPA), extended sampler implementation from [DFM](https://github.com/facebookresearch/flow_matching) and [MaskGIT](https://github.com/google-research/maskgit). Thanks to all the authors for their great work.

## Citation
```
@article{ma2025reddit,
      title={ReDDiT: Rehashing Noise for Discrete Visual Generation}, 
      author={Tianren Ma and Xiaosong Zhang and Boyu Yang and Junlan Feng and Qixiang Ye},
      year={2025},
      eprint={2505.19656},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
