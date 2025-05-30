# [Rehashing Noise For Discrete Visual Generation](http://arxiv.org/abs/2406.07524)

We introduce *ReDDiT*, the **Re**hashing noise Framework for **D**iscrete **Di**ffusion **T**ransformer.

In this repo, we release:
* **The ReDDiT framework.**
  1. Multi-index corruption and derived rehash sampler (in **diffusion.py**).
  2. Modified objective with RepA for discrete diffusion processes.
  3. 2D-RoPE, min-SNR-5 and RMSNorm modification for modern transformers.
* **Sampler Extension.**
  1. Predict-remask sampler for MaskGIT-style methods.
  2. Discrete Flow Matching sampler.

We will release the weights for further study. Please star to stay tuned.

By the way, we find that expanding mask indices **more useful** on text-to-image generation training, which will be updated and discussed later. If you use our rehashing noise design, please cite our work, thanks!

### Acknowledgements
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm).

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
