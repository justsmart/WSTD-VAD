# WSTD
This repo contains the Pytorch implementation of our paper:
Weakly Supervised Video Anomaly Detection via Self-Guided Temporal Discriminative Transformer

**Accepted at TCYB 2022.**  



## Training
**Please download the extracted I3d features and checkpoint for ShanghaiTech dataset for a demo:**

>[**ShanghaiTech train i3d onedirve**](https://drive.google.com/file/d/1E5y6KvTapFP3SzOEUAVXmdkHtU0KIOJ4/view?usp=share_link)

>[**ShanghaiTech Checkpoint**](https://drive.google.com/file/d/1uypc52Qd-ad02LeB0EFLzVaJWBVD4vgC/view?usp=share_link)


You should change following files:

(1) Change the file paths to the download datasets above in `list/shanghai-i3d-test.list` and `list/shanghai-i3d-train.list`.

(2) Move checkpoint file into path './ckpt_final/'.

(3) Change the hyperparameters in `src/option.py` if you like.
### Train and test the model
You can run 'python main.py' to train a model, or
run 'python test_cur.py' to test a trained model.



## Citation

If you find this repo useful for your research, please cite our paper:

```bibtex
@article{huang2022weakly,
  title={Weakly Supervised Video Anomaly Detection via Self-Guided Temporal Discriminative Transformer},
  author={Huang, Chao and Liu, Chengliang and Wen, Jie and Wu, Lian and Xu, Yong and Jiang, Qiuping and Wang, Yaowei},
  journal={IEEE Transactions on Cybernetics},
  year={2022},
  publisher={IEEE}
}

```
**Thank https://github.com/tianyu0207/RTFM**
