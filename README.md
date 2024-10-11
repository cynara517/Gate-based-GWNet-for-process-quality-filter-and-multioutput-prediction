# Gate-based-GWNet-for-process-quality-filter-and-multioutput-prediction
A repository containing the code implementation and the dataset of the Gate-based GWNet model for process quality filter and multioutput prediction, as described in the paper.
# Abstract
Industrial fermentation is a crucial process for producing commonly used drugs like penicillin. The Gate-based GWNet model addresses the challenges of capturing complex relationships and dynamic changes within fermentation processes. It introduces innovative modules for graph structure acquisition, time gating, dynamic graph structure updating, and adaptive filtering of temporal features. The model is evaluated on three industrial fermentation datasets, showcasing superior performance compared to existing models like MTGNN and Autoformer.
# Usage
The model files are located in the Model folder, the Penisim dataset and IndPenisim dataset are in the dataset data set, and the graph structure generation file graph and graph_layers, as well as the training data generation file generate_training, are in the Preprocess folder.
# Dependencies
Model is built based on PyTorch and [EasyTorch](https://github.com/cnstark/easytorch).
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**After ensuring** that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```
# Main Results
See the paper “Gate-based GWNet for process quality filter and multioutput prediction”
