# 10-703 Deep Reinforcement Learning Assignment 3
First, install PyTorch from <http://pytorch.org/>.

Then, install visdom from source:
```bash
git clone https://github.com/facebookresearch/visdom
cd visdom
pip install -e .
easy_install .
```

Start a visdom server by
```bash
python -m visdom.server
```

Navigate to <http://localhost:8097> to see the plots.