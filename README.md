# Implementation of 3 RL Algorithms

### Installation
First, install box2d from source:
```bash
git clone https://github.com/jonasschneider/box2d-py/
cd box2d-py
python setup.py build
python setup.py install
```

Then, install visdom from source:
```bash
git clone https://github.com/facebookresearch/visdom
cd visdom
pip install -e .
easy_install .
```

Install [TensorFlow](https://www.tensorflow.org/install/) and [PyTorch](http://pytorch.org/) following the instructions on the website.

Finally, check all required packages have been installed by 
```bash
pip install -r requirements.txt
```

### Imitation Learning
Train a cloned policy by
```bash
python imiatation.py --train --num_episodes [NUM_EPISODES] --log_dir [LOG_DIR]
```
where `NUM_EPISODES` is the number of expert episodes in the training set.

Run `python imitation.py -h` for more options.

### REINFORCE
First, start a visdom server by
```bash
python -m visdom.server
```
To train a REINFORCE agent with default parameters, run
```bash
python reinforce.py --task_name [TASK_NAME]
```
You can navigate to <http://localhost:8097> to see the learning curve and other plots.

Run `python reinforce.py -h` to see the hyperparameters that can be set.

### Advantage Actor-Critic
First, start a visdom server by
```bash
python -m visdom.server
```
To train an A2C agent with default parameters, run
```bash
python a2c.py --task_name [TASK_NAME] -n [N]
```
where `N` is the number bootstrapping steps.

You can navigate to <http://localhost:8097> to see the learning curve and other plots.

Run `python a2c.py -h` to see the hyperparameters that can be set.
