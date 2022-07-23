Code for the paper "Cross-device OCTA Generation by Patch-based 3D Multi-
scale Feature Adaption"

## Parameters
The following data parameters are required for training:
```python
# in ./BaseProcess/datasetting.py
parser.add_argument('--train_A_root', type=str, default='source domain OCT cube dir')
parser.add_argument('--train_C_root', type=str,default='source domain OCTA cube dir')
parser.add_argument('--train_B_root', type=str, default='target domain OCT cube dir')
parser.add_argument('--result_root', type=str, default='result save dir')
parser.add_argument('--tensorboard_log_dir', type=str, default='required')

# in ./main/train_p1.py or ./main/train_p2.py
parser.add_argument('--exp_name', type=str, default='required')
parser.add_argument('--result_dir', type=str, default='required')

# in ./main/test_p1.py or ./main/test_p2.py
parser.add_argument('--model_save_dir', type=str, default='xxx/checkpoints')

# two specific parameters for the second-stage training
# in  ./main/train_p2.py, line 43
A2_root = 'required; the generated S->T image'
# in  ./main/trainer_p2.py, line 32
pretrained_path = 'required; stage-1 checkpoint'
```
For the setting of other parameters, please refer to the paper.

## Data structure
Our `./datasets/dataset_for_3d` requires the following data structure for each domain images:
```python
- train_A_root
    - cube 1
      ...
    - cube N # total N OCT or OCTA cubes.
        - 1.png # total M bscan images.
        - 2.png
          ...
        - M.png
```

## Environment
please refer to `requirements.txt`

