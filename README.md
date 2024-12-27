
# ARAM: Efficient Action-Constrained Reinforcement Learning via Acceptance-Rejection Method and Augmented MDPs

**Efficient Action-Constrained Reinforcement Learning via Acceptance-Rejection Method and Augmented MDPs**

Wei Hung\*, [Shao-Hua Sun](https://shaohua0116.github.io/), [Ping-Chun Hsieh](https://pinghsieh.github.io/)

![The complete training process of ARAM.](https://github.com/user-attachments/assets/75a80bec-aa66-418b-a6b9-1f28d618fb88)

---

## Code Structure

The codebase is organized as follows:

- `main.py`: Entry point for training ARAM.
- `models/`: Contains neural network implementations for policy and value networks.
- `environments/`: Includes custom environments and action constraints.
- `utils/`: Utility functions for batch operations.
- `Constraint_Check.py`: Used to verify if actions satisfy the constraints.
- `Constraint_Proj.py`: Ensures actions interacting with the environment strictly adhere to constraints using QP computations.

---

## Requirements

- **Python version**: Tested in Python 3.9.12
- **Operating system**: Tested in Ubuntu 20.04
- **PyTorch version**: 1.3.1

Install other required packages:

```bash
pip install -r requirements.txt
```

---

## Basic Usage

### Training

Run the following command to train the model with a specific problem setup:

```bash
python main.py --env_id MO_reacher_L2_005-v0 --prob_id Re+L2_005 --start_steps 1000 --seed 1
```
The `env_id` corresponds to a specific environment.
The `prob_id` corresponds to a specific environment and its associated action constraints.

### Testing

To evaluate a trained model, use:

```bash
python eval.py --env_id MO_reacher_L2_005-v0 --prob_id Re+L2_005 --pref 0.1 0.9 --model_path /tmp/policy.pth 
```

The `env_id` corresponds to a specific environment.
The `prob_id` corresponds to a specific environment and its associated action constraints.
The `pref` defines two preference weights for multi-objective evaluation.

---

## Problem Setup

Define a custom problem by specifying the environment and action constraints. The following table shows the action constraints for various environments:

| **Environment** | **Action Constraint** | **prob_id** |
|------------------|------------------------|-------------|
| HopperVel        | \(\sum_{i=1}^3 \max(w_i a_i, 0) \leq 10\) | H+M_10 |
| Hopper           | \(\sum_{i=1}^3 \max(w_i a_i, 0) \leq 10\) | H+M_10 |
| Reacher          | \(a_1^2 + a_2^2 \leq 0.05\) | Re+L2_005 |
| HalfCheetah      | \(\sum_{i=1}^6 |w_i a_i| \leq 20\) | HC+O_20 |
| Ant              | \(\sum_{i=1}^8 |w_i a_i| \leq 2\) | An+L2_2 |
| NSFnet           | \(\sum_{i \in \text{link}_j} a_i \leq 50, \ a_i^2 \leq 2, \ \forall j \in \{1,2,\ldots,8\}\) | NSFnetV2+S |
| BSS5z            | \(\|\sum_{i=1}^5 a_i - 150\| \leq 5, \ a_i \leq 40\) | BSS5z+S+D40 |
| BSS3z            | \(\|\sum_{i=1}^3 a_i - 90\| \leq 5, \ a_i \leq 40\) | BSS3z+S+D40 |

---

## Reference

- [rltorch](https://github.com/toshikwa/rltorch): A simple framework for reinforcement learning in PyTorch.
