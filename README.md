# Flappy Bird Auto

## Overview
Flappy Bird Auto is nothing but [my Python implementation of the classic Flappy Bird game](https://github.com/vinhdq842/Flappy-Bird) plus Deep Q-Learning. The repository provides functionalities for training, selecting checkpoints, running, and observing the Deep Q-Learning agent's performance in navigating the Flappy Bird environment.

## Features
- **Flappy Bird Game:** A Python implementation of the Flappy Bird game.
- **Deep Q-Learning:** The game is enhanced with a Deep Q-Learning agent that autonomously learns to play.
- **Configuration:** Adjust parameters in `configs/config.yaml` for customization.

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/vinhdq842/Flappy-Bird-Auto.git
    cd Flappy-Bird-Auto
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Instant trial
If you want to try it right away, there are two checkpoints for an instant run.

- [full-bg-0001540000.pth](https://drive.google.com/file/d/1ymXyCj9oRtomPMbWXt3ZxdjhokU6Q2Be/view?usp=sharing)
- [no-bg-0000320000.pth](https://drive.google.com/file/d/1M_fB-KboiPsf3GQz1Kgbt5yfhZSoftuh/view?usp=sharing)

Just download, put them into the corresponding `outputs` subfolders, and run:

```bash
python test.py -c configs/the_specified_config_file.yaml
```

### Configuration
Customize the agent's behavior and training parameters in the `configs/config.yaml` file. You can also adjust game settings here.

### Training
Execute the following command to train the Deep Q-Learning agent:

```bash
python train.py
```

This will initiate the training process, and the agent will learn to play the Flappy Bird game autonomously.

### Validation
After that, validate the training results using:

```bash
python validate.py
```

This script assesses how well the trained agents navigate the Flappy Bird environment and tell which is the best checkpoint to use.

### Testing
To see the final performance, substitute `best_checkpoint` in your configuration file with the checkpoint name yielded from validation stage, and execute:

```bash
python test.py
```

This script shows the best trained agent's ability to play the Flappy Bird game.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT](LICENSE) License.

## Acknowledgments
- Inspired by this nice repo [Chrome-dino-deep-Q-learning-pytorch](https://github.com/uvipen/Chrome-dino-deep-Q-learning-pytorch).
- Thanks to Hugging Face Team for their great [🤗 Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction).