# Flappy Bird Auto

## Overview

Flappy Bird Auto is nothing but [Python implementation of the classic Flappy Bird game](https://github.com/vinhdq842/Flappy-Bird) plus Deep Q-Learning. The repository provides functionalities for training, validating, and testing the Deep Q-Learning agent's performance in navigating the Flappy Bird environment.

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

### Training

Execute the following command to train the Deep Q-Learning agent:

```bash
python train.py
```

This will initiate the training process, and the agent will learn to play the Flappy Bird game autonomously.

### Validation
After training, validate the agent's performance using:

```bash
python validate.py
```

This script assesses how well the trained agent navigates the Flappy Bird environment.

### Testing
To test the agent's performance, execute:

```bash
python test.py
```

This script evaluates the trained agent's ability to play the Flappy Bird game.

## Configuration
Customize the agent's behavior and training parameters in the `configs/config.yaml` file. You can also adjust game settings here.

## Demo


## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT](LICENSE) License.