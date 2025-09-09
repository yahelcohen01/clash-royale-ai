# clash-royale-ai — Your Modified AI Agent

## Project overview

`clash-royale-ai` is an autonomous Clash Royale agent that captures emulator frames, extracts structured game state via object detection, and uses a Deep Q-Network (DQN) implemented in PyTorch to select actions. The agent performs actions in an Android emulator (e.g., BlueStacks) using PyAutoGUI and can be integrated with a Roboflow inference server; Docker may be used to containerize the inference environment.

This README documents the training pipeline, how each component fits together, and links to the official technology documentation.

---

## Repository quick map

Key files in this repository (where relevant code lives):

- `train.py` — training loop and orchestration
- `dqn_agent.py` — PyTorch DQN model, target network, optimizer, and update logic
- `env.py` — environment wrapper: screenshot capture, call to inference, state construction & reward shaping
- `Actions.py` — translates chosen discrete actions into PyAutoGUI mouse/keyboard events
- `requirements.txt` — Python dependencies for the project
- `models/`, `main_images/`, `screenshots/` — assets and model artifacts used during development

(These files and folders are in the repository root.)

---

## Step-by-step training pipeline (implementation-level)

1. **Start supporting services**

   - Start your Roboflow inference server (if you're self-hosting models) or ensure your Roboflow deployment is reachable.
   - Launch BlueStacks and open Clash Royale — make the emulator window front-most and sized/positioned consistently.
   - Activate your Python environment and run the training script: `python train.py`.

2. **Frame capture (environment)**

   - `env.py` captures a screenshot of the emulator window (using an image library, e.g., PIL/OpenCV or a Roboflow client helper). The captured image is the raw input for detection.

3. **Object detection**

   - The frame is sent to the Roboflow inference endpoint (or other detector) which returns detections with bounding boxes, confidence scores, and class labels (cards, troops, towers). These detections convert pixels into a symbolic game state.

4. **State construction**

   - Build a numeric state tensor from detections: normalized positions (x,y), one-hot encodings for cards in hand, tower HP estimates, elixir approximation, and optional occupancy/heatmap channels. Stack recent frames if your agent benefits from temporal context.

5. **Action selection — DQN**

   - The DQN takes the state tensor and outputs Q-values for a discrete action space (for example: choose card index × choose one of several discrete drop locations; include "wait/no-op").
   - Policy: ε-greedy with a decay schedule (start with higher ε for exploration and decay toward exploitation).

6. **Action execution (PyAutoGUI)**

   - Map the selected discrete action to emulator gestures via `Actions.py`. Typical sequence: move mouse to the card UI slot, click/hold, move to the target battlefield coordinates, release/click. Insert human-like timing delays where helpful.

7. **Transition & reward**

   - After action execution, wait for the game to update, capture a new frame, re-run detection, and compute reward.
   - Rewards can include tower damage delta, troop kills, elixir-efficiency terms, and terminal win/loss signals. Reward shaping speeds learning but be careful of unintended incentives.

8. **Replay buffer**

   - Store transitions `(state, action, reward, next_state, done)` in a fixed-size replay buffer. Sample minibatches for training.

9. **Learning updates**

   - Sample batches and compute targets: `y = r + γ * max_a' Q_target(next_s, a')` for non-terminal transitions.
   - Compute loss (e.g., MSE or Huber), backpropagate, and update the online network via an optimizer (commonly Adam).
   - Periodically synchronize or soft-update the target network from the online network.

10. **Logging & checkpointing**

    - Log episode rewards, win-rate, loss curves, and hyperparameters.
    - Save model checkpoints periodically (e.g., every N episodes or when validation performance improves).

11. **Evaluation**
    - Run an inference-only mode with ε near 0 to evaluate behavior and gather play samples or replays.

---

## Roles of the main technologies

- **Python** — Orchestration, environment wrappers, and training scripts.  
  Official docs: https://www.python.org/doc/ . cite{python}

- **PyTorch** — Neural network models, autograd, optimizers, and checkpointing (DQN implementations).  
  Official docs: https://pytorch.org/docs/ . cite{pytorch}

- **Roboflow (Inference Server)** — Object detection models and an inference server/SDK to convert screenshots into bounding boxes and labels. Useful for fast local inference and production-style deployments.  
  Docs & inference: https://inference.roboflow.com/ and https://docs.roboflow.com/ . cite{roboflow}

- **PyAutoGUI** — Controls the mouse and keyboard to send input events to the emulator, translating agent actions into real interactions.  
  Docs: https://pyautogui.readthedocs.io/ . cite{pyautogui}

- **BlueStacks (emulator)** — Runs Clash Royale; the bot targets BlueStacks window coordinates for input events.  
  Support/docs: https://support.bluestacks.com/ . cite{bluestacks}

- **Docker (optional)** — Containerize the inference server and other services for reproducible environments.  
  Docs: https://docs.docker.com/ . cite{docker}

---

## Getting started (recommended)

> ⚠️ **Notice:** Because I used Mac and there is no emulator for Clash Royale here, I used screen mirroring. All the measurements in the project are for my IPhone 14 pro when screen-mirrored to the right side of the screen. If you have different device you'll have to enter this manually.

1. Clone your repo and set up a virtualenv:

```bash
git clone https://github.com/yahelcohen01/clash-royale-ai
cd clash-royale-ai
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Start Roboflow inference, I used docker for self hosting (or point the project to your hosted model). Example CLI (Roboflow):

```bash
# If you're using Roboflow CLI / inference server
inference server start
```

3. Start Screen Mirroring/BlueStacks and open Clash Royale. Ensure consistent window size/position.

4. Train:

```bash
python train.py
```

5. Evaluate / play:

```bash
python play.py --weights checkpoints/latest.pt
```

(Adjust commands if your scripts use different flags. See `train.py` for precise arguments.)

---

## Configuration & tips

- Keep a stable emulator position and resolution to keep coordinate mappings consistent.
- Start with coarse discretization of battlefield drop zones; increase granularity later.
- Use dense rewards (tower damage) early; reduce shaping later to encourage robust strategies.
- Use a GPU for faster training if the model architecture is large.
- Monitor for "reward hacking"—agents finding degenerate ways to maximize shaped rewards.

---

## License & ethical note

This repository uses the MIT license (see `LICENSE` in repo). Be mindful of game ToS and terms of service when automating gameplay. Use this project responsibly.
YOU CAN GET YOUR ACCOUNT BANNED IF SUPERCELL DETECTS THAT ITS A BOT PLAYING

---

## Links to official docs and useful resources

- Python docs — https://www.python.org/doc/ . cite{python_link}
- PyTorch docs — https://docs.pytorch.org/ . cite{pytorch_link}
- PyAutoGUI docs — https://pyautogui.readthedocs.io/ . cite{pyautogui_link}
- Roboflow Inference — https://inference.roboflow.com/ and https://docs.roboflow.com/ . cite{roboflow_link}
- Docker docs — https://docs.docker.com/ . cite{docker_link}
- BlueStacks support — https://support.bluestacks.com/ . cite{bluestacks_link}

---

## Where to look in the code

- `train.py` — training orchestration & CLI entry
- `dqn_agent.py` — model architecture, optimizers, update logic
- `env.py` — screenshot capture, Roboflow client calls, state & reward construction
- `Actions.py` — PyAutoGUI wrappers

---

If you want, I can:

- Extract exact hyperparameters (epsilon schedule, gamma, batch size, buffer size) from `dqn_agent.py` and `train.py` and inject them into a `CONFIG.md` or directly into this README.
- Add badges, CLI usage examples (auto-filled from `train.py` arg parsing), or a minimal `docker-compose.yml` snippet for Roboflow inference.
