# SHAP Interpretability for a PPO CartPole Agent

A Python project showing how to use **SHAP (SHapley Additive exPlanations)** to interpret a **PPO** (Proximal Policy Optimization) agent trained on **CartPole-v1**.  
The workflow: train a PPO agent (RLlib), extract PyTorch policy logits, then apply **SHAP DeepExplainer** to visualize how each state feature
(cart position, cart velocity, pole angle, pole angular velocity) affects the decision to move **left** or **right**.

---

## 🛠 Features
- **Training:** PPO agent via RLlib on CartPole-v1
- **Interpretability:** SHAP DeepExplainer on the PyTorch policy network
- **Logits Extraction:** Small wrapper to make RLlib policy SHAP-friendly
- **Visualization:** Action-specific SHAP summary plots

---

## 📂 Project Structure
- `cartpole_ppo.py` — Trains and saves the PPO agent
- `interpret_cp.py` — Loads the trained agent, runs SHAP, and saves plots
- `requirements.txt` — Python dependencies
- `images/`
  - `shap_left.png` — SHAP summary for “move left”
  - `shap_right.png` — SHAP summary for “move right”

---

## 📦 How to Run

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# 1) Train the PPO agent (saves policy checkpoints or weights)
python cartpole_ppo.py

# 2) Run SHAP interpretation (saves plots to ./images)
python interpret_cp.py
