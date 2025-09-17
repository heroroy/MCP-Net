---

# 📄 `README.md`

```markdown
# MCP-Net: Enhancing Rheumatoid Arthritis Detection in Metacarpophalangeal Joints

Official implementation of the paper:  
**"MCP-Net: Enhancing Rheumatoid Arthritis Detection in Metacarpophalangeal Joints through Global Context Integration and Attention Mechanisms"**  
_Submitted to The Visual Computer, 2025_

---

## 🔹 Overview
MCP-Net is a deep learning framework for detecting rheumatoid arthritis (RA) in **Metacarpophalangeal (MCP) joints** using ultrasound images.  
The architecture integrates:

- **ResNet50 backbone** for robust feature extraction  
- **Global Context integration** for joint-level representation  
- **Channel & Spatial Attention mechanisms** for enhanced feature selection  
- **Classification head** for final decision making  

This combination improves RA detection performance by leveraging both **global contextual cues** and **fine-grained local attention**.

---

## 📂 Repository Structure
```

MCP-Net/
├─ src/
│  ├─ models.py    # MCP-Net architecture
│  ├─ dataset.py   # Dataset loader & preprocessing
│  ├─ train.py     # Training loop
│  ├─ eval.py      # Evaluation & metrics
│  └─ utils.py     # Helper functions
├─ requirements.txt
├─ README.md

````

---

## ⚙️ Installation & Requirements
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/MCP-Net.git
   cd MCP-Net
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a virtual environment:

   ```bash
   python -m venv mcpnet_env
   source mcpnet_env/bin/activate   # Linux/macOS
   mcpnet_env\Scripts\activate      # Windows
   ```

---

## 📊 Dataset

* The framework is designed for **ultrasound MCP joint images**.
* Training data should be organized as:

  ```
  data/
  ├─ train/
  │  ├─ class_0/
  │  ├─ class_1/
  ├─ val/
  │  ├─ class_0/
  │  ├─ class_1/
  ├─ test/
     ├─ class_0/
     ├─ class_1/
  ```
* Replace `class_0`, `class_1` with your dataset labels (e.g., *healthy*, *RA*).

---

## 🚀 Usage

### Train the Model

```bash
python src/train.py --data_root ./data --epochs 50 --batch_size 32 --save_dir results/
```

### Evaluate the Model

```bash
python src/eval.py --model_path results/best_model.h5 --data_root ./data/test
```

### Expected Outputs

* Model checkpoints in `results/checkpoints/`
* Training logs in `results/logs/`
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

---

## 📑 Citation

If you use this code, please cite:

```
H. Roy et al.,
"MCP-Net: Enhancing Rheumatoid Arthritis Detection in Metacarpophalangeal Joints
through Global Context Integration and Attention Mechanisms",
submitted to The Visual Computer, 2025.
```

**DOI (code & data):** \[insert Zenodo DOI here]
**GitHub repo:** [https://github.com/](https://github.com/)<your-username>/MCP-Net

---

## 📌 Notes

* This code is directly associated with the manuscript submitted to *The Visual Computer*.
* Readers are encouraged to cite the manuscript when using this repository.
* For long-term archival, the repository is mirrored on Zenodo.

---

## 🙏 Acknowledgements

* TensorFlow/Keras for deep learning framework
* ResNet backbone pretrained on ImageNet
* Ultrasound dataset (institutional source; data sharing subject to ethical approval)

---

```

---

👉 Do you want me to also prepare a **`requirements.txt`** (with TensorFlow/Keras + dependencies) and a **MIT `LICENSE` file** so your repo is fully ready for GitHub + Zenodo archival?
```
