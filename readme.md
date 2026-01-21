# Project: Learning-based Adaptive Uncertainty Estimation for Robust DOGM

**A Deep Learning Framework for Sensor Fusion in Mobile Robots using Spatio-Temporal Heterogeneous GAT (ST-HGAT)**

---

## 1. Problem Statement & Motivation
This research aims to overcome the limitations of **Fixed Uncertainty Modeling** in sensor fusion systems for autonomous mobile robots.

### 1.1. Limitations of Static Modeling
Traditional DOGM algorithms assume sensor noise ($\sigma$) is **constant**, regardless of environmental context. This leads to critical failures:
* **Ghost Vulnerability:** Ghost objects caused by Radar multipath are treated with the same confidence as real objects, creating false particles in empty spaces.
* **Doppler Blindness:** When an object moves **tangentially** (crossing the sensor's field of view), the Radial Velocity ($v_r$) drops to zero. Static models misclassify this as a static object, causing track loss.

### 1.2. Research Goal
We propose an **Adaptive Uncertainty Estimation** framework using Deep Learning.
* **Dynamic Inference:** Real-time estimation of **dynamic standard deviation ($\sigma_{point}$)** for each measurement point based on environmental context.
* **Robustness:** Effectively filters ghost noise and maintains tracking performance even in Doppler blind spots.

---

## 2. Methodology: ST-HGAT
We utilize a **Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT)** to fuse LiDAR and Radar data and learn **Heteroscedastic Aleatoric Uncertainty**.

### 2.1. Network Architecture (3-Branch Structure)
To ensure physical consistency, the network treats `LiDAR`, `Radar1` (Left), and `Radar2` (Right) as distinct nodes.

1.  **Feature Embedding (MLP):**
    * Expands sensor inputs into 64-dim feature vectors ($h_L, h_{R1}, h_{R2}$).
    * **Odom** information is implicitly used for coordinate transformation, not as direct input.
2.  **3-Layer ST-HGAT (Context Learning):**
    * **Layer 1 (Local):** Extracts intra-sensor features (Geometry & Signal Quality).
    * **Layer 2 (Cross-Modal):** Exchanges information between LiDAR and Radars (LiDAR provides velocity context; Radar provides location context).
    * **Layer 3 (Temporal):** Learns **Motion Consistency** by connecting 4 historical frames ($t-3 \sim t$).
3.  **Decoupled Output Heads:**
    * **LiDAR Head:** $\rightarrow (\hat{\mu}_{pos}, \hat{\sigma}_{pos})$
    * **Radar Heads:** $\rightarrow (\hat{\mu}_{vel}, \hat{\sigma}_{vel})$

### 2.2. Multi-Task Learning with Decoupled Loss
We apply **Kendall’s Weighting** to balance position and velocity regression tasks automatically.

$$
\mathcal{L}_{Total} = \frac{1}{2 e^{s_1}} \mathcal{L}_{LiDAR}(\text{Pos}) + \frac{1}{2 e^{s_2}} \mathcal{L}_{Radar1}(\text{Vel}) + \frac{1}{2 e^{s_3}} \mathcal{L}_{Radar2}(\text{Vel}) + \frac{1}{2}(s_1 + s_2 + s_3)
$$

---

## 3. Core Physics: Coordinate System Logic
To handle multi-modal data correctly, we apply a specific coordinate transformation logic.

| Data Type | Source | Frame | Processing Logic |
| :--- | :--- | :--- | :--- |
| **Position ($x, y$)** | LiDAR, Radar | **Base_Link** (Robot) | **Ego-Motion Compensated** using Odom ($T_{rel} \times P_{past}$). |
| **Velocity ($v_r$)** | Radar | **Sensor Local** | **Raw Value Maintained**. Doppler velocity is a scalar and cannot be rotated. |
| **Odom** | Odometry | **Global** | Used only for calculating $T_{rel}$ to align past frames to the current frame. |

---

## 4. Key Capability: Overcoming Doppler Blindness
This framework resolves the tangential motion issue using the following logic:

### 4.1. The Conflict
* **Radar Input:** Measures $v_r \approx 0$ (Static) with High SNR (High Confidence).
* **LiDAR Input:** Observe continuous **position shift ($\Delta x$)** over 4 frames ($t-3 \sim t$).

### 4.2. Resolution Mechanism
The ST-HGAT learns the pattern: *"Velocity is 0, but position is shifting, and signal is clear."*
1.  **Temporal Consistency Check:** Layer 3 detects the position shift from historical frames.
2.  **Adaptive Decision:** The network learns that LiDAR's position change is more reliable than Radar's zero velocity in this specific context.
3.  **Result:**
    * LiDAR Head outputs **Low $\sigma_{pos}$** (High Certainty).
    * Radar Head outputs **High $\sigma_{vel}$** (Low Certainty).

### 4.3. Impact on DOGM
During the particle filter update, the **sharp likelihood of LiDAR** (due to low $\sigma_{pos}$) dominates the update rule, preventing particle death and maintaining the track.

---

## 5. Input Data Specification
Data consists of 3 types of text files logged at 10Hz.
* **Structure:** Columns separated by **Space**, Frames separated by **New line**.
* **Temporal Stacking:** Input includes current frame $t$ + past frames $t-3$.

| File Type | Columns | Description |
| :--- | :--- | :--- |
| `LiDARMap_BaseScan` | `[t, x, y, I]` | Local Coordinates + Intensity |
| `Radar[1/2]Map_BaseScan` | `[t, x, y, vr, SNR]` | **Base_Scan Coordinates** for $x, y$ + **Raw Doppler Velocity** ($v_r$) in Sensor Local Coordinates |
| `odom_filtered` | `[t, x, y, yaw, v, w]` | Global Pose (for time alignment) |

---

## 6. System Integration
The trained network acts as a pre-processor for the DOGM pipeline.

1.  **Inference:** Input Sensor Data (10Hz) $\rightarrow$ Predict $\sigma_{pos}, \sigma_{vel}$.
2.  **Update Weights:** Plug predicted $\sigma$ into the Particle Filter Likelihood function.

$$
P(Z|X) \propto \exp \left( - \frac{(Meas - Pred)^2}{2 \cdot \sigma_{predicted}^2} \right)
$$

---

## 7. Project Structure
```text
ST_learning/
├── config/
│   └── params.yaml       # Configuration (Data versions, Hyperparams)
├── data/
│   ├── Basescan/         # Sensor Data (LiDAR, Radar1, Radar2)
│   └── odom_filtered_*.txt # Odometry Data
├── output/               # Saved Models (.pth)
├── src/
│   ├── dataset.py        # 3-Branch Loader & Coordinate Logic
│   ├── model.py          # ST-HGAT Architecture
│   ├── loss.py           # Multi-Task Loss
│   └── utils.py          # Dynamic Graph Generation
└── train.py              # Training Script
```