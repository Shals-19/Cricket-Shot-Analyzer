# 🏏 Cricket Shot Analyzer

A smart, real-time web application that analyzes cricket shots using pose estimation and biomechanical metrics. Designed to evaluate cover drive technique, the system segments motion phases, detects bat-ball contact, and generates a detailed HTML report with annotated visuals and performance feedback.

---

## 📌 Features

- **Pose Estimation** using MediaPipe
- **Biomechanical Metrics**:
  - Elbow angle, spine lean, head-knee alignment, foot direction
- **Automatic Phase Segmentation**:
  - Stance → Stride → Downswing → Impact → Follow-through
- **Contact Moment Detection** via wrist velocity spike
- **Temporal Smoothness Chart** for elbow and spine motion
- **Skill Grading**: Beginner, Intermediate, Advanced
- **Modular Evaluation**: Footwork, Head Position, Swing Control, Balance, Follow-through
- **Annotated Video Output** with phase labels and feedback cues
- **HTML Report Generation** with embedded plots and metrics
- **Streamlit Web Interface** for easy video upload and result visualization

---

## 🧠 Core Components

### 🔙 Backend (Python)

- `cover_drive_analysis_realtime.py`: Main analysis pipeline with helper functions
- `report_template.html`: Jinja2 template for HTML report
- `app.py`: Streamlit frontend for user interaction

### 📦 Output Files

- `annotated_video.mp4`: Phase-labeled video with biomechanical overlays
- `contact_frame.png`: Frame showing likely bat-ball contact
- `metrics_plot.png`: Biomechanical trends over time
- `smoothness_chart.png`: Temporal smoothness visualization
- `report.html`: Embedded HTML report
- `evaluation.json`: Modular skill scores and feedback

---

## ⚙️ Tech Stack

| Layer       | Technology            |
|-------------|------------------------|
| Pose Model  | MediaPipe              |
| Processing  | OpenCV, NumPy          |
| Visualization | Matplotlib           |
| Report Engine | Jinja2 (HTML templating) |
| Web Interface | Streamlit            |
| Output Format | MP4, PNG, HTML, JSON |

---

## 📂 Folder Structure

```
Athlete Rise/
├── app.py
├── cover_drive_analysis_realtime.py
├── report_template.html
├── requirements.txt
├── output/
│   ├── annotated_video.mp4
│   ├── contact_frame.png
│   ├── metrics_plot.png
│   ├── smoothness_chart.png
│   ├── report.html
│   └── evaluation.json
└── README.md
```

---

## ▶️ How to Run

1. Create a virtual environment (make sure python version is 3.10)
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment
   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Upload a cricket shot video (`.mp4`) and view:
   - Annotated video
   - Biomechanical metrics
   - Phase-wise plots
   - Impact frame
   - Downloadable report and video

6. Deactivate the virtual environment
   ```bash
   deactivate
   ```
---

## 📈 Metrics Computed

- **Elbow Angle**: Measures arm extension during swing
- **Spine Lean**: Indicates posture and balance
- **Head-Knee Alignment**: Assesses body control and positioning
- **Foot Direction**: Evaluates stance stability
- **Wrist Velocity**: Used for impact detection
- **Smoothness Score**: Frame-to-frame consistency in motion

---


