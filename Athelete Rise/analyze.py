import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import base64
from jinja2 import Environment, FileSystemLoader

# ---------- Helper Functions ----------

def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def compute_spine_lean(hip, shoulder):
    spine_vector = np.array(shoulder) - np.array(hip)
    vertical_vector = np.array([0, -1])  # y-axis points down
    cosine = np.dot(spine_vector, vertical_vector) / np.linalg.norm(spine_vector)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def compute_foot_direction(toe, heel):
    foot_vector = np.array(toe) - np.array(heel)
    x_axis = np.array([1, 0])
    cosine = np.dot(foot_vector, x_axis) / np.linalg.norm(foot_vector)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def compute_head_knee_alignment(head, knee):
    return abs(head[0] - knee[0])  # x-axis distance


def extract_keypoints(frame, pose_model, width, height):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)
    if not results.pose_landmarks:
        return None
    keypoints = {
        i: (int(lm.x * width), int(lm.y * height))
        for i, lm in enumerate(results.pose_landmarks.landmark)
    }
    return keypoints, results.pose_landmarks

def compute_metrics(keypoints):
    elbow_angle = compute_angle(keypoints[11], keypoints[13], keypoints[15])
    spine_lean = compute_spine_lean(keypoints[23], keypoints[11])
    head_knee_alignment = compute_head_knee_alignment(keypoints[0], keypoints[25])
    foot_direction = compute_foot_direction(keypoints[27], keypoints[29])

    return {
        "elbow_angle": elbow_angle,
        "spine_lean": spine_lean,
        "head_knee_alignment": head_knee_alignment,
        "foot_direction": foot_direction
    }


def annotate_frame(frame, metrics, landmarks, mp_pose, mp_drawing):
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

    height, width, _ = frame.shape

    cv2.putText(frame, f"Elbow Angle: {metrics['elbow_angle']:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Spine Lean: {metrics['spine_lean']:.2f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Head-Knee Align: {metrics['head_knee_alignment']:.2f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Foot Direction: {metrics['foot_direction']:.2f}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    # Feedback cues
    if metrics["elbow_angle"] > 160:
        cv2.putText(frame, "Good elbow elevation", (30, height-120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Elbow too low", (30, height-120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if metrics["head_knee_alignment"] > 50:
        cv2.putText(frame, "Head not over front knee", (30, height-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Good head alignment", (30, height-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if 10 <= metrics["spine_lean"] <= 30:
        cv2.putText(frame, "Balanced spine lean", (30, height-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif metrics["spine_lean"] > 50:
        cv2.putText(frame, "Upright posture", (30, height-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Excessive lean", (30, height-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    if metrics["foot_direction"] < 30:
        cv2.putText(frame, "Stable foot direction", (30, height-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif metrics["foot_direction"] < 60:
        cv2.putText(frame, "Slight misalignment", (30, height-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Poor foot direction", (30, height-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def plot_metrics(metric_list, output_path):
    frames = range(len(metric_list))
    elbow = [m["elbow_angle"] for m in metric_list]
    spine = [m["spine_lean"] for m in metric_list]
    head_knee = [m["head_knee_alignment"] for m in metric_list]
    foot = [m["foot_direction"] for m in metric_list]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(frames, elbow, color='green')
    axs[0, 0].set_title("Elbow Angle Over Time")
    axs[0, 0].set_ylabel("Degrees")

    axs[0, 1].plot(frames, spine, color='blue')
    axs[0, 1].set_title("Spine Lean Over Time")
    axs[0, 1].set_ylabel("Degrees")

    axs[1, 0].plot(frames, head_knee, color='red')
    axs[1, 0].set_title("Head-Knee Alignment Over Time")
    axs[1, 0].set_ylabel("Pixels")

    axs[1, 1].plot(frames, foot, color='orange')
    axs[1, 1].set_title("Foot Direction Over Time")
    axs[1, 1].set_ylabel("Degrees")

    for ax in axs.flat:
        ax.set_xlabel("Frame")

    plt.tight_layout()
    plt.savefig(output_path)



def grade_skill(metrics):
    elbow = metrics["elbow_angle"]
    spine = metrics["spine_lean"]
    head_knee = metrics["head_knee_alignment"]
    foot = metrics["foot_direction"]

    score = 0
    score += 2 if 160 <= elbow <= 180 else 1 if 130 <= elbow < 160 else 0
    score += 2 if 10 <= spine <= 30 else 1 if 30 < spine <= 50 else 0
    score += 2 if head_knee <= 20 else 1 if head_knee <= 50 else 0
    score += 2 if foot <= 30 else 1 if foot <= 60 else 0

    if score >= 7:
        return "Advanced"
    elif score >= 4:
        return "Intermediate"
    else:
        return "Beginner"




def generate_report(metrics, grade, image_paths: dict, output_path):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('report_template.html')

    # Convert all images to base64
    encoded_images = {}
    for key, path in image_paths.items():
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            encoded_images[key] = f"data:image/png;base64,{encoded}"

    html = template.render(metrics=metrics, grade=grade, images=encoded_images)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

        

def final_evaluation(metrics) -> dict:
    score = {}
    feedback = {}

    # Footwork: based on foot direction
    fd = metrics["foot_direction"]
    if fd < 30:
        score["Footwork"] = 9
        feedback["Footwork"] = "Stable foot alignment toward crease."
    elif fd < 60:
        score["Footwork"] = 6
        feedback["Footwork"] = "Foot slightly misaligned; adjust stance."
    else:
        score["Footwork"] = 3
        feedback["Footwork"] = "Poor foot direction; affects balance."

    # Head Position: based on head-knee alignment
    hka = metrics["head_knee_alignment"]
    if hka < 20:
        score["Head Position"] = 9
        feedback["Head Position"] = "Head well aligned over knee."
    elif hka < 50:
        score["Head Position"] = 6
        feedback["Head Position"] = "Moderate misalignment; lean forward."
    else:
        score["Head Position"] = 3
        feedback["Head Position"] = "Head too far back; affects control."

    # Add similar logic for Swing Control, Balance, Follow-through
    # For now, use elbow and spine lean as proxies

    score["Swing Control"] = 8 if 130 <= metrics["elbow_angle"] <= 160 else 5
    feedback["Swing Control"] = "Controlled elbow angle during swing."

    score["Balance"] = 8 if 10 <= metrics["spine_lean"] <= 30 else 5
    feedback["Balance"] = "Good spine lean indicates balance."

    score["Follow-through"] = 7  # Placeholder
    feedback["Follow-through"] = "Follow-through looks consistent."

    return {"scores": score, "feedback": feedback}


def save_evaluation(evaluation, path="output/evaluation.json"):
    import json
    with open(path, "w") as f:
        json.dump(evaluation, f, indent=4)

def compute_joint_velocity(joint_positions):
    velocities = []
    for i in range(1, len(joint_positions)):
        prev = np.array(joint_positions[i - 1])
        curr = np.array(joint_positions[i])
        velocity = np.linalg.norm(curr - prev)
        velocities.append(velocity)
    return velocities


# ---------- Main Function ----------

def analyze_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frames, metrics_list = [], []
    wrist_positions = []
    elbow_angles = []
    spine_leans = []
    keypoints_over_time = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = extract_keypoints(frame, pose, width, height)
        if result:
            keypoints, landmarks = result
            wrist_positions.append(keypoints[15])  # Left wrist
            
            keypoints_over_time.append(keypoints)
            metrics = compute_metrics(keypoints)
            elbow_angles.append(metrics["elbow_angle"])
            spine_leans.append(metrics["spine_lean"])
            metrics_list.append(metrics)
            frame = annotate_frame(frame, metrics, landmarks, mp_pose, mp_drawing)
        frames.append(frame)
    
    cap.release()

    
    #wrist_speeds = compute_joint_velocity(keypoints_over_time)
    wrist_velocities = compute_joint_velocity(wrist_positions)

    # Heuristic segmentation
    phases = []
    for i in range(1, len(elbow_angles)):
        delta_elbow = elbow_angles[i] - elbow_angles[i - 1]
        delta_spine = spine_leans[i] - spine_leans[i - 1]
        wrist_v = wrist_velocities[i] if i < len(wrist_velocities) else 0

    # Prioritize impact detection
        if wrist_v >= 30:
            phases.append("Impact")
        elif delta_elbow > 1 and wrist_v >= 2:
            phases.append("Downswing")
        elif delta_spine > 2 and abs(delta_elbow) < 5:
            phases.append("Stride")
        elif abs(delta_elbow) < 2 and abs(delta_spine) < 1 and wrist_v < 10:
            phases.append("Stance")
        elif wrist_v < 10 and delta_elbow < 2:
            phases.append("Follow-through")
        else:
            phases.append("Transition")
    phases.insert(0, "Stance")


    # Find peak wrist velocity within downswing
    impact_indices = [i for i, p in enumerate(phases) if p == "Impact"]
    impact_velocities = [wrist_velocities[i] for i in impact_indices if i < len(wrist_velocities)]

    if impact_velocities:
        local_peak_index = np.argmax(impact_velocities)
        impact_frame_index = impact_indices[local_peak_index]
        impact_frame = frames[impact_frame_index]
        cv2.putText(impact_frame, "Likely Contact Moment", (30, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite("output/contact_frame.png", impact_frame)
    else:
        impact_frame_index = np.argmax(wrist_velocities)  # fallback
        impact_frame = frames[impact_frame_index]
        cv2.putText(impact_frame, "Likely Contact Moment", (30, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite("output/contact_frame.png", impact_frame)

    cv2.putText(frame, f"Phase: {phases[i]}", (30, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save video
    out_path = "output/annotated_video.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    for i, f in enumerate(frames):
        if i < len(phases):
            cv2.putText(f, f"Phase: {phases[i]}", (60,470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out.write(f)


    # Save plot
    plot_path = "output/metrics_plot.png"
    plot_metrics(metrics_list, plot_path)

    elbow_deltas = [abs(elbow_angles[i] - elbow_angles[i - 1]) for i in range(1, len(elbow_angles))]
    spine_deltas = [abs(spine_leans[i] - spine_leans[i - 1]) for i in range(1, len(spine_leans))]

    elbow_smoothness = np.mean(elbow_deltas)
    spine_smoothness = np.mean(spine_deltas)

    plt.figure(figsize=(10, 5))
    plt.plot(elbow_angles, label="Elbow Angle")
    plt.plot(spine_leans, label="Spine Lean")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Temporal Smoothness")
    plt.legend()
    plt.savefig("output/smoothness_chart.png")
    smoothness_plot_path = "output/smoothness_chart.png"

    # Generate report
    grade = grade_skill(metrics_list[-1])
    report_path = "output/report.html"
    image_path={
        "plot_path": plot_path,
        "smoothness_plot_path": smoothness_plot_path,
        "impact_frame_path": "output/contact_frame.png"
    }
    generate_report(metrics_list[-1], grade, image_path, report_path)

    evaluation = final_evaluation(metrics_list[-1])
    save_evaluation(evaluation)

    #elbow_angles = [m["elbow_angle"] for m in metrics_list]
    #wrist_speeds = compute_joint_velocity(keypoints_over_time, joint_id=15)  # wrist



    return {
        "metrics": metrics_list[-1],
        "output_video": out_path,
        "plot_path": plot_path,
        "report_path": report_path,
        "smoothness_plot_path": smoothness_plot_path,
        "impact_frame_path": "output/contact_frame.png"
    }
