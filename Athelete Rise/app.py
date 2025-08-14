import streamlit as st
import tempfile
from analyze import analyze_video

st.title("🏏 Cricket Shot Analyzer")

uploaded_file = st.file_uploader("Upload the video", type=["mp4", "mov"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.info("Processing video... This may take a few seconds.")
    result = analyze_video(video_path)

    with open(result["output_video"], "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)
    st.subheader("📊 Biomechanical Metrics")
    st.json(result["metrics"])
    st.image(result["plot_path"])
    st.image(result["smoothness_plot_path"])
    st.image(result["impact_frame_path"])
    

    with open(result["output_video"], "rb") as f:
        st.download_button("Download Annotated Video", f, file_name="annotated_video.mp4")

    with open(result["report_path"], "rb") as f:
        st.download_button("Download Report", f, file_name="report.html")
