# SAM 2 Experiments

## Initial Setup
- Chose the **SAM 2** model for initial testing.
- Conducted tests using **Nvidia RTX 2060 GPU**.
- Faced installation issues on Windows, leading to a decision to install on **WSL**.
- Explored **WSL, Python, and environment configurations**.
- Installed **SAM 2** using **Conda**, creating a dedicated environment with necessary dependencies.

## Image Segmentation Tests
- Tested a **SAM 2 image predictor model checkpoint** with suggested images to understand the process.
- Applied the same model checkpoint to **multiphase flow images** (`geom1_qc2_0001`).
- Performed an **interactive segmentation refinement process**:
  - Selected **10 points** and labeled them as part of the segmentation or background.
  - Utilized **LabelMe** ([GitHub](https://github.com/wkentaro/labelme)) for annotation.
  - Encountered installation issues on Windows, requiring specific **WSL configurations**.
  - Created a separate **Conda environment** for LabelMe.
  - Conducted **three iterations** of the refinement process, selecting from three generated masks per iteration.
  - Observed that **higher-scoring masks** captured general features (entire pipe and bubbles), whereas **lower-scoring masks** segmented specific bubbles but missed some.

## Video Segmentation Tests
- Tested the **SAM 2 video predictor** with the same model checkpoint using a suggested video.
- Initial propagation was slow:
  - **First run**: 4 hours to process half of a 6-second video (~200 frames).
  - **After reset**: Full 200-frame video processed in **4 minutes**.
- Selected a **16-second video** (`qc2_qd1_10x_formacao.avi`, 600MB) for multiphase flow image testing.
- Converted video into frames using **FFmpeg**:
  - Installed FFmpeg inside the **dedicated Python environment**.
  - Used command line to extract **493 frames** into a directory.

## Performance Observations
- **First-time runs** of video segmentation were extremely slow:
  - 500-frame video: **10 hours** (~60-70s per frame).
  - Subsequent runs of the same model: **<1 hour** (~3-4s per frame).
  - A smaller **6-second video** showed a similar trend:
    - First run: **~20s per frame**.
    - Subsequent runs: **~1s per frame**.
- Explanation for **subsequent speed improvements**:
  - **First-time execution** triggers PyTorch and CUDA to **compile and optimize GPU kernels**.
  - **Compilation caching** significantly speeds up future runs.

## Results and Insights
- **Video segmentation provided better results** compared to single image segmentation.
- **Bubbles were accurately tracked** across frames in the video segmentation process, but these results can be misleading because the video used and the frame images had better identifiable bubbles, with the bubbles being bigger and contrasting well with the background.
- Performance improvements observed with **subsequent runs** due to **model caching** and **GPU optimization**.

---
This document presents the **setup, challenges, and insights** gained from testing SAM 2 on multiphase experimental images and videos. Further optimization and model tuning are planned for improved segmentation accuracy and efficiency.
