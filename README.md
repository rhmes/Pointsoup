# Pointsoup: High-Performance and Extremely Low-Decoding-Latency Learned Geometry Codec for Large-Scale Point Cloud Scenes

## Jupyter Notebook Workflow & Device-Aware Environment Setup

The recommended workflow for Pointsoup is documented in the Jupyter notebook `Pointsoup_notebook.ipynb`. This notebook guides you through:
- Cloning/pulling the repository
- Setting up the conda environment (with device auto-detection)
- Downloading and preparing the ModelNet40 dataset
- Training the model on ModelNet40 (sample code included)
- Evaluating and visualizing results (PSNR, bitrate)
- Comparing results between the vanilla Pointsoup and the updated Pointsoup with attention

### Environment Setup: Device Auto-Detection

The environment setup script (`environment/env_create.sh`) automatically detects your device (CUDA, MPS, or CPU) and selects the appropriate environment file:
- CUDA: Uses `environment.yml`
- MPS/CPU: Uses `environment_cpu.yml`

To create the environment, simply run:
```bash
source ./environment/env_create.sh
```

### Model Paths
- Training checkpoints: `model/session/`
- Pretrained models: `model/exp/` (vanilla), `model/exp-sa/` (with attention)

### Dataset Preparation
- Download ModelNet40 (8192 points) from the link in the notebook
- Extract to `data/ModelNet40_pc_01_8192p/`
- Training and test `.ply` files should be available for notebook cells

### Project Structure
```
Pointsoup/
├── compress.py
├── decompress.py
├── evaluate.py
├── eval_PSNR.py
├── train.py
├── Pointsoup_notebook.ipynb   # Main workflow notebook
├── csv/                    # Evaluation results
├── data/
│   ├── example_pc_1023/    # Example point clouds
│   ├── compressed/         # Compressed files
│   ├── decompressed/       # Decompressed files
├── environment/
│   ├── env_create.sh       # Device-aware environment setup
│   ├── environment.yml     # CUDA environment
│   ├── environment_cpu.yml # MPS/CPU environment
├── kit/
│   ├── io.py
│   ├── op.py
│   ├── utils.py
├── model/
│   ├── session/            # Training checkpoints
│   ├── exp/                # Pretrained vanilla model
│   ├── exp-sa/             # Pretrained attention model
└── README.md
```

### Notes
- Environment setup is device-aware and automated
- See `Pointsoup_notebook.ipynb` for the latest recommended workflow
- Ensure all required dependencies are installed (see environment.yml).

# Pointsoup Project Updates & Contributions

## Contributions
- **Encoder Update:** The Pointsoup encoder has been updated to include a simple attention layer, improving feature aggregation and overall compression performance. This change enhances the model's ability to capture global context and improves reconstruction quality.
- **New Scripts:**
    - `Pointsoup_notebook.ipynb`: Jupyter notebook providing a comprehensive workflow for environment setup, data preparation, model training, evaluation, and visualization.
    - `evaluate.py`: End-to-end evaluation script for compressing, decompressing, and evaluating point clouds. Calculates metrics such as PSNR and Chamfer Distance, and saves results to CSV.
    - `eval_PSNR.py`: Standalone script for evaluating geometry PSNR, Chamfer Distance, and bits-per-point (bpp) from compressed/decompressed point clouds. Results are saved in the `csv/` folder.
- **Documentation:** Updated README with detailed usage instructions, command examples, and explanations of new features.
## Effect of Updated Encoder
- The addition of the simple attention layer in the encoder improves the aggregation of local and global features, resulting in:
    - Higher PSNR and lower Chamfer Distance in reconstructions.
    - More robust compression, especially for complex point cloud scenes.
    - Smoother training curves and better generalization.

## Usage & Commands

### End-to-End Evaluation
```bash
python evaluate.py --input_glob '<input_glob>' --compressed_path '<compressed_path>' --decompressed_path '<decompressed_path>' --model_load_path '<model_load_path>' --model_type '<pointsoup|pointsoup_sa>'
```
Example:
```bash
python evaluate.py --input_glob '../point-cloud-compression/data/ModelNet40_pc_01_8192p/**/test/*.ply' --compressed_path './data/compressed-sa/' --decompressed_path './data/decompressed-sa/' --model_load_path './model/exp-raghda-edit/ckpt-36600.pt' --model_type 'pointsoup_sa'
```

### Geometry PSNR & Chamfer Evaluation
```bash
python eval_PSNR.py --input_glob '<input_glob>' --compressed_path '<compressed_path>' --decompressed_path '<decompressed_path>' --resolution <peak_signal> --csv_dir './csv/'
```
Results are saved in the `csv/` folder as `<decompressed_folder>_psnr_results.csv`.

## Automated Pipeline: `evaluate.py`

The [`evaluate.py`](./evaluate.py) script automates the full pipeline for point cloud compression, decompression, and evaluation. It sequentially runs three main processes:

1. **Compression** ([compress.py](./compress.py)): Compresses point clouds using the trained model and saves the results.
2. **Decompression** ([decompress.py](./decompress.py)): Decompresses the compressed files to reconstruct the point clouds.
3. **Evaluation** ([eval_PSNR.py](./eval_PSNR.py)): Computes metrics such as D1 PSNR, Chamfer Distance, and bits-per-point (bpp) for the reconstructed point clouds.

Each process is executed via a subprocess call, and the script prints progress for each stage. This ensures reproducibility and makes it easy to benchmark model performance with a single command.

## Important Paths
- `data/`: Contains input point clouds, compressed, and decompressed files.
- `model/exp/` & `model/exp-sa/`: Store model checkpoints for loading in evaluation and training.
- `csv/`: Stores CSV files generated by evaluation scripts.
- `environment/`: Contains environment setup scripts and configuration files.


**The following technical appendix and documentation is based on the original Pointsoup repository.**
# Pointsoup: High-Performance and Extremely Low-Decoding-Latency Learned Geometry Codec for Large-Scale Point Cloud Scenes

## News
> Despite considerable progress being achieved in point cloud geometry compression, there still remains a challenge in effectively compressing large-scale scenes with sparse surfaces. Another key challenge lies in reducing decoding latency, a crucial requirement in real-world application. In this paper, we propose Pointsoup, an efficient learning-based geometry codec that attains high-performance and extremely low-decoding-latency simultaneously. Inspired by conventional Trisoup codec, a point model-based strategy is devised to characterize local surfaces. Specifically, skin features are embedded from local windows via an attention-based encoder, and dilated windows are introduced as cross-scale priors to infer the distribution of quantized features in parallel. During decoding, features undergo fast refinement, followed by a folding-based point generator that reconstructs point coordinates with fairly fast speed. Experiments show that Pointsoup achieves state-of-the-art performance on multiple benchmarks with significantly lower decoding complexity, i.e., up to 90$\sim$160$\times$ faster than the G-PCCv23 Trisoup decoder on a comparatively low-end platform (e.g., one RTX 2080Ti). Furthermore, it offers variable-rate control with a single neural model (2.9MB), which is attractive for industrial practitioners.

## Environment
The environment we use is as follows：
- Python 3.10.14
- Pytorch 2.0.1 with CUDA 11.7
- Pytorch3d 0.7.5
- Torchac 0.9.3

For reproducibility, use the device-aware environment setup script:

```bash
source ./environment/env_create.sh
```

## Data

In our paper, point clouds with the coordinate range of [0, 1023] are used as input.

Example point clouds are saved in ``./data/example_pc_1023/``, trained model is saved in ``./model/exp/``.

## Compression
First and foremost, the `tmc3` is need to perform predtree coding on bone points. If the `tmc3` file we provided cannot work on your platform, please refer to [MPEGGroup/mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for manual building.

```
chmod +x ./tmc3
```

You can adjust the compression ratio by simply adjusting the parameter `local_window_size`. In our paper, we use `local_window_size` in the range of 2048~128.

```
python ./compress.py \
    --input_glob='./data/example_pc_1023/*.ply' \
    --compressed_path='./data/compressed/' \
    --model_load_path='./model/exp/ckpt.pt'\
    --local_window_size=200 \
    --tmc_path='./tmc3'\
    --verbose=True
```

## Decompression

```
python ./decompress.py \
    --compressed_path='./data/compressed/' \
    --decompressed_path='./data/decompressed/' \
    --model_load_path='./model/exp/ckpt.pt'\
    --tmc_path='./tmc3'\
    --verbose=True
```

## Evaluation

```
python ./eval_PSNR.py \
    --input_glob='./data/example_pc_1023/*.ply' \
    --decompressed_path='./data/decompressed/' \
    --pcc_metric_path='./PccAppMetrics' \
    --resolution=1023
```

## Disucssion
Merits:

- High Performance - SOTA efficiency on multiple large-scale benchmarks.
- Low Decoding Latency - 90~160× faster than the conventional Trisoup decoder.
- Robust Generalizability - Applicable to large-scale samples once trained on small objects.
- High Flexibility - Variable-rate control with a single neural model.
- Light Weight - Fairly small with 761k parameters (about 2.9MB).

Limitations:

- Rate-distortion performance is inferior to G-PCC Octree codec at high bitrates (e.g., bpp>1). The surface approximation-based approaches (Pointsoup and Trisoup) seem hard to characterize accurate point positions even if given enough bitrate budget.

- Naive outdoor LiDAR frame coding efficacy is unsatisfactory. Due to the used sampling&grouping strategy, the pointsoup is limited to point clouds with relatively uniform distributed points, such as [S3DIS](http://buildingparser.stanford.edu/dataset.html), [ScanNet](https://github.com/ScanNet/ScanNet), [dense point cloud map](https://github.com/PRBonn/deep-point-map-compression), [8iVFB (human body)](https://plenodb.jpeg.org/pc/8ilabs), [Visionair (objects)](https://github.com/yulequan/PU-Net), etc.


## Citation

If you find this work useful, please consider citing our work:

```
@inproceedings{ijcai2024p595,
  title     = {Pointsoup: High-Performance and Extremely Low-Decoding-Latency Learned Geometry Codec for Large-Scale Point Cloud Scenes},
  author    = {You, Kang and Liu, Kai and Yu, Li and Gao, Pan and Ding, Dandan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {5380--5388},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/595},
  url       = {https://doi.org/10.24963/ijcai.2024/595},
}
```
