# NeRF
This is a pytorch implementation of NeRF models in the following list.
1. [NeRF-Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
2. [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://arxiv.org/abs/2103.13415)


## Setup
1. Setup conda or venv environment with python>=3.11.
    ```
    conda create -n nerf python=3.11
    conda activate nerf
    ```
2. Install dependencies with
    ```
    pip install -r requirements.txt
    ```


## Demo with Blender Dataset
1. Obtain blender dataset
- Download a scene from [this link](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi).
- Place the zip file in **data/** and unzip it with
    ```
    unzip data/{file_name}.zip -d data/
    ```
2. Choose the *nerf_type* in **configs/nerf.yaml**, either *vanila* or *mip*.
3. Start training with
    ```
    python main.py -t
    ```
4. Generate gif with
    ```
    python main.py -r
    ```
5. You can check the results in **log/**.