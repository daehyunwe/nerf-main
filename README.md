# NeRF
This is a pytorch implementation of NeRF models in the following list.
1. [NeRF-Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
2. [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://arxiv.org/abs/2103.13415)

## Demo with Lego Dataset
1. Setup conda or venv environment with python>=3.11.
    ```
    conda create -n nerf python=3.11
    conda activate nerf
    ```
2. Install dependencies with
    ```
    pip install -r requirements.txt
    ```
3. Obtain lego dataset
- Download the dataset from [this link](https://drive.google.com/file/d/1EitqzKZLptJop82hdNqu1YCogxgNgN5u/view?usp=share_link).
- Place the zip file in **data/** and unzip it with
    ```
    unzip data/lego.zip -d data/
    ```
4. Choose the *nerf_type* in **configs/nerf.yaml**, either *vanila* or *mip*.
5. Start training with
    ```
    python main.py -t
    ```
6. Generate gif with
    ```
    python main.py -r
    ```
7. You can check the results in **log/**.