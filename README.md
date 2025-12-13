# AnimeL2M

AnimeL2M is a deep learning model that aims to detect AI-generated forgeries in animated images. It leverages the domain knowledge of animation styles, as well as fingerprints left by different generative models. The original github repository can be found [here](https://github.com/FlyTweety/AnimeDL2M).


# Requirements

To replicate results of this project, you need to first download the dataset on [google drive](https://drive.google.com/drive/folders/1f2wZ1naVYU9jf-RxKh3c3PUML30u5tEQ). I utilized `rclone` to download them. For the sake of time and resources limit, I only used the very first subset `0000` from the original dataset. Then, we randomly dropped the data to create a smaller dataset of 6,742 real images, 6,162 fake images from Danbooru, and 2,247 fake images from Civitai (for test set only).

Note that the `preprocess.py` is designed specifically to parse the structures of the downloaded dataset. Please modify accordingly if any changes on the dataset structure on the filesystem. Since it takes forever to wait for the queue on Yale `Bouchet`cluster, I moved all my dataset and training outputs to `Milgram` cluster. But you can also find the dataset in my scratch space on Bouchet under `/nfs/roberts/scratch/cpsc4710/cpsc4710_yz2483/animatedl2m_data/`.

Once you have the dataset ready, please install the environments using `env.ym`.

