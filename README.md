# x-LightFM
An official implement of "xLightFM: Extremely Memory-Efficient Factorization Machine"

### 整理

#### code for dataset
1. dataset/avazu.py : avazu dataset

2. dataset/criteo.py : criteo dataset

3. dataset/movielens.py : movielens dataset

#### code for model
    
1. layer.py : several basic network 

2. model/xxx.py : specific model xxx

#### code for training

1. train.py 


### Command

1. pre-train model 

    ```shell
    python train.py --model fm
    ```

2. Lightfm(it's important to train the model without "--pre_quat" for the first time):

    ```shell
    python train.py --model qfm --pre_train --pre_quat
    ```

3. xLightfm:
    
    ```shell
    python train.py --model qfm_gs --memory_limit 27 --retrain --pre_train --pre_quat --K 2048
    ```