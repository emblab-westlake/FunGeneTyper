# ![image-20231031181844548](example/log.png)

**FunGeneTyper** is a unified functional classification framework of protein coding genes. This repository contains codes and data for classifying protein coding genes of antibiotic resistance genes (ARGs)  and virulence factor genes (VFGs) at Type/Subtype level.  You can find more details about **FunGenTyper** in our papers (["**Zhang, Ju et al. 2024. BIB**"](https://academic.oup.com/bib/article/25/4/bbae319/7713721?login=true); ["**Zhang, Ju et al. 2022. bioxiv**"](https://www.biorxiv.org/content/10.1101/2022.12.28.522150v2)). 

FunGeneTyper can be extended to other classes of functional genes. Our expectation is that the straightforward adapter architecture will foster the gradual growth of an inclusive community focused on functional gene classification, leveraging the capabilities of FunGeneTyper.



## Requirements

FunGeneTyper communicates with  the following separate libraries and packages:

- [PyTorch](https://github.com/pytorch/pytorch)   (test on version `1.8.0`)
- [faiss](https://github.com/facebookresearch/faiss)     (test on version `1.7.1`)
- [tqdm](https://github.com/tqdm/tqdm)      
- [Biopython](https://biopython.org/)  



## First-time setup	
### Dependence
**`PyTorch`** 

We strongly recommend using the GPU version of PyTorch if you have a GPU. If not, you can install the CPU version of PyTorch by using the following command. For GPU version installation, Please refer to  [PyTorch official website](https://github.com/pytorch/pytorch)

```python
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**`faiss`**

```python
conda install faiss-cpu -c pytorch 
```

**`tqdm`**

```python
pip install tqdm
```

**`Biopython`**

```python
pip install biopython
```

notes: If you encounter problems with MKL missing libraries, install the specified version of MKL with `conda install mkl==2021.4.0`.


### Download FunGeneTyper and initialization Settings
You can download the code to the specified directory in the following way.
```
git clone https://github.com/emblab-westlake/FunGeneTyper.git
```
Before starting the first analysis, please run the `download_pretrain_weights.sh` to  download pretrain weight parameters for FunGeneTyper.




## Usage

#### :heavy_exclamation_mark: **Important Note**

FunGeneTyper is a framework for deep learning models. Although a CPU version of the model is available, we highly recommend using a GPU-accelerated server to enhance efficiency and maximize performance.



#### :house_with_garden:  **Description**

FunGeneType is built based on a protein pre-training language model. Due to the hundreds of millions of parameters in the pre-training model, fine-tuning all parameters for every task is redundant and inefficient. We propose adapter modules inserted into each Transformer block as a few trainable parameters. The Adapter architecture method allows each research group to only save one large-scale pre-trained model parameter locally, and only a small number of Adapter parameters need to be downloaded or uploaded separately for each functional classification task.The Adapter architecture not only facilitates model training, but also makes it easier for others in the community to use it, thereby building a more powerful protein function annotation community.



**Parameters**

python classifier.py -h

- `--input` : 
  - The input protein sequence file is required to be in fasta format.

- `--output`: 
  - This is the prefix of output result file.

- `--batch_size`:
  -  The number of sequences to be predicted once iteratively, and this value can be set larger, the default setting is 10.

- `--nogpu`:
  -  Use the CPU to run programs.

- `--gpu`:
  -  Run the program using the GPU. This value determines which GPU is used to run the program.

- `--adapter`:
  -  Adapter, default including `ARGs` and `VFs`.

- `--topK`:
  -  Top K representative subtype. Default 1.
  
- `--group`:
  -  Perform subtype-level functional classification on protein sequences. If this parameter is not specified, subtype-level classification will not be performed.





#### :houses: Classification & Train

In a single GPU or CPU environment, run `classifier.py`. In addition, to speed up functional protein annotation, an easy-to-use program that uses multiple GPUs simultaneously is also provided as `classifier_Multi_GPUs.py`

```python
# Take an example of the class and group classification of resistance genes
# Using CPU
python classifier.py \
--input example/Resistance_gene/test.fasta \
--output example/Resistance_gene/output_class \
--adapter ARGs \
--group \
--nogpu

# Using GPU 
python classifier.py \
--input example/Resistance_gene/test.fasta \
--output example/Resistance_gene/output_class\
--adapter ARGs \
--group \
--gpu 0

# Using Multi-GPUs
CUDA_VISIBLE_DEVICES="0,1" 
python -m torch.distributed.launch --nproc_per_node=2 classifier-Multi-GPUs.py \
--input example/Resistance_gene/test.fasta \
--output example/Resistance_gene/output_class \
--adapter ARGs \
--group 
-------------------------------------------
python -m torch.distributed.launch --nproc_per_node=2 classifier-Multi-GPUs.py \
--input example/Resistance_gene/test.fasta \
--output example/Resistance_gene/output_class \
--adapter ARGs \
--group 
```



---



**Train**

You can also train a class-level classification or group-level classification model based on your own data, refer to the related notebooks under the **'Tutorials/'** for detailed dataset construction process.  **You can generate all datasets related to antibiotic resistance genes (ARGs) directly using these notebooks and reproduce the results presented in our paper. Additionally, you can also download these datasets through the provided links and store them in the specified local directory.**



Taking the training of resistance genes (ARGs) as an example, you can train the class classification by running the following code.  Please download the datasets through [the provided link](https://drive.google.com/drive/folders/1uKP9-IIkOXqgQYSSfruycdCyl0otY41J?usp=drive_link) and place them in the 'example/ARGs_class_TrainingData/' folder. 

```python
python Train_class.py --Train_category 20
```

`--Train_category`: Number of categories to classify 



Similarly, you can train the Group classification by running the following code. Please download the datasets through [the provided link](https://drive.google.com/drive/folders/1QZHu0lY1-l_qdL9xu7BVZMtaEzVnydwO?usp=drive_link) and place them in the 'example/ARGs_group_TrainingData/' folder. 

```python
python Train_group.py
```



## Citation

```
@article{zhang2022ultra,
  title={Ultra-Accurate Classification and Discovery of Functional Protein-Coding Genes from Microbiomes Using FunGeneTyper: An Expandable Deep Learning-Based Framework},
  author={Zhang, Guoqing and Wang, Hui and Zhang, Zhiguo and Zhang, Lu and Guo, Guibing and Yang, Jian and Yuan, Fajie and Ju, Feng},
  journal={bioRxiv},
  pages={2022--12},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

If you have any problems with the framework, please raise the issue or contact zhangguoqing84@westlake.edu.cn,we will actively update and maintain.
