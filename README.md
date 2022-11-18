# README

FunGeneType is a unified functional annotation framework of proteins. It now can annotate proteins of antibiotic resistance genes (ARGs)  and virulence factors(VFs) at class/group level.  Due to the easy-to-use Adapter architecture, FunGeneTyper is expected to be gradually built into an open community. 



## Requirements

FunGeneType communicates with  the following separate libraries and packages:

- [PyTorch](https://github.com/pytorch/pytorch)   (test on versions `1.8.0`)

- [faiss](https://github.com/facebookresearch/faiss)     (test on versions `1.7.1`)

- [tqdm](https://github.com/tqdm/tqdm)      

- [Biopython](https://biopython.org/)  



## First-time setup

**`PyTorch`** 

We strongly recommend using the GPU version of PyTorch if you have a GPU. If not, you can install the CPU version of PyTorch by using the following command. For GPU version installation, see [PyTorch official website](https://github.com/pytorch/pytorch)

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



## Usage

####:house_with_garden:  **Description**

FunGeneType is built based on a protein pre-training language model. Due to the hundreds of millions of parameters in the pre-training model, fine-tuning all parameters for every task is redundant and inefficient. We propose adapter modules inserted into each Transformer block as a few trainable parameters. The Adapter architecture method allows each research group to only save a large-scale pre-trained model parameter locally and only needs to download or upload a small number of Adapter parameters separately for each functional annotation task. The Adapter architecture is not only conducive to model training but also more convenient for others in the community to use, thereby building a more powerful protein function annotation community.



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
  -  Use the GPU to run the program. This value determines which GPU is used to run the program.

- `--adapter`:
  -  Adapter used, default including `ARGs` and `VFs`.

- `--group`:
  -  Perform group-level functional classification on protein sequences. If this parameter is not specified, group-level classification will not be performed.



####:houses: Classification & Train

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
python -m torch.distributed.launch --nproc_per_node=2 classifier_Multi_GPUs.py \
--input example/Resistance_gene/test.fasta \
--output example/Resistance_gene/output_class \
--adapter ARGs \
--group 
-------------------------------------------
python -m torch.distributed.launch --nproc_per_node=2 classifier_Multi_GPUs.py \
--input example/Resistance_gene/test.fasta \
--output example/Resistance_gene/output_class \
--adapter ARGs \
--group 
```



---



**Train**

You can also train a class-level classification or group-level classification model based on your own data, refer to the examples we gave. Taking the training of resistance genes (ARGs) as an example, you can train the class classification by running the following code:

```python
python Train_class.py --Train_category 20
```

`--Train_category`: Number of categories to classify ##Todo, modify 



Similarly, you can train the Group classification by running the following code:

```python
python Train_group.py
```

