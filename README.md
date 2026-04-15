# GraphDREAM

## Requirement
Checking and installing environmental requirements
```python
pip install -r requirements.txt
```
## Datasets
链接: https://pan.baidu.com/s/1-H8tIWtQS_ImhhJo6eocMQ?pwd=yg7r 提取码: yg7r 

Adding the dataset path to the corresponding location in the run.py file, e.g. IEMOCAP_path = "".

## Run
### IEMOCAP-6
```bash
bash script/run_iemocap.sh
```

### MELD-7
```bash
bash script/run_meld.sh
```

### CH-SIMSv2
```bash
bash script/run_chsims.sh 2   
bash script/run_chsims.sh 3
bash script/run_chsims.sh 5
```

### CMUMOSEI
```bash
bash script/run_mosei.sh has0
bash script/run_mosei.sh non0
```


## Acknowledgements

Special thanks to the following open-source projects for their foundational work, which greatly inspired the development of **GraphDREAM**:

* [MMSA](https://github.com/thuiar/MMSA?tab=readme-ov-file) - A Multimodal Sentiment Analysis framework.
* [GraphSmile](https://github.com/lijfrank/GraphSmile) - Our internal baseline and experimental framework.
* [Himalgg](https://github.com/qianfegkui/himallgg) - For its innovative approach to graph-based emotion recognition.

We appreciate the contributors of these projects for their dedication to the research community.


