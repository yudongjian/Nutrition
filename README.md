

# 🎉 This work was accepted in ACM MM 2025!

---

## 🧾 Paper Information

<p align="center">

# Spatial-Aware Multi-Modal Information Fusion for Food Nutrition Estimation

</p>

<p align="center" style="font-size:small;">

Dongjian Yu¹, Weiqing Min², Xin Jin¹, Qian Jiang¹, Shuqiang Jiang²

</p>

<p align="center" style="font-size:small;">

¹ Yunnan University

</p>

<p align="center" style="font-size:small;">

² Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences

</p>

### Please feel free to contact me at yudongjian@stu.ynu.edu.cn if you have any questions.

## 📄 Paper Link

[[📄 Paper Link]](https://doi.org/10.1145/3746027.3755750)  

## 🚧 Code Release Notice
We recommend using the 2D-only version, as it runs significantly faster while still achieving competitive performance.
The 3D-enhanced version provides additional spatial cues but incurs higher computational cost.
```bash
# Train the model without 3D information (2D-only, recommended)
python train2D-mm.py --b 8 --log ./logs/log1  

or
# Train the model with 3D information
python train2D-mm.py --b 8 --log ./logs/log1
```


## 📚 Reference


```bash
@inproceedings{10.1145/3746027.3755750,
    author = {Yu, Dongjian and Min, Weiqing and Jin, Xin and Jiang, Qian and Jiang, Shuqiang},
    title = {Spatial-Aware Multi-Modal Information Fusion for Food Nutrition Estimation},
    year = {2025},
    isbn = {9798400720352},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3746027.3755750},
    doi = {10.1145/3746027.3755750},
    booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
    pages = {8863–8871},
    numpages = {9},
    keywords = {deep learning, food computing, food nutrients estimation, multi-modal fusion},
    location = {Dublin, Ireland},
    series = {MM '25}
}
```

