

将下面的语句翻译成英文， 使其成为 github 首页中的 ReadMe,  尽量保持其友好阅读的语气， 可以设置合理的排版与插图

# 1. introduction

这个仓库提供了一个端到端深度学习框架用于声音事件检测， 并且我们将其应用于呼吸音的声音事件检测，

我们借鉴了计算机视觉中先验框的概念， 将先验框的设计以及区间偏移量可学习的思想引入到其中。

通过在数据生成阶段使用desed_task/dataio/datasets_resp_v9_8_7.py： RespiraGnnSet(Dataset) 该类中的 generate_anchor_intervals 函数来生成先验区间，  并在网络模型中通过 desed_task/nnet/EzhouNet_v9_7_9.py： GraphRespiratory(nn.Module):  Interval_Refine 模块来学习先验区间的偏移量，

通过上述方式，我们实现了声音事件区间的可学习，  从而避免了使用帧级别的后处理方式来生成事件的区间，需要说明的是，尽管这种端到端的声音事件检测的方式被证明是有效的， 但在我们的呼吸音声音事件检测的场景中， 当前的效果却远没有达到临床使用的阶段， 这里只是提供了一个参考给后续的研究者，    更多的设计原理可以参考论文，同样 需要说明的是，这篇论文仅用于参考设计原理的理解，  在文章投稿之后， 我们更新许多模块的组件， 用于更好的训练优化。





# 2. getting  started



1.  首先 参考这个仓库（https://github.com/DCASE-REPO/DESED_task）中的步骤安装好 声音事件检测评价函数， 用于后续评估声音事件的检测指标；
2.  安装 python=3.8;  pytorch-1.13.1;     pytorch-lightning–2.2.5 ;    torch_geometric–2.5.2;   安装所有的依赖 pip  install -r  requirements.txt
3.   准备好你任务场景下声音事件检测的数据集；  在我们的呼吸音的场景中，  我们使用  SPRsound   以及   HF Lung V1  这两个数据集；



训练脚本 

1.    基于先验区间， 学习区间初始偏移量以及结束偏移量的方式，   可以通过设置 requires_grad = False or True   来控制基础偏移量是否是可学习的。 

   ```
           self.start_weight_params = nn.ParameterList([
               nn.Parameter(torch.linspace(-1.50, 1.50, dist_bins_list[i]), requires_grad= False)
               for i in range(self.num_scales)
           ])
           self.end_weight_params = nn.ParameterList([
               nn.Parameter(torch.linspace(-1.50, 1.50, dist_bins_list[i]), requires_grad= False)
               for i in range(self.num_scales)
           ])
   ```

  

```
python 
```

2.  类似于 yolo 中的思想，  学习区间中心点的偏移量， 以及区间宽度的偏移量；

```
            a_w = (ends - starts).clamp(min=1e-6)  #  width for the anchor interval, seconds
            a_c = 0.5 * (starts + ends)  #  center for the anchor interval

            pred_centers = a_c + t_c_pred * a_w
            pred_widths = a_w * torch.exp(t_w_pred.clamp(min=-6.0, max=6.0))

            s = (pred_centers - 0.5 * pred_widths).clamp(min=0.0, max=float(audio_len))
            e = (pred_centers + 0.5 * pred_widths).clamp(min=0.0, max=float(audio_len))
```





​          3.  将中心点偏移量的方式 与起始终止位置偏移量的方式结合起来；

​          python   train_respiratory_lab10_1_3.py



在呼吸音的声音事件的检测任务上， 当前方法可以获得的效果如下：





#   3.further  step

对于后续， 如果想基于这个工作进行改进的话， 个人建议可以从如下的三个方面进行改进：

1.   试着尝试去除对特征图进行循环切片的方式，  尽管分组特征提取是本工作的关键点之一，  但是这种在神经网络架构中应用循环切片的方式，  是极其不利于对该网络进行量化，部署到生产环境中。
2.  对于呼吸音特征的选择， 可以尝试按照这篇文章中的思路（Benchmarking of eight recurrent
   neural network variants for breath phase and adventitious sound detection on
   a self-developed open-access lung sound database—hf_lung_v1)    选择语谱图， MFCCs,   能量和进行统计。
3.  对于特征节点的更新， 可以尝试使用多尺度图卷积 multi-scale graph convolutional networks（https://github.com/xuyuankun631/IcicleGCN），（https://github.com/Eydcao/BSMS-GNN） 以及 Scalable Graph Learning（https://github.com/PKU-DAIR/SGL）  替换原来的GAT 模块来更新节点特征。





# Wanna say

本文的灵感是源于，在鄂州科研期间去参加了一场生物医学的会议，   看到了图神经网络架构被广泛应用于生物信号的处理，因此，我们也想尝试将其应用于呼吸音信号， 幸运的是这成功了，    这就是为什么我们将其命名为 EzhouNet,   ezhou  是这个城市的名字。



在鄂州进行科研的这段期间， 我认识了一位朋友kun， 他带我去观光了梁子湖，   kun  说道，  世人只知 武汉东湖， 却不知鄂州 梁子湖， 这里确实是一个生态良好的栖息地。





Happy  coding  and  good luck  with your  project.









```
Using confidence threshold: conf=0.501
Category-specific NMS IoU thresholds:
  Stridor: 0.5
  Wheeze: 0.4
  Crackle: 0.15
  Rhonchi: 0.3
	 call the compute event based metrics  

the Event based overall   f score: 0.19796610169491524, 	 error rate : 2.3084479371316307

the Event based class wise average f score: 0.1848416711564406,	 error rate : 2.966633604392851

  Class-wise metrics
  ======================================
    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
    Rhonchi      | 29      94    | 14.6%  9.6%   31.0%   | 3.62   0.69   2.93    |
    Stridor      | 5       18    | 17.4%  11.1%  40.0%   | 3.80   0.60   3.20    |
    Crackle      | 287     496   | 17.4%  13.7%  23.7%   | 2.25   0.76   1.49    |
    Wheeze       | 188     358   | 24.5%  18.7%  35.6%   | 2.19   0.64   1.55    |

Using confidence threshold: conf=0.65
Category-specific NMS IoU thresholds:
  Stridor: 0.5
  Wheeze: 0.4
  Crackle: 0.15
  Rhonchi: 0.3
	 call the compute event based metrics  

the Event based overall   f score: 0.20275862068965514, 	 error rate : 2.257367387033399

the Event based class wise average f score: 0.19081504850632036,	 error rate : 2.8438182069170024

  Class-wise metrics
  ======================================
    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
    Rhonchi      | 29      88    | 15.4%  10.2%  31.0%   | 3.41   0.69   2.72    |
    Stridor      | 5       17    | 18.2%  11.8%  40.0%   | 3.60   0.60   3.00    |
    Crackle      | 287     486   | 17.9%  14.2%  24.0%   | 2.21   0.76   1.45    |
    Wheeze       | 188     350   | 24.9%  19.1%  35.6%   | 2.15   0.64   1.51    |

Using confidence threshold: conf=0.8
Category-specific NMS IoU thresholds:
  Stridor: 0.5
  Wheeze: 0.4
  Crackle: 0.15
  Rhonchi: 0.3
	 call the compute event based metrics  

the Event based overall   f score: 0.20889202540578686, 	 error rate : 2.1886051080550097

the Event based class wise average f score: 0.20316344967739192,	 error rate : 2.6116479647528896

  Class-wise metrics
  ======================================
    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
    Rhonchi      | 29      82    | 16.2%  11.0%  31.0%   | 3.21   0.69   2.52    |
    Stridor      | 5       14    | 21.1%  14.3%  40.0%   | 3.00   0.60   2.40    |
    Crackle      | 287     479   | 18.3%  14.6%  24.4%   | 2.18   0.76   1.43    |
    Wheeze       | 188     333   | 25.7%  20.1%  35.6%   | 2.06   0.64   1.41    |
```





--------------

Here’s a polished **README.md** draft in English for your GitHub homepage. I kept it **friendly, clear, and structured**, with some Markdown formatting and light illustrative elements (like emoji for readability).

------



# Introduction 

🎉    Welcome to **EzhouNet** . This repository provides an **end-to-end deep learning framework** for **sound event detection (SED)**.   We focus on **respiratory sound events **, and the idea was inspired by **anchor boxes in computer vision**.  

Instead of using frame-level post-processing, we directly **learn event intervals** by:  
- Generating **anchor intervals** with  
  `desed_task/dataio/datasets_resp_v9_8_7.py → RespiraGnnSet(Dataset).generate_anchor_intervals`  
- Refining interval offsets with  
  `desed_task/nnet/EzhouNet_v9_7_9.py → GraphRespiratory(nn.Module).Interval_Refine`  

⚠  Please note: while this method has been shown effective for sound event detection,   in the **respiratory sound detection** scenario, it is **not yet ready for clinical use**.  
And this repo serves as a **reference implementation** for researchers.  The original design principles are detailed in our paper, though many modules have since been updated.

---



# 🚀 Getting Started

1. Install the **evaluation functions** following the steps in [DESED_task](https://github.com/DCASE-REPO/DESED_task).  
   These will be used to compute  sound event  detection metrics.
2. Set up your environment:  
   - `python=3.8`  
   - `pytorch=1.13.1`  
   - `pytorch-lightning=2.2.5`  
   - `torch_geometric=2.5.2`  
   - Install dependencies:  
     ```bash
     pip install -r requirements.txt
     ```
3. Prepare your dataset.  
   - For respiratory sounds, we used **SPRsound** and **HF Lung V1**.

---

## 🏋️ Training



cd  into  the this path :

```
/Respira_SED_LGNN/recipes/dcase2023_task4_baseline/
```



### 1. Learn start & end offsets of anchor intervals  

Set `requires_grad=True` or `False` to control whether bins  are learnable:
```python
self.start_weight_params = nn.ParameterList([
    nn.Parameter(torch.linspace(-1.50, 1.50, dist_bins_list[i]), requires_grad=False)
    for i in range(self.num_scales)
])
self.end_weight_params = nn.ParameterList([
    nn.Parameter(torch.linspace(-1.50, 1.50, dist_bins_list[i]), requires_grad=False)
    for i in range(self.num_scales)
])
```



```
python   train_respiratory_lab9_8_6.py
```



### 2. YOLO-style learning of center & width offsets

```python
a_w = (ends - starts).clamp(min=1e-6)  # anchor width, seconds
a_c = 0.5 * (starts + ends)            # anchor center

pred_centers = a_c + t_c_pred * a_w
pred_widths = a_w * torch.exp(t_w_pred.clamp(min=-6.0, max=6.0))

s = (pred_centers - 0.5 * pred_widths).clamp(min=0.0, max=float(audio_len))
e = (pred_centers + 0.5 * pred_widths).clamp(min=0.0, max=float(audio_len))
```

```
 python   train_respiratory_lab10_1_2.py
```



### 3. Combine both methods

Mixing center-offset and start/end-offset learning improves detection performance.

```
       python   train_respiratory_lab10_1_3.py
```



here  is  a  reference result:

```
Using confidence threshold: conf=0.501
Category-specific NMS IoU thresholds:
  Stridor: 0.5
  Wheeze: 0.4
  Crackle: 0.15
  Rhonchi: 0.3
	 call the compute event based metrics  

the Event based overall   f score: 0.19796610169491524, 	 error rate : 2.3084479371316307

the Event based class wise average f score: 0.1848416711564406,	 error rate : 2.966633604392851

  Class-wise metrics
  ======================================
    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
    Rhonchi      | 29      94    | 14.6%  9.6%   31.0%   | 3.62   0.69   2.93    |
    Stridor      | 5       18    | 17.4%  11.1%  40.0%   | 3.80   0.60   3.20    |
    Crackle      | 287     496   | 17.4%  13.7%  23.7%   | 2.25   0.76   1.49    |
    Wheeze       | 188     358   | 24.5%  18.7%  35.6%   | 2.19   0.64   1.55    |

Using confidence threshold: conf=0.65
Category-specific NMS IoU thresholds:
  Stridor: 0.5
  Wheeze: 0.4
  Crackle: 0.15
  Rhonchi: 0.3
	 call the compute event based metrics  

the Event based overall   f score: 0.20275862068965514, 	 error rate : 2.257367387033399

the Event based class wise average f score: 0.19081504850632036,	 error rate : 2.8438182069170024

  Class-wise metrics
  ======================================
    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
    Rhonchi      | 29      88    | 15.4%  10.2%  31.0%   | 3.41   0.69   2.72    |
    Stridor      | 5       17    | 18.2%  11.8%  40.0%   | 3.60   0.60   3.00    |
    Crackle      | 287     486   | 17.9%  14.2%  24.0%   | 2.21   0.76   1.45    |
    Wheeze       | 188     350   | 24.9%  19.1%  35.6%   | 2.15   0.64   1.51    |

Using confidence threshold: conf=0.8
Category-specific NMS IoU thresholds:
  Stridor: 0.5
  Wheeze: 0.4
  Crackle: 0.15
  Rhonchi: 0.3
	 call the compute event based metrics  

the Event based overall   f score: 0.20889202540578686, 	 error rate : 2.1886051080550097

the Event based class wise average f score: 0.20316344967739192,	 error rate : 2.6116479647528896

  Class-wise metrics
  ======================================
    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |
    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |
    Rhonchi      | 29      82    | 16.2%  11.0%  31.0%   | 3.21   0.69   2.52    |
    Stridor      | 5       14    | 21.1%  14.3%  40.0%   | 3.00   0.60   2.40    |
    Crackle      | 287     479   | 18.3%  14.6%  24.4%   | 2.18   0.76   1.43    |
    Wheeze       | 188     333   | 25.7%  20.1%  35.6%   | 2.06   0.64   1.41    |
```



------

# 🔮 Further Steps

If you’d like to improve upon this work, here are some suggestions:

1. **Avoid  group  cyclic slicing** of  spectrogram feature maps.
    While useful for grouped feature extraction, it makes quantization & deployment difficult.
2. **Try alternative respiratory features**:
    spectrograms, MFCCs, energy, or statistical features (see the paper  *Benchmarking of eight RNN variants for breath phase and adventitious sound detection on hf_lung_v1*).
3. **Explore advanced  multi scale graph convolution modules**    for node updates , e.g.:
   - [IcicleGCN](https://github.com/xuyuankun631/IcicleGCN)
   - [BSMS-GNN](https://github.com/Eydcao/BSMS-GNN)
   - [Scalable Graph Learning](https://github.com/PKU-DAIR/SGL)

------

# 💡 Inspiration

The idea came  a biomedical conference in  the University,   where I saw **graph neural networks**  being widely applied to biosignals.   That’s how **EzhouNet** was born — named after the city of **Ezhou**.

During the  research time in Ezhou, I met a friend there, Kun, who took me to visit Liangzi Lake. He said:

> “People knows Wuhan’s East Lake, but few know Liangzi Lake in Ezhou.”
>  It truly is an ecological gem. 🌿🌊

------



   Feel free to fork, experiment, and improve your  lab.   Happy coding, and good luck with your projects! 🚀
        



![liangzi1](/home/respecting_god/Downloads/liangzi1.png)

![liangzi2](/home/respecting_god/Downloads/liangzi2.png)