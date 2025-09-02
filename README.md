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
        



![liangziLake](PSDS_Eval/liangzi1.png)

![liangzi2](PSDS_Eval//liangzi2.png)