# DSensD+: Decision-level Sensor Dropout in combination with Mutual Distillation for classification tasks

> Public repository of our work "*Multi-sensor Model for Earth Observation Robust to Missing Data via Sensor Dropout and Mutual Distillation*"

![dsensdp](imgs/dsensdp_model.png)
In the previous image is **DSensD+**, while in our research work we also introduce **DSensD**, a simplified version only with Sensor Dropout at the decision-level. We focus and validate in classification tasks in the Earth observation domain.

> [!NOTE]  
> Read about the used data in [data folder](./data)

### Training

* To train the our novel **DSensD+** (decision-level sensor dropout with mutual distillation) run
```
python train_multi.py -s config/dsensdp_ex.yaml
```

* To train the our **DSensD** (decision-level sensor dropout) run
```
python train_multi.py -s config/dsensd_ex.yaml
```

* To train the baseline **FSensD** (feature-level sensor dropout) run
```
python train_multi.py -s config/fsensd_ex.yaml
```

* To train the baseline **ISensD** (input-level sensor dropout) run
```
python train_single.py -s config/isensd_ex.yaml
```

> [!NOTE]  
> Other competitors were used from their original code and also following our previous work at [CoM-views](https://github.com/fmenat/CoM-views). 


### Evaluation
![missing views](imgs/missing_views.jpg)

* To evaluate the model by the prediction performance:
```
python evaluate.py -s config/eval_ex.yaml
```


## Installation
Please install the required packages with the following command:
```
pip install -r requirements.txt
```

# :scroll: Source

* :unlock: [Published version](x)

# 🖊️ Citation

Mena, Francisco, et al. "*Multi-sensor Model for Earth Observation Robust to Missing Data via Sensor Dropout and Mutual Distillation*." Accepted at IEEE Access, 2025.

