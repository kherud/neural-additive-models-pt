# [Neural Additive Models (PyTorch)](https://github.com/google-research/google-research/tree/master/neural_additive_models)

This is a PyTorch re-implementation for neural additive models, check out:

- [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912).
- [TensorFlow OG Implementation](https://github.com/google-research/google-research/tree/master/neural_additive_models)

<img src="https://i.imgur.com/Hvb7sb2.jpg" width="50%" alt="Neural Additive Model" >

## Install Package

```
pip install nam-pt
```

## Dependencies

- torch==1.7.0
- fsspec==0.8.4
- pandas==1.1.4
- tqdm==4.54.0
- sklearn==0.0
- absl-py==0.11.0
- gcsfs==0.7.1

## Usage

```
conda env create -f environment.yml
conda activate nam-pt
python run.py
```

In Python:

``` python
from nam import NeuralAdditiveModel

model = NeuralAdditiveModel(input_size=x_train.shape[-1],
                            shallow_units=100,
                            hidden_units=(64, 32, 32),
                            shallow_layer=ExULayer,
                            hidden_layer=ReLULayer,
                            hidden_dropout=0.1,
                            feature_dropout=0.1)
logits, feature_nn_outputs = model.forward(x)
```

Citing
------
If you use this code in your research, please cite the following paper:

> Agarwal, R., Frosst, N., Zhang, X., Caruana, R., & Hinton, G. E. (2020).
> Neural additive models: Interpretable machine learning with neural nets.
> arXiv preprint arXiv:2004.13912


      @article{agarwal2020neural,
        title={Neural additive models: Interpretable machine learning with neural nets},
        author={Agarwal, Rishabh and Frosst, Nicholas and Zhang, Xuezhou and
        Caruana, Rich and Hinton, Geoffrey E},
        journal={arXiv preprint arXiv:2004.13912},
        year={2020}
      }

---

*Disclaimer about COMPAS dataset: It is important to note that
developing a machine learning model to predict pre-trial detention has a
number of important ethical considerations. You can learn more about these
issues in the Partnership on AI
[Report on Algorithmic Risk Assessment Tools in the U.S. Criminal Justice System](https://www.partnershiponai.org/report-on-machine-learning-in-risk-assessment-tools-in-the-u-s-criminal-justice-system/).
The Partnership on AI is a multi-stakeholder organization -- of which Google
is a member -- that creates guidelines around AI.*

*We’re using the COMPAS dataset only as an example of how to identify and
remediate fairness concerns in data. This dataset is canonical in the
algorithmic fairness literature.*

*Disclaimer: This is not an official Google product.*
