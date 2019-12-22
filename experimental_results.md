# Experimental Results
## Notations
Report the only Accuracy for each test set
- **val** is the accuracy on the original test set
- **aval** is the accuracy on the **adversarial** test set

## CIFAR10

### Experimental Setup
- Attack : PGD-Linf
- Epsilon : 16.0 / 255.0
- Others :

    |index|eps_iter|num_steps|restarts|
    |:--|--:|--:|--:|
    |p1|2.0|10|1|
    |p2|4.0|400|1|
    |p3|4.0|400|100|

### Original Paper

|method|val|aval - p1|aval - p2|aval-p3|
|:--|--:|--:|--:|--:|
|Plain|83.0%|0.0%|0.0%|0.0%|
|CLP|73.9%|2.8%|0.4%|0.0%|
|LSQ|81.7%|27.0%|7.0%|1.7%|
|Plain + ALP|71.5%|23.6%|11.7%|10.7%|
|50% AT + ALP|70.4%|21.8%|11.5%|10.5%|
|50% AT|73.8%|18.6%|8.0%|7.3%|


### Our Implementation

|method|val|aval - p1|aval - p2|aval-p3|
|:--|--:|--:|--:|--:|
|Plain|87.0%|0.0%|0.0%|%|
|CLP|84.9%|35.2%|7.0%|%|
|LSQ|84.8%|36.3%|5.9%|%|
|Plain + ALP|71.4%|23.0%|12.7%|%|
|50% AT + ALP|72.3%|19.9%|8.8%|%|
|50% AT|78.0%|17.1%|7.8%|%|


## MNSIT

### Experimental Setup
- Attack : PGD-Linf
- Epsilon : 76.5 / 255.0
- Others :

    |set_name|eps|eps_iter|num_steps|restarts|
    |:--|--:|--:|--:|--:|
    |p1|0.3|2.55/255|200|1|
    |p2|0.3|50.0/255|200|1|
    |p3|0.3|50.0/255|200|10000|


### Original Paper

|method|val|aval - p1|aval - p2|
|:--|--:|--:|--:|
|Plain|99.2%|0.0%|0.0%|
|CLP|98.8%|62.4%|29.1%|
|LSQ|98.8%|70.6%|39.0%|
|Plain + ALP|98.7%|95.7%|93.8%|
|50% AT + ALP|98.3%|97.2%|95.3%|
|50% AT|99.1%|95.6%|93.1%|


### Original Implementation

|method|val|aval - p1|aval - p2|
|:--|--:|--:|--:|
|Plain|99.3%|0.00%|%|
|CLP|99.0%|69.8%|%|
|LSQ|99.0%|67.9%|%| |Plain + ALP|98.7%|95.7%|%| |50% AT + ALP|98.6%|95.8%|%|
|50% AT|99.1%|94.7%|%|


### My Implementation

|method|val|aval - p1|aval - p2|
|:--|--:|--:|--:|
|Plain|99.3%|0.00%|%|
|CLP|99.1%|72.8%|%|
|LSQ|98.9%|58.6%|%|
|Plain + ALP|98.8%|95.8%|%|
|50% AT + ALP|98.7%|96.1%|%|
|50% AT|99.1%|94.5%|%|