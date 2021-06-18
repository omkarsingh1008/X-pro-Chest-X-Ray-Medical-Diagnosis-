# Chest X-Ray Medical Diagnosis with Deep Learning
![alt text](https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg)

## about the project
chest x-ray classification based computer vision algorithm. it will detect 14 disease. dieases which it will can predict are mass, edema,fibrosis etc.

we will be using [chestX-ray8 dataset](https://arxiv.org/abs/1705.02315) which contains 108,948 frontal-view x-ray images of 32,717 unique patients.

1. Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions.
2. These in turn can be used by physicians to diagnose 8 different diseases.
3. We will use this data to develop a single model that will provide binary classification predictions for each of the 14 labeled pathologies.
4. In other words it will predict 'positive' or 'negative' for each of the pathologies.

but we use sampale of this dataset. the dataset includes a CSV file that provides the labels for each X-ray

## technology stack

* tensorflow
* keras
* matplotlib
* seaborn
* pandas
* numpy
* opencv
* css
* html
* Flask

---------------------------------------------------------------------------------------------------------------------------------------------------
when we deal with healthcare problem we face many challenges.
* Class imbalance
* data size

## class imbalance problem
one of the challenges with working whith medical diagnostic datasets is the large class imbalance present in such datasets. Lets plot the frequency of each of labels in our dataset:

![alt text](https://github.com/omkarsingh1008/X-pro-Chest-X-Ray-Medical-Diagnosis-/blob/main/download.png)

you can see that the was imbalanced. if we train model on this dataset it will be prioritize the majority class, since it contributes more to the loss.

let's visualize these two contribution ratios next to each other for each of the pathologies:

![alt text](https://github.com/omkarsingh1008/X-pro-Chest-X-Ray-Medical-Diagnosis-/blob/main/download%20(5).png)

As we see in the above plot, the contributions of positive cases is significantly lower than that of the negative ones. However, we want the contributions to be equal. One way of doing this is by multiplying each example from each class by a class-specific weight factor, w_pos and w_neg, so that the overall contribution of each class is the same.

after applying freq :-

![alt text](https://github.com/omkarsingh1008/X-pro-Chest-X-Ray-Medical-Diagnosis-/blob/main/download%20(6).png)

now you can see each class equal participate in loss

* we also remove the data leakage from dataset means that each pateints image should me in same dataset it maybe in train or test.

``` bash
def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 : dataframe describing first dataset
        df2 (: dataframe describing second dataset
        patient_col : string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    
    
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    
    leakage = len(patients_in_both_groups) > 0 
    
   
    
    return leakage
 ```
 
if there is leakage in dataset it will return True if not then False.

* accuacy and loss after 100 epoch is  loss: 0.0210  binary_accuracy: 0.9932.

# Flask app

![alt text](https://github.com/omkarsingh1008/X-pro-Chest-X-Ray-Medical-Diagnosis-/blob/main/web_image.png)

## Local installation
1. clone git repository
```bash
https://github.com/omkarsingh1008/X-pro-Chest-X-Ray-Medical-Diagnosis-.git
```
2. install the packages
3.at last, push in the command
```bash
python app.py
```
