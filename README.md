# Multivariate-Classification
Multivariate Time Series  Classification Using LSTM

### Multivariate Time Series Classification using LSTM - Keras 

**How to Run:**  `main.py` and `Multi-variate-Time-series-Data.xlsx` need to be on the same folder. The script will create four images
**dataset Stats:**
`Multi-variate-Time-series-Data.xlsx`
- Total Number of Time Series : 205
- Data For Training :  72%
- Data For Validation : 8%
- Data For Testing : 20%

**Method Overview :** I have used Keras framework and an LSTM Network to design the model
**Train-Test Data Generation**

 **Model Sumary**
![](sample_result/modelSummary.png) 

 **Parameter setting**
`learning rate=0.001`
`nb_epoch=50`
`batch_size=64`

 **Output:**

1. Performance of model on `test set`,

Train ROC             |  Train Confusion matrix
:-------------------------:|:-------------------------:
![](sample_result/test_roc.png)  |  ![](sample_result/test_conf.png)

2. Performance of model on `train set`,

Test ROC             |  Test Confusion matrix
:-------------------------:|:-------------------------:
![](sample_result/train_roc.png)  |  ![](sample_result/train_conf.png)


 
 **Parameter Setting:** The following changes can be done to adjust the paraeters.
 
