# ERA-S6-Assignment

## Part 1
Following the screenshot of the excel sheet:

<img width="1069" alt="Screen Shot 2023-06-10 at 12 13 34 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/c5fe9eb4-0089-4f0b-a65c-06ed973c5f1e">

### Major Steps in Backpropagation

- For BackPropagation to happen, first forward propagation must happen
- h1 is first calculated based as `w1*i1 + w2*i2`
- Activation function is applied on h1 to get a_h1.
- Similarly for h2 the calculation is `w3*i1 + w4*i2`.
- Activation function is applied on h2 to get a_h2.
- Output neurons are calculated similarly with a_h1 and a_h2 as input neurons and w5, w6, w7 and w8 as weights
- Activation function is applied on the output neurons o1 and o2 to get a_o1 and a_o2.
- Individual loss (E1 and E2) are calculated using predefined formulae. Then E_Total is updated.

Sole objective of BackPropagation is to update the weights so that desired loss is achieved.

- For that we need to calculate, how oes E_total change for slight change in the weights.
- We need to go back step by step . First how E1 and E2 are affected by these weights.
- E1 and E2 change with respect to a_o1 and a_o2 is calculated.
- a_o1 and a_o2 change with respect to o1 and o2 are calculated.
- o1 and o2 change with respect to w5, w6, w7 and w8 are calculated.
- E1 and E2 change with respect to a_h1 and a_h2 is calculated.
- a_h1 and a_h2 change with respect to h1 and h2 are calculated.
- h1 and h2 change with respect to w1, w2, w3 and w4 are calculated.

Thus in backpropagation, we follow the same path for forward propagation except in backward direction.

- Weights are updated as `wn' = wn - learning_rate * delta change in loss with respect to wn`



### Change in graph based on learning rate


learning rate = 0.1

<img width="490" alt="Screen Shot 2023-06-10 at 7 27 58 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/9a5eb173-c032-4d39-8f33-e72cba989ad9">

learning rate = 0.2

<img width="487" alt="Screen Shot 2023-06-10 at 7 21 28 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/b43b2684-1854-4ea5-9961-af6b046e2f74">


learning rate = 0.5

<img width="489" alt="Screen Shot 2023-06-10 at 7 22 49 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/a60d203f-8023-43a5-ab5f-724cfcb8b3f6">


learning rate = 0.8

<img width="485" alt="Screen Shot 2023-06-10 at 7 24 27 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/cbe69608-0035-44a9-84f9-2a9059dcb49a">


learning rate = 1

<img width="486" alt="Screen Shot 2023-06-10 at 7 26 23 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/11396389-35ca-4824-8bf5-4ba6c12de8af">

learning rate = 2

<img width="490" alt="Screen Shot 2023-06-10 at 7 27 19 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/dee37c14-cb6c-4e17-b5fd-5744a2cdf29c">


Thus we see that as learning rate increases the graph falls steeply which means the rate of change increases if we increase the learning rate. Too small a rate will take a lot of time to converge, too high a rate will converge instantly. Optimal value is desired.



## Part 2

Optimized model

```
class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      # Extracting 16 features using 3x3 kernel but keeping size same
      self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #rin = 1 rout= 3
      # Performing batchNormalization
      self.bn1 = nn.BatchNorm2d(16)
      # Performing maxPooling assuming 1st level of features are extracted
      self.pool1 = nn.MaxPool2d(2, 2); #rin = 3 rout= 4
      # Avoiding overfitting
      self.dropout1 = nn.Dropout(0.10);
      # Extracting 2nd level of features
      self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #rin = 4 rout= 8
      # Performing batchNormalization
      self.bn2 = nn.BatchNorm2d(32)
      # Performing maxPooling assuming 2nd level of features are extracted
      self.pool2 = nn.MaxPool2d(2, 2); #rin = 8 rout= 10
      # Avoiding overfitting
      self.dropout2 = nn.Dropout(0.10);
      # Performing fully connected but maintaining spatial information
      self.conv3 = nn.Conv2d(32, 64, 1) #rin = 10 rout= 10
      self.bn3 = nn.BatchNorm2d(64)
      # Extract the important information and increase receptive field
      self.pool3 = nn.MaxPool2d(2, 2); #rin = 10 rout = 14
      # Getting info for 10 classes
      self.conv4 = nn.Conv2d(64, 10, 3) #rin = 14 rout= 30
      
  def forward(self, x):
    x = self.pool1(self.bn1(F.relu(self.conv1(x))))
    x = self.dropout1(x)
    x = self.pool2(self.bn2(F.relu(self.conv2(x))))
    x = self.dropout2(x)
    x = self.pool3(self.bn3(F.relu(self.conv3(x))))
    x = self.conv4(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=1)
  ```
  
- I started with 16 features to keep the overall count low, and this is MNIST data
- Maximum number of features is 64
- Dropout percentage was 10, more than that the weight change was too slow.
- Maxpooling mainly I did to increase the receptive field.
- Total receptive field achieved is 30 for 28 x 28
- I avoided fully connected layers, as I could not go above 99% with that.
- Scheduler also I used because towards the end, the losses with jumping from high to low back to high.
- Total parameters is 12906
- Maximum Validation accuracy achieved is 99.33

Model Summary 

<img width="581" alt="Screen Shot 2023-06-10 at 7 51 32 AM" src="https://github.com/kurchi1205/ERA-S6-Assignment/assets/40196782/2276636a-2c97-4f2a-b0aa-03ffcfffe32f">
