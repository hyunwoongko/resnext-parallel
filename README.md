# ResNeXt-Pararllel
* Keras Implementation of "Aggregated Residual Transformations for Deep Neural Networks" (CVPR 2017)
* You can check the paper here : [https://arxiv.org/pdf/1611.05431.pdf](https://arxiv.org/pdf/1611.05431.pdf) 
* Support parallel group convolution operation. (using tf.while_loop)
* Only support tensorflow backend. (theano backend doesn't work)
<br><br><br>

## 1. Group Convolution

![fig](https://user-images.githubusercontent.com/38183241/66824032-6ae9b980-ef82-11e9-988b-4db471c7d913.png)
<br><br>
![code](https://user-images.githubusercontent.com/38183241/67158691-55a3cf00-f376-11e9-8a0f-8d3ea26f9f30.png)
<br><br><br>

## 2. Block Architecture (Basic Block)

![fig](https://user-images.githubusercontent.com/38183241/66823637-9ddf7d80-ef81-11e9-8482-c6c45591e6a0.png)
<br><br>
![code](https://user-images.githubusercontent.com/38183241/67158135-263d9400-f36f-11e9-95fb-e5e619e62cb4.png)
<br><br><br>

## 3. Network Architecture (ResNeXt-18)

![fig](https://user-images.githubusercontent.com/38183241/66823390-1db91800-ef81-11e9-87f2-f70e15ee81b9.png)
<br><br>
![code](https://user-images.githubusercontent.com/38183241/66823932-2827e180-ef82-11e9-9a37-e084f0c2a0f2.png)
<br><br><br>

## 4. Licence

    Copyright 2019 Hyunwoong Go

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
