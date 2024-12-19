# Simple Deep Learning Projects
jupyter notebook 단위로 project 구현 / Imcommit 채널에서 각각에 대해 다룰지도
 
 
### 1. DDPM-notebook.ipynb : DDPM 논문 알고리즘 구현
Paper: [https://arxiv.org/pdf/2006.11239.pdf](https://arxiv.org/pdf/2006.11239.pdf)<br />
Youtube: [https://www.youtube.com/watch?v=svSQhYGKk0Q](https://www.youtube.com/watch?v=svSQhYGKk0Q)
    
    실험적인 세부사항(Exponential Moving Average 등)은 제외
    - coefficient 구현이 핵심 
    Dataset: CIFAR-10

### 2. pi_digit_estimation_GRU.ipynb : π 패턴 예측
[https://youtu.be/kdmrlMAaCiA](https://youtu.be/kdmrlMAaCiA)

    될 리가 있나

### 3. Neural_ODE.ipynb : official code 약간 수정
Paper: [https://arxiv.org/pdf/1806.07366.pdf](https://arxiv.org/pdf/1806.07366.pdf)<br />
Official github(Reference): [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)<br />
[https://www.youtube.com/watch?v=NS-C_QjjcT4](https://www.youtube.com/watch?v=NS-C_QjjcT4)

    출력 코드 정리
    model 구성 변경(official code보다 학습이 약간 어렵도록)
    Dataset: toy example

### 4. vae.ipynb : VAE, AE, AE with z-regularization
paper: [https://arxiv.org/pdf/1312.6114.pdf](https://arxiv.org/pdf/1312.6114.pdf)<br />

    Autoencoder: reconstruction loss only
    AE with z-regularization: loss = reconstruction loss + z.abs().mean()
    VAE: loss = reconstruction loss + kl divergence
    Dataset: MNIST dataset

### 5. flowmatching.ipynb : flow matching for generative modeling
Paper: [https://arxiv.org/pdf/2210.02747](https://arxiv.org/pdf/2210.02747) <br />

    실험적인 세부사항(Exponential Moving Average 등)은 제외
    - train과 sample 함수가 핵심
    - Dataset: MNIST
