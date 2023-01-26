# leture7
## Deep Q Network
이전 실습에서 Q Network를 통해 수행한 결과를 보니 생각보다 높지 않다
![image](https://user-images.githubusercontent.com/121830114/214941871-13e4b068-0451-488d-977d-1e2ff9c535c6.png)
leture 6에 진행했던 Q network는 최적의 정책으로 수렴하지 않음, 즉, 발산한다고 표현

### optimal policy로 수렴하지 않는 이유
1. Network가 너무 단순하고 얕음(Shallow Network) <br>
2. 입력으로 들어가는 dataset들 간 correlation이 큼(독립적인 사건이 아닌, 서로 간의 종속을 갖는 데이터들을 이용하여 Network를 학습시키기 때문에, 제대로 된 parameter update가 일어나지 않음 <br>
3. non-staionary targets 문제 target이 고정적이지 않고 계속 움직임 <br>
Q Network target 수식

모델 예측값 수식

Network parameter update
