# leture7
## Deep Q Network
이전 실습에서 Q Network를 통해 수행한 결과를 보니 생각보다 높지 않다
![image](https://user-images.githubusercontent.com/121830114/214941871-13e4b068-0451-488d-977d-1e2ff9c535c6.png)<br>
leture 6에 진행했던 Q network는 최적의 정책으로 수렴하지 않음, 즉, 발산한다고 표현

### optimal policy로 수렴하지 않는 이유
1. Network가 너무 단순하고 얕음(Shallow Network) <br>
해결책: 충분한 weight와 hidden layer를 추가
2. 입력으로 들어가는 dataset들 간 correlation이 큼(독립적인 사건이 아닌, 서로 간의 종속을 갖는 데이터들을 이용하여 Network를 학습시키기 때문에, 제대로 된 parameter update가 일어나지 않음 <br>
해결책: 모든 데이터는 즉각적으로 Network에 들어가는 대신 buffer에 저장하는 방식 취함<br>
loss function 최소화에 쓰이는 데이터인 state, action, reward, 다음 state를 계속 쌓아둔 후 일정 시간 이후 buffer에 쌓인 데이터 중 random하게 데이터를 뽑아 학습에 사용(데이터 간 correlation 피할 수 있음)
<img width="727" alt="스크린샷 2023-01-27 오전 5 40 11" src="https://user-images.githubusercontent.com/121830114/214945739-570c0e3a-6787-4097-9fae-57ae42a3fda4.png">
3. non-staionary targets 문제 target이 고정적이지 않고 계속 움직임 <br>
<br>
Q Network target 수식
<img width="452" alt="스크린샷 2023-01-27 오전 5 27 00" src="https://user-images.githubusercontent.com/121830114/214943534-ccb644c6-1e6f-4c7d-9247-836064305361.png">
모델 예측값 수식
<img width="222" alt="스크린샷 2023-01-27 오전 5 27 09" src="https://user-images.githubusercontent.com/121830114/214943546-cd2ce88a-f75e-49a6-af8b-4ee8dfd03326.png">
Network parameter update
<img width="629" alt="스크린샷 2023-01-27 오전 5 27 22" src="https://user-images.githubusercontent.com/121830114/214943563-7aacde05-3cc8-4c0d-9015-a2e483018dd9.png">
예측값을 내는 데에 사용하는 네트워크(theta)와 target값을 내는 데에 사용하는 네트워크(theta)가 같은 네트워크<br>
즉, 모델이 target과의 차이(error)를 줄이기 위해 네트워크를 업데이트 시키는 순간, target의 네트워크도 함께 업데이트 됨<br>
해결책: target Network를 분리시킴. 즉, 두 개의 Network 사용<br>

위 세개의 문제점을 해결한 알고리즘
