---
title: "MDS를 활용한 주가 데이터의 기하학적 분석에 관하여"  # 글 제목 (한글 가능)
date: 2025-12-14 16:00:00 +0900           # 날짜 및 시간
categories: [Research, Finance]           # [대분류, 소분류]
tags: [python, econophysics, mds]         # 태그 (소문자 권장)
math: true                                # 수학 수식($$) 켜기 (필수!)
toc: true                                 # 우측에 목차 자동 생성
---


## 1. 연구 개요 (Introduction)

주식 시장은 수천 개의 종목이 서로 영향을 주고받는 **거대한 복잡계(Complex System)**이다. 기존의 시계열 분석(Time-series Analysis)이 개별 종목의 시간적 변동에 집중했다면, 이 연구는 종목 간의 **'관계(Relationship)'**를 공간상의 **'거리(Distance)'**로 치환하여 시장의 숨겨진 구조를 파악하는 것을 목표로 한다.

본 포스트에서는 100여 개 주식 티커의 로그 수익률 데이터를 바탕으로 **피어슨 상관계수(Pearson Correlation)**를 거리 함수로 정의하고, **MDS(다차원 척도법)**와 **t-SNE**를 이용해 이를 2차원 공간에 시각화(Embedding)한다. 이를 통해 시장의 위상학적 구조(Topological Structure)와 섹터별 군집화(Clustering) 현상을 기하학적으로 분석해 본다.

## 2. 이론적 배경 (Theoretical Background)

### 2.1. 상관관계의 거리 변환 (Metric Definition)
주가 $i$와 $j$의 상관계수 $\rho_{ij}$는 $-1$에서 $1$ 사이의 값을 가진다. 이를 유클리드 공간상의 거리(Metric)로 변환하기 위해 다음과 같은 수식을 사용한다. 이는 정규화된 수익률 벡터 사이의 유클리드 거리와 수학적으로 동치이다.

$$
d_{ij} = \sqrt{2(1 - \rho_{ij})}
$$

* $\rho_{ij} = 1$ (완전 동조): 거리 $d = 0$
* $\rho_{ij} = 0$ (무상관): 거리 $d = \sqrt{2}$
* $\rho_{ij} = -1$ (역상관): 거리 $d = 2$

### 2.2. 차원 축소 알고리즘 (Dimensionality Reduction)

* **MDS (Multidimensional Scaling):** 객체 간의 거리 행렬(Distance Matrix)을 입력받아, 그 거리 관계를 최대한 보존하는 좌표를 찾는다. 데이터의 전역적(Global) 구조를 파악하는 데 유리하다.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** 고차원 공간에서의 이웃 관계(확률 분포)를 저차원에서도 유지하도록 학습한다. 가까운 데이터끼리의 군집(Local Structure)을 시각화하는 데 탁월하다.

## 3. 데이터 및 방법론 (Methodology)

### 3.1. 데이터 전처리
가격(Price) 데이터는 비정상성(Non-stationary)을 가지므로, 이를 제거하기 위해 **로그 수익률(Log Returns)**로 변환하여 사용한다.

$$r_t = \ln(P_t) - \ln(P_{t-1})$$

### 3.2. Python 구현 코드
`scikit-learn` 라이브러리를 활용하여 MDS와 t-SNE를 구현하였다.

```python
import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt

# 1. 데이터 준비 (Log Returns)
# price_df는 (Time x Tickers) 형태의 DataFrame 가정
log_returns = np.log(price_df / price_df.shift(1)).dropna()

# 2. 거리 행렬 계산 (Distance Matrix)
corr_matrix = log_returns.corr()
dist_matrix = np.sqrt(2 * (1 - corr_matrix))
dist_matrix = dist_matrix.fillna(0)

# 3. 모델링: MDS (Global Structure)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords_mds = mds.fit_transform(dist_matrix)

# 4. 모델링: t-SNE (Local Cluster)
# metric='precomputed' 설정 필수
tsne = TSNE(n_components=2, metric='precomputed', perplexity=30, random_state=42)
coords_tsne = tsne.fit_transform(dist_matrix)
