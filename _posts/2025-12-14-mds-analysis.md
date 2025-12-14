---
title: "MDS를 활용한 주식 위상 수학적 분석"  # 글 제목 (한글 가능)
date: 2025-12-14 16:00:00 +0900           # 날짜 및 시간
categories: [Research, Finance]           # [대분류, 소분류]
tags: [python, econophysics, mds]         # 태그 (소문자 권장)
math: true                                # 수학 수식($$) 켜기 (필수!)
toc: true                                 # 우측에 목차 자동 생성
---

## 1. 연구 개요
이 프로젝트는 주식 시장을 하나의 복잡계 네트워크로 보고...

## 2. 수식 배경
피어슨 상관계수 $\rho_{ij}$를 거리 $d_{ij}$로 변환하는 식은 다음과 같다.

$$
d_{ij} = \sqrt{2(1 - \rho_{ij})}
$$

## 3. 코드 및 결과
MDS를 적용하여 100개 종목을 2차원에 투영하였다.

```python
import pandas as pd
from sklearn.manifold import MDS

# MDS 좌표 변환 코드
embedding = MDS(n_components=2)
