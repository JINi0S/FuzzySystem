#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install -c conda-forge/label/gcc7 scikit-fuzzy 


# In[5]:


import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_fund = np.arange(0, 101, 1)
x_ps = np.arange(0, 101, 1)
x_risk  = np.arange(0, 101, 1)

# fuzzy membership functions 생성
fund_lo = fuzz.trapmf(x_fund, [0, 0, 25, 45])
fund_md = fuzz.trimf(x_fund, [31, 50, 69])
fund_hi = fuzz.trapmf(x_fund, [55, 75, 100, 100])

ps_lo = fuzz.trapmf(x_ps, [0, 0, 25, 64])
ps_md = fuzz.trapmf(x_ps, [36, 75, 100, 100])

risk_lo = fuzz.trapmf(x_risk, [0, 0, 25, 42])
risk_md = fuzz.trapmf(x_risk, [30, 45, 55, 70])
risk_hi = fuzz.trapmf(x_risk, [58, 75, 100, 100])

# 시각화
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_fund, fund_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_fund, fund_md, 'g', linewidth=1.5, label='Middle')
ax0.plot(x_fund, fund_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('Project funding')
ax0.legend()

ax1.plot(x_ps, ps_lo, 'b', linewidth=1.5, label='Less')
ax1.plot(x_ps, ps_md, 'r', linewidth=1.5, label='Many')
ax1.set_title('Project  personnel')
ax1.legend()

ax2.plot(x_risk, risk_lo, 'b', linewidth=1.5, label='Low')
ax2.plot(x_risk, risk_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_risk, risk_hi, 'r', linewidth=1.5, label='High')
ax2.set_title('Risk')
ax2.legend()

# 상, 우측 축 지우기
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()


# 퍼지화 - 퍼센트 설정
fund_level_lo = fuzz.interp_membership(x_fund, fund_lo, 35)
fund_level_md = fuzz.interp_membership(x_fund, fund_md, 35)
fund_level_hi = fuzz.interp_membership(x_fund, fund_hi, 35)

ps_level_lo = fuzz.interp_membership(x_ps, ps_lo, 60)
ps_level_md = fuzz.interp_membership(x_ps, ps_md, 60)


# Rule1,2,3 설정
active_rule1 = np.fmax(fund_level_hi, ps_level_lo)
risk_activation_lo = np.fmin(active_rule1, risk_lo)  

active_rule2 = np.fmin(fund_level_md, ps_level_md)
risk_activation_md = np.fmin(active_rule2, risk_md)

risk_activation_hi = np.fmin(fund_level_lo, risk_hi)
risk0 = np.zeros_like(x_risk)


# Rule1 시각화
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_risk, risk0, risk_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_risk, risk_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.set_title('Output Rule1')

# 상, 우측 축 지우기
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()


# Rule2시각화
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_risk, risk0, risk_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_risk, risk_md, 'g', linewidth=0.5, linestyle='--')
ax0.set_title('Output Rule2')

# 상, 우측 축 지우기
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()


# Rule3 시각화
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_risk, risk0, risk_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_risk, risk_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output Rule3')

# 상, 우측 축 지우기
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()


# Rule 전체 시각화
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_risk, risk0, risk_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_risk, risk_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_risk, risk0, risk_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_risk, risk_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_risk, risk0, risk_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_risk, risk_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Integration of rule')

# 상, 우측 축 지우기
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()



# 세 가지 출력 멤버십 함수를 모두 집계
aggregated = np.fmax(risk_activation_lo,
                     np.fmax(risk_activation_md, risk_activation_hi))

# 역퍼지화 - 중심 계산
risk = fuzz.defuzz(x_risk, aggregated, 'centroid')
risk_activation = fuzz.interp_membership(x_risk, aggregated, risk)  # for plot

# 역퍼지화 시각화
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_risk, risk_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_risk, risk_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_risk, risk_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_risk, risk0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([risk, risk], [0, risk_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Defuzzification result (line)')

# 상, 우측 축 지우기
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()


print(risk)


# In[ ]:





# In[ ]:




