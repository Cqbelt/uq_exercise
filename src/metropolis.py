import numpy as np
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 自由度为3的t分布
def t_distribution(x):    # n=3
    p = 2/(np.sqrt(3)*np.pi*np.square(1+np.square(x)/3))
    return p

T = 10000   # 迭代次数
sigma = 1.  # 正态分布标准差
sample_x = np.zeros(T+1)
sample_x[0] = uniform.rvs(size=1)[0]   # 初始化马尔科夫链初值
for i in range (1, T+1):
    hat_x = norm.rvs(loc = sample_x[i-1], scale=sigma, size=1, random_state=None)   # 从正态分布中生成候选值
    alpha = min(1, t_distribution(hat_x[0])/t_distribution(sample_x[i-1]))  # 计算接受概率
    alpha_t = uniform.rvs(size=1)[0]  # 生成接受概率判定值
    if alpha_t <= alpha :      # 若判定值小于接受概率则接受候选值，否则拒绝候选值
        sample_x[i] = hat_x[0]
    else:
        sample_x[i] = sample_x[i-1]

fig, ax = plt.subplots(1, 1)
df = 3   # t分布的自由度为3
mean, var, skew, kurt = t.stats(df, moments='mvsk')
x = np.linspace(t.ppf(0.01, df), t.ppf(0.99, df), 100)
p1 = ax.plot(x, t.pdf(x, df), 'k-', lw=5, alpha=0.6, label='t pdf')     # pdf: Probability density function；画自由度为3的标准t分布曲线
p2 = plt.hist(sample_x[9001:], 100, density=True, alpha=0.5, histtype='stepfilled', facecolor='red', label='sample_t')   # 画生成的马尔科夫链的标准化柱状图
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title(u'用MCMC法采样自由度为3的t-分布')
plt.show()

