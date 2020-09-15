import pandas as pd
import matplotlib.pyplot as plt

# monitor.csvの読み込み (1)
df = pd.read_csv('logs/moniter2.csv', names=['r', 'l', 't'])
df = df.drop(range(2)) # 1〜2行目の削除

# 報酬のグラフの表示 (2)
fig = plt.figure()
x = range(len(df['r'])) # エピソードのインデックス
y = df['r'].astype(float) # 報酬
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
fig.savefig('PPO2')

# エピソード長のグラフの表示 (2)
x = range(len(df['l'])) # エピソードのインデックス
y = df['l'].astype(float) # エピソード長
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('episode len')
plt.show()