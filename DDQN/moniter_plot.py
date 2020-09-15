import pandas as pd
import matplotlib.pyplot as plt

# monitor.csvの読み込み (1)
df = pd.read_csv('logs/monitor.csv', names=['r', 'l', 't'])
df = df.drop(range(2)) # 1〜2行目の削除

# 報酬のグラフの表示 (2)
x = range(len(df['r'])) # エピソードのインデックス
y = df['r'].astype(float) # 報酬
fig = plt.figure()
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
fig.savefig('DDQN.png')

# エピソード長のグラフの表示 (2)
x = range(len(df['l'])) # エピソードのインデックス
y = df['l'].astype(float) # エピソード長
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('episode len')
plt.show()