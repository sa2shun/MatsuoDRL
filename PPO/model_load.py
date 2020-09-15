import gym
import retro
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN,PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from baselines.common.retro_wrappers import *
from util import log_dir, callback, AirstrikerDiscretizer, CustomRewardAndDoneEnv
import time

# 環境の生成
env = retro.make(game='Airstriker-Genesis', state='Level1')
env = AirstrikerDiscretizer(env)  # 行動空間を離散空間に変換
env = CustomRewardAndDoneEnv(env)  # 報酬とエピソード完了の変更
env = StochasticFrameSkip(env, n=4, stickprob=0.25)  # スティッキーフレームスキップ
env = Downsample(env, 2)  # ダウンサンプリング
env = Rgb2gray(env)  # グレースケール
env = FrameStack(env, 4)  # フレームスタック
env = ScaledFloatFrame(env)  # 状態の正規化
env = Monitor(env, log_dir/demo, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# モデルの読み込み (1)
model = PPO2.load('airstriker_model')

# モデルのテスト
state = env.reset()
total_reward = 0
for i in range(5000):
    # 環境の描画
    env.render()

    # スリープ
    time.sleep(1 / 30)

    # モデルの推論
    action, _ = model.predict(state)

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    # エピソード完了
    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0
