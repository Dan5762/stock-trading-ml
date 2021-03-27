import logging

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from data.data_prep import prepare_data
from basic_model import create_dataset_x, load_model
from agent_env import StockTradingEnv


# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


class Model(tf.keras.Model):

    def __init__(self, num_obs, num_actions):
        super().__init__('mlp_policy')
        # Note: no tf.get_variable(), just simple Keras API!
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.actions = kl.Dense(num_actions, activation='sigmoid', name='value')
        self.value = kl.Dense(1, activation='relu', name='value')

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_actions = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.actions(hidden_actions), self.value(hidden_vals)

    def action_value(self, obs):
        obs = obs.reshape(1, len(obs))
        # Executes `call()` under the hood.
        actions, value = self.predict_on_batch(obs)

        return np.squeeze(actions, axis=0), np.squeeze(value, axis=-1)


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = kl.Dense(num_hidden_units, activation="relu")
        self.actor = kl.Dense(num_actions)
        self.critic = kl.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


def env_step(action: np.ndarray):
    """Returns state, reward and done flag given an action."""
    state, reward, done, _ = env.step(action)

    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action: tf.Tensor):
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int):
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True):
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


def get_dataset(sector, model_name, look_back):
    dates, close_prices, normalised_close_prices, N_tickers, normaliser, symbols = prepare_data(sector, split=False, save_model_stocks=False)

    x = create_dataset_x(normalised_close_prices, look_back)
    y_test_real = close_prices[look_back:]

    model = load_model(model_name)
    y_pred = model.predict(x)

    d = {symbol: close_price for symbol, close_price in zip(symbols, y_test_real.T)}
    d.update({f"{symbol}_grad": pred_grad for symbol, pred_grad in zip(symbols, y_pred.T)})

    df = pd.DataFrame(data=d)

    return df


if __name__ == "__main__":
    sector = 'tech'
    model_name = 'tech_lstm'
    look_back = 50

    data_df = get_dataset(sector, model_name, look_back)

    env = StockTradingEnv(data_df)

    with tf.Graph().as_default():
        model = Model(num_obs=env.observation_space.n, num_actions=env.action_space.n)

        agent = A2CAgent(model)
        rewards_history = agent.train(env)
        print("Finished Training")
        print(agent.test(env))
