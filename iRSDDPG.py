import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Concatenate, LSTM
from tensorflow.keras import Model
from collections import deque
from utils import soft_update_weights, print_env_step_info


class Environment(object):
    """ My Environment
    """

    def __init__(self, data_path, repeat_time, state_size, action_size, reward_size):
        self.data_path = data_path
        self.repeat_time = repeat_time
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.state_data, self.action_data, self.reward_data, self.next_state_data = self.init_env_data()
        self.step_position = 0

    def init_env_data(self):
        data = pd.read_excel(self.data_path)
        datas = [data] * self.repeat_time
        all_data = pd.concat(datas)

        temp_size = 0
        state_data = all_data.iloc[:, :temp_size + self.state_size]
        temp_size = temp_size + self.state_size
        action_data = all_data.iloc[:, temp_size: temp_size + self.action_size]
        temp_size = temp_size + self.action_size
        reward_data = all_data.iloc[:, temp_size: temp_size + self.reward_size]
        temp_size = temp_size + self.reward_size
        next_state_data = all_data.iloc[:, temp_size: temp_size + self.state_size]

        return state_data.values, action_data.values, reward_data.values, next_state_data.values

    def get_action_size(self):
        return len(self.action_data)

    def get_state_size(self):
        return self.state_data.shape

    def reset(self):
        """ Resets the game, clears the state buffer.
        """
        # Clear the state buffer
        self.step_position = 0

    def step(self):
        """
        :param action:
        :return:
        """
        done = False
        if self.step_position >= len(self.state_data) - 1:
            done = True

        state_ = self.state_data[self.step_position]
        action_ = self.action_data[self.step_position]
        next_state_ = self.next_state_data[self.step_position]
        reward_ = self.reward_data[self.step_position][0]

        self.step_position += 1

        return state_, action_, next_state_, reward_, done


num_episodes = 2000
num_steps = 5882
num_his = 8  # T the length of the history
test_num_steps = 100
obs_dim, state_size = 33, 33
action_dim, action_size = 3, 3
action_ub = None
action_lb = None

env_data_path = r'C:\Users\Think\Desktop\RL\rf_learning\Data\raw_data_RL-train.xlsx'
reward_size = 1

env = Environment(env_data_path, 1, state_size, action_size, reward_size)

buffer_size = 10000  # replay_buffer size
lstm_size = 256
dense_size = 128
seed = 1
batch_size = 50
initial_learning_rate = 0.0001
decay_steps = 1
decay_rate = 0.9
target_network_update_rate = 0.005
discount = 0.5


class RNNCritic(Model):
    """
    Input:
    obs_history - (batch, T, obs_dim)
    action_history - (batch, T, action_dim)
    Output:
    value_sequence - (batch, T, 1)
    """

    def __init__(self, lstm_size, dense_size, name='RNNCritic'):
        super(RNNCritic, self).__init__(name=name)

        self.lstm1 = LSTM(lstm_size, name="LSTM1", return_sequences=True)
        self.dense1 = Dense(dense_size, name="HiddenDense", activation='relu')
        self.dense2 = Dense(1, name="OutputDense")
        self.concat = Concatenate()

    def call(self, obs_history, action_history):
        lstm_in = self.concat([obs_history, action_history])
        lstm_out = self.lstm1(lstm_in)
        x = self.dense1(lstm_out)
        x = self.dense2(x)

        return x
        # squeeze the second dimension so that the output shape will be (batch, )


class RNNActor(Model):
    """
    Input:
    obs_history - (batch, T, obs_dim) o_1, ..., o_T
    action_history - (batch, T, action_dim) a_0, ..., a_(T-1)
    Output:
    action - (batch, T, action_dim) a*_1, ..., a*_T
    """

    def __init__(self,
                 action_dim,
                 lstm_size,
                 dense_size,
                 his_length,
                 obs_size,
                 act_size,
                 action_lb=None,
                 action_ub=None,
                 name='RNNActor'):
        super(RNNActor, self).__init__(name=name)

        self.action_lb = action_lb
        self.action_ub = action_ub
        self.action_dim = action_dim
        self.lstm1 = LSTM(lstm_size, name="LSTM1", return_sequences=True)
        self.dense1 = Dense(dense_size, name="HiddenDense", activation='relu')
        self.dense2 = Dense(action_dim, name="OutputDense", activation='sigmoid')
        self.concat = Concatenate()

    def call(self, inputs):
        # batch_size = action_history.shape[0]
        #  action0 = tf.zeros([batch_size, 1, self.action_dim])
        # print(f"obs_hist shape: {obs_history.shape}")
        # print(f"action_hist shape: {action_history.shape}")
        # print(f"action0 shape: {action0.shape}")
        #  augmented_action_history = tf.concat([action0, action_history], axis=1)
        # lstm_in = self.concat([obs_history, action_history])
        x = self.lstm1(inputs)
        x = self.dense1(x)
        action = self.dense2(x)

        if self.action_lb is not None and self.action_ub is not None:
            mid = (self.action_lb + self.action_ub) / 2
            span = (self.action_ub - self.action_lb) / 2
            action = span * tf.nn.tanh(action) + mid
        return action


class History:
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.obs_hist = tf.zeros([0, obs_dim])

        self.action_dim = action_dim
        self.action_hist = tf.zeros([0, action_dim])

        self.reward_hist = tf.zeros([0, 1])

        self.size = 0
        self.sum_reward = 0

    @staticmethod
    def _insert(hist, new_value):
        # hist is in tf.float32, hence cast new_value to tf.float32
        new_value = tf.cast(tf.expand_dims(new_value, 0), dtype=tf.float32)
        return tf.concat([hist, new_value], 0)

    def insert_obs(self, obs):
        self.size += 1
        self.obs_hist = self._insert(self.obs_hist, obs)

    def insert_action(self, action):
        self.action_hist = self._insert(self.action_hist, action)

    def insert_reward(self, reward):
        # reward is a scalar, need to convert it to a tensor with 1 dimension
        # first.
        self.sum_reward += reward
        self.reward_hist = self._insert(self.reward_hist, [reward])

    def get_action_history(self):
        return self.action_hist

    def get_obs_history(self):
        return self.obs_hist

    def get_reward_history(self):
        return self.reward_hist


class RNNReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=10000, seed=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.seed = seed
        self.buffer = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(self.seed)

    def put(self, obs_history, action_history, reward_history):
        """
        obs_history: Tensor, sequence_length (T) * obs_dim
        action_history: Tensor, sequence_length (T) * action_dim
        reward_history: Tensor, sequence_length (T) * 1
        """
        self.buffer.append(
            tf.concat([obs_history, action_history, reward_history], 1))

    def get(self, size):
        buffer_arr = np.array(self.buffer, dtype=object)
        samples = buffer_arr[: size]
        samples_tensor = tf.convert_to_tensor(samples, dtype=tf.float32)

        batch_obs_history = samples_tensor[:, :, :self.obs_dim]
        batch_action_history = samples_tensor[:, :,
                               self.obs_dim:(self.obs_dim +
                                             self.action_dim)]
        batch_reward_history = samples_tensor[:, :, (self.obs_dim + self.action_dim):(
                self.obs_dim + self.action_dim +
                1)]
        return batch_obs_history, batch_action_history, batch_reward_history

    def sample(self, batch_size, replacement=True):
        idx = self.rng.choice(self.size(),
                              size=batch_size,
                              replace=replacement)
        buffer_arr = np.array(self.buffer, dtype=object)
        samples = buffer_arr[idx]
        samples_tensor = tf.convert_to_tensor(samples, dtype=tf.float32)

        batch_obs_history = samples_tensor[:, :, :self.obs_dim]
        batch_action_history = samples_tensor[:, :,
                               self.obs_dim:(self.obs_dim +
                                             self.action_dim)]
        batch_reward_history = samples_tensor[:, :, (
                                                            self.obs_dim + self.action_dim):(
                                                                self.obs_dim + self.action_dim +
                                                                1)]
        return batch_obs_history, batch_action_history, batch_reward_history

    def size(self):
        return len(self.buffer)


replay_buffer = RNNReplayBuffer(obs_dim,
                                action_dim,
                                capacity=buffer_size,
                                seed=seed)
test_buffer = RNNReplayBuffer(obs_dim,
                              action_dim,
                              capacity=buffer_size,
                              seed=seed)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

critic_loss_fun = tf.keras.losses.MeanSquaredError()
actor_supervised_loss_func = tf.keras.losses.MeanSquaredError()
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

critic_loss_fun = tf.keras.losses.MeanSquaredError()
critic_1_loss_fun = tf.keras.losses.MeanSquaredError()
actor_supervised_loss_func = tf.keras.losses.MeanSquaredError()
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

critic = RNNCritic(lstm_size, dense_size)
critic_1 = RNNCritic(lstm_size, dense_size)
actor = RNNActor(action_dim,
                 lstm_size,
                 dense_size, num_his, obs_dim, action_dim,
                 action_lb=action_lb,
                 action_ub=action_ub)
target_critic = RNNCritic(lstm_size, dense_size)
target_critic_1 = RNNCritic(lstm_size, dense_size)
target_actor = RNNActor(action_dim,
                        lstm_size,
                        dense_size, num_his, obs_dim, action_dim,
                        action_lb=action_lb,
                        action_ub=action_ub)

# making the weights equal
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())
target_critic_1.set_weights(critic_1.get_weights())

env.reset()

episode = 0
step_history = []
done = False

# generate samples
for t in range(num_steps + test_num_steps):
    if done:
        break

    state_, action_, next_state_, reward_, done = env.step()
    history = History(obs_dim, action_dim)
    # add to queue
    step_history.append(history)
    start_pos = 0 if t < num_his else t - num_his
    stop_pos = len(step_history)
    for i in range(start_pos, stop_pos):
        curr_his = step_history[i]
        if curr_his.size == num_his:
            # print(f'Step {t}: index = {i}: history reward sum = {curr_his.sum_reward}: size = {curr_his.size}')
            if t >= num_steps:
                # testing sample
                test_buffer.put(curr_his.get_obs_history(), curr_his.get_action_history(),
                                curr_his.get_reward_history())
            else:
                # training sample
                replay_buffer.put(curr_his.get_obs_history(), curr_his.get_action_history(),
                                  curr_his.get_reward_history())
        elif t == 0:
            # curr_his.insert_obs(next_state_)
            curr_his.insert_obs(state_)
            curr_his.insert_action(np.zeros([action_dim]))
            curr_his.insert_reward(0)
        else:
            # curr_his.insert_obs(next_state_)
            curr_his.insert_obs(state_)
            curr_his.insert_action(action_)
            curr_his.insert_reward(reward_)

    if done:
        break
print('total number of training samples', replay_buffer.size())
print('total number of testing samples', test_buffer.size())

obs_test, action_test, reward_test = test_buffer.get(test_num_steps)

# training
train_his_info = []
for episode in range(num_episodes):

    print(f'Episode {episode}: replay buffer size = {replay_buffer.size()}')
    episode += 1
    obs_history, action_history, reward_history = replay_buffer.sample(batch_size)

    with tf.GradientTape(persistent=True) as tape:
        # obs_history: 1, ..., T; action_history: 0, 1, ..., T-1
        # target_actions: 1, ..., T;
        target_actions = target_actor(tf.concat([obs_history[:, 1:, :], action_history[:, :-1, :]], axis=2))

        # y1*, ..., yT*; o1, ..., oT; a*_1, ..., a*_T
        target_critic_output = target_critic(obs_history[:, 1:, :], target_actions)
        target_critic_output_1 = target_critic_1(obs_history[:, 1:, :], target_actions)

        # minimum target_critic_output/average target_critic_output
        # min_target_critic_output = tf.minimum(target_critic_output, target_critic_output_1)
        min_target_critic_output = tf.add(target_critic_output, target_critic_output_1) / 2

        # reward_history 0, 1, ..., T - 1, target_critic_output 1, ... T,
        # target_values 0, ..., T - 1
        target_values = reward_history[:, :-1, :] + discount * min_target_critic_output
        target_1_values = reward_history[:, :-1, :] + discount * min_target_critic_output

        # yhat 0, ..., T - 1, obs_history 0, ... T - 1, aciton_history 0, ... T - 1
        Qpredicts = critic(obs_history[:, :-1, :], action_history[:, :-1, :])

        critic_loss = critic_loss_fun(tf.stop_gradient(target_values)[:, -1, :],
                                      Qpredicts[:, -1, :])
        critic_1_loss = critic_loss_fun(tf.stop_gradient(target_1_values)[:, -1, :],
                                        Qpredicts[:, -1, :])

    critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_gradients, critic.trainable_variables))

    critic_1_gradients = tape.gradient(critic_1_loss, critic_1.trainable_variables)
    critic_1_optimizer.apply_gradients(
        zip(critic_1_gradients, critic_1.trainable_variables))

    with tf.GradientTape(persistent=True) as tape:
        # obs_history: 1, ..., T, action_history: 0, 1, ..., T-1,
        # actor_actions: 1, ..., T
        actor_actions = actor(tf.concat([obs_history[:, 1:, :], action_history[:, :-1, :]], axis=2))
        print(actor_actions[-1, -1, :])
        print(action_history[-1, -1, :])
        # tmp_weight = tf.sigmoid(actor.w)
        actor_loss_weight = 0.6
        # actor supervised_loss + Q loss
        actor_supervised_loss = actor_supervised_loss_func(actor_actions[:, -1, :], action_history[:, -1, :])
        predict_q_value = tf.math.reduce_mean(critic(obs_history[:, 1:, :], actor_actions)[:, -1, :])
        supervised_actor_loss = actor_loss_weight * actor_supervised_loss - (1 - actor_loss_weight) * predict_q_value

        # test_critic_loss
        target_test_pred_actions = target_actor(tf.concat([obs_test[:, 1:, :], action_test[:, :-1, :]], axis=2))
        target_test_pred_critic_output = target_critic(obs_test[:, 1:, :], target_test_pred_actions)
        target_test_pred_critic_1_output = target_critic_1(obs_test[:, 1:, :], target_test_pred_actions)
        min_target_critic_test_output = tf.add(target_test_pred_critic_output, target_test_pred_critic_1_output) / 2
        test_target_values = reward_test[:, : -1, :] + discount * min_target_critic_test_output
        test_Qpredicts = critic(obs_test[:, :-1, :], action_test[:, :-1, :])
        test_critic_loss = critic_loss_fun(test_target_values[:, -1, :], test_Qpredicts[:, -1, :])

        # test_actor_loss
        actor_test_pred_actions = actor(tf.concat([obs_test[:, :-1, :], action_test[:, :-1, :]], axis=2))
        actor_supervised_test_loss = actor_supervised_loss_func(actor_test_pred_actions[:, -1, :],
                                                                action_test[:, -1, :])
        test_predict_q_value = tf.math.reduce_mean(critic(obs_test[:, 1:, :], actor_test_pred_actions)[:, -1, :])
        test_supervised_actor_loss = actor_loss_weight * actor_supervised_test_loss - (
                    1 - actor_loss_weight) * test_predict_q_value

        # sava data
        if episode == num_episodes:
            actor_test_pred_actions = numpy.squeeze(actor_test_pred_actions[:, -1:, :].numpy())  # (None, act_dim)
            actor_test_actions = numpy.squeeze(action_test[:, -1:, :].numpy())
            actor_test_res_df = pd.DataFrame(numpy.concatenate([actor_test_pred_actions, actor_test_actions], axis=1))
            actor_test_res_df.to_excel(r'C:\Users\Think\Desktop\RL\rf_learning\Data\raw_data_RL-test-irssddpg-his.xlsx',
                                       index=None)

    supervised_actor_gradients = tape.gradient(supervised_actor_loss, actor.trainable_variables)

    actor_optimizer.apply_gradients(
        zip(supervised_actor_gradients, actor.trainable_variables))
    # print(actor.trainable_variables)
    soft_update_weights(target_critic.variables, critic.variables,
                        target_network_update_rate)
    soft_update_weights(target_critic_1.variables, critic_1.variables,
                        target_network_update_rate)
    soft_update_weights(target_actor.variables, actor.variables,
                        target_network_update_rate)

    print(
        f'critic_loss {critic_loss}: supervised_actor_loss= {supervised_actor_loss}: actor_supervised_loss = {actor_supervised_loss} : predict_q_value = {predict_q_value}'
        f'test_critic_loss = {test_critic_loss}: test_supervised_actor_loss= {test_supervised_actor_loss}: actor_supervised_test_loss= {actor_supervised_test_loss} : test_predict_q_value = {test_predict_q_value}')
    train_his_info.append([episode
                              , critic_loss.numpy()
                              , supervised_actor_loss.numpy()
                              , actor_supervised_loss.numpy()
                              , predict_q_value.numpy()
                              , test_critic_loss.numpy()
                              , test_supervised_actor_loss.numpy()
                              , actor_supervised_test_loss.numpy()
                              , test_predict_q_value.numpy()])

train_his_df = pd.DataFrame(train_his_info,
                            columns=['episode', 'critic_loss', 'supervised_actor_loss', 'actor_supervised_loss',
                                     'predict_q_value',
                                     'test_critic_loss', 'test_supervised_actor_loss', 'actor_supervised_test_loss',
                                     'test_predict_q_value'])
train_his_df.to_excel(r'C:\Users\Think\Desktop\RL\rf_learning\Data\raw_data_RL-train-irsddpg-his.xlsx', index=None)

# save model
actor.save(r'C:\Users\Think\Desktop\RL\rf_learning\models\irsddpg\actor')
