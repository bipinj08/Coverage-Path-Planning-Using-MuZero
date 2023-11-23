import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from src.MuZero.replay_memory import ReplayBuffer

import numpy as np
import pickle

#from src.MuZero.classes import Node

def train(network_model, config, replay_buffer):

    with tf.GradientTape() as tape:
        print('training started')

        loss = 0

        #tf.keras.optimizers.legacy.Adam

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config['train']['learning_rate'],
                                                    beta_1=config['train']['beta_1'], beta_2=config['train']['beta_2'])

        for mem in replay_buffer.sample():

            memory_length = len(mem.reward_history)
            #print(memory_length, 'is length of memory..........................')
            sampled_index = np.random.choice(range(memory_length))
            memory_state= (mem.state_history[sampled_index])
            new_states = [np.asarray([memory_state.get_boolean_map()]).astype('float32'),
                          np.asarray([memory_state.movement_budget]).astype('float32')]
            hidden_state = network_model.representation_function(new_states)
            if (sampled_index + config['train']['num_unroll_steps']) < memory_length:
                num_unroll_steps = int(config['train']['num_unroll_steps'])
            else:
                num_unroll_steps = memory_length - 1 - sampled_index

            for start_index in range(sampled_index, sampled_index + num_unroll_steps):

                hidden_state, pred_reward = network_model.dynamics_function([hidden_state, mem.action_history[start_index]])
                #print(hidden_state.shape, pred_reward.shape, 'hidden state shape, reward shape')
                pred_policy, pred_value = network_model.prediction_function(hidden_state)
                #print(pred_policy, pred_value)

                if (memory_length - start_index - 1) >= config['train']['num_bootstrap_timesteps']:
                    true_value = sum([mem.reward_history[i] * (config['self_play']['discount_factor'] ** (i - start_index)) \
                                      for i in range(start_index, int(start_index + config['train']['num_bootstrap_timesteps']))]) + \
                                 mem.value_history[start_index + int(config['train']['num_bootstrap_timesteps'])] * \
                                 (config['self_play']['discount_factor'] ** (config['train']['num_bootstrap_timesteps']))

                else:
                    true_value = sum([mem.reward_history[i] * (config['self_play']['discount_factor'] ** (i - start_index)) for i in range(start_index, memory_length)])

                true_reward = mem.reward_history[start_index]
                true_policy = mem.policy_history[start_index + 1]
                true_policy = tf.convert_to_tensor(np.expand_dims(true_policy, axis=0))
                true_policy = tf.cast(true_policy, tf.float32)

                ### calculate loss ###
                loss += (1 / num_unroll_steps) * (mean_squared_error(true_reward, pred_reward) + mean_squared_error(true_value, pred_value) + binary_crossentropy(true_policy, pred_policy))  # take the average loss among all unroll steps
        loss += tf.reduce_sum(network_model.representation_function.losses) + tf.reduce_sum(network_model.dynamics_function.losses) + tf.reduce_sum(network_model.prediction_function.losses)  # regularization loss
        #self.loss = loss
        print(loss, 'is lossssssss')

    ### update network_model weights ###
    grads = tape.gradient(loss, [network_model.representation.trainable_variables,
                                 network_model.dynamic.trainable_variables,
                                 network_model.prediction.trainable_variables])
    #print(len(grads))
    #trainables = network_model.representation.trainable_variables + network_model.dynamic.trainable_variables + network_model.prediction.trainable_variables
    #print(len(trainables))

    #for r in range(len(grads)):
    #    optimizer.apply_gradients(zip(grads[r], trainables))


    #optimizer.apply_gradients( zip(grads[0], network_model.representation.trainable_variables))
    #optimizer.apply_gradients( zip(grads[1], network_model.dynamic.trainable_variables))
    #optimizer.apply_gradients( zip(grads[2], network_model.prediction.trainable_variables))

    trainables = [network_model.representation.trainable_variables,
                  network_model.dynamic.trainable_variables,
                  network_model.prediction.trainable_variables]

    for i in range(len(trainables)):
        if grads[i] is not None:
            optimizer.apply_gradients(zip(grads[i], trainables[i]))
