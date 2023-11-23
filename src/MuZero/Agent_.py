import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Conv3D
from tensorflow.keras.regularizers import L2
Glorot_func = tf.keras.initializers.GlorotUniform(seed=0)
import numpy as np

from src.CPP.State import CPPState

import os



class MuZeroAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        #self.hidden_updated_layer_size = 32
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5

        # Global-Local Map
        self.use_global_local = True
        self.global_map_scaling = 3
        self.local_map_size = 17

class MuZeroAgent(object):
    def __init__(self, params, example_state: CPPState, example_action, stats=None):
        self.params = params
        gamma = tf.constant(0.99, dtype=float)

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.scalars = example_state.get_num_scalars()
        self.num_map_channels = self.boolean_map_shape[2]
        self.action_size = len(type(example_action))

        # Create shared inputs
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,
                  scalars_input]

        # Initialization
        self.state_size = states
        #self.state_shape = len(states)
        self.max_average = 0  # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.shuffle = False

        # Create network models
        self.Network_model = Network(input_states=self.state_size,
                                num_actions=self.action_size,
                                params=self.params)


        if stats:
            stats.set_model(self.Network_model.model)
class PartModel:
    def __init__(self, params):
        self.params = params

    def create_map_proc(self, conv_in, name='MuZeroNetwork'):
        conv_in = tf.convert_to_tensor(conv_in, dtype = tf.float32)
        global_map = tf.stop_gradient(
            AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

        self.global_map = global_map
        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                strides=(1, 1),
                                name=name + 'global_conv_' + str(k + 1))(global_map)

        flatten_global = Flatten(name=name + 'global_flatten')(global_map)

        crop_frac = float(self.params.local_map_size) / float(conv_in.shape[1])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        self.local_map = local_map

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        flatten_local = Flatten(name=name + 'local_flatten')(local_map)

        return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])


    def build_model(self, bool_map, scalars, inputs, num_actions, name):
        flatten_map = self.create_map_proc(bool_map)
        layer = tf.keras.layers.Concatenate(name=name + 'concat')([flatten_map, scalars])
        layer_ = layer

        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        hidden_state_output_layer = tf.keras.layers.Dense(self.params.hidden_layer_size, activation = 'relu', bias_initializer = Glorot_func,
                              kernel_regularizer=L2(1e-3), bias_regularizer = L2(1e-3))(layer)
        self.representation_function = tf.keras.Model(inputs, hidden_state_output_layer)
        print(tf.keras.Model.summary)



        #Building Dynamic layer
        hidden_state_input_layer = Input(self.params.hidden_layer_size)
        action_input_layer = Input(num_actions)
        concat_layer = Concatenate()([hidden_state_input_layer, action_input_layer])
        hidden_layer = tf.keras.layers.Dense(self.params.hidden_layer_size, activation='relu', bias_initializer=Glorot_func,
                             kernel_regularizer=L2(1e-3), bias_regularizer=L2(1e-3))(concat_layer)
        for _ in range(self.params.hidden_layer_num):
            hidden_layer = tf.keras.layers.Dense(self.params.hidden_layer_size, activation='relu', bias_initializer=Glorot_func,
                             kernel_regularizer=L2(1e-3), bias_regularizer = L2(1e-3))(hidden_layer)
        hidden_state_output_layer = tf.keras.layers.Dense(self.params.hidden_layer_size, activation='relu', bias_initializer =Glorot_func,
                             kernel_regularizer=L2(1e-3), bias_regularizer=L2(1e-3))(hidden_layer)
        transition_reward_output_layer = tf.keras.layers.Dense(1, activation='linear', bias_initializer=Glorot_func, \
                                               kernel_regularizer = L2(1e-3), bias_regularizer = L2(1e-3))(hidden_layer)

        self.dynamics_function = tf.keras.Model([hidden_state_input_layer,action_input_layer], \
                                       [hidden_state_output_layer, transition_reward_output_layer])

        #Building prediction function layers
        hidden_state_input_layer = Input(self.params.hidden_layer_size)
        hidden_layer = tf.keras.layers.Dense(self.params.hidden_layer_size, activation = 'relu', bias_initializer = Glorot_func, \
                             kernel_regularizer = L2(1e-3), bias_regularizer =L2(1e-3))(hidden_state_input_layer)
        for _ in range(self.params.hidden_layer_num):
            hidden_layer = tf.keras.layers.Dense(self.params.hidden_layer_size, activation = 'relu', bias_initializer =Glorot_func, \
                             kernel_regularizer = L2(1e-3), bias_regularizer = L2(1e-3))(hidden_layer)
        policy_output_layer = tf.keras.layers.Dense(num_actions, activation = 'softmax', bias_initializer=Glorot_func, \
                                    kernel_regularizer = L2(1e-3), bias_regularizer = L2(1e-3))(hidden_layer)
        value_output_layer = tf.keras.layers.Dense(1, activation = 'linear', bias_initializer=Glorot_func, \
                                    kernel_regularizer = L2(1e-3), bias_regularizer = L2(1e-3))(hidden_layer)

        self.prediction_function = tf.keras.Model(hidden_state_input_layer, [policy_output_layer, value_output_layer])

        print(self.representation_function.summary(), self.prediction_function.summary(), self.dynamics_function.summary())

        return self.representation_function, self.dynamics_function, self.prediction_function

    def build_combined_model(self, bool_map, inputs, num_actions, name):
        #inputs = Input(shape=bool_map.shape)
        #hidden_state_output_layer = self.create_map_proc(bool_map)
        hidden_state_output_layer = self.representation_function(inputs)

        action_input_layer = Input(shape=(num_actions,))
        hidden_state_output_layer, transition_reward_output_layer = self.dynamics_function(
            [hidden_state_output_layer, action_input_layer])

        policy_output_layer, value_output_layer = self.prediction_function(hidden_state_output_layer)

        model = tf.keras.Model(inputs=[inputs, action_input_layer],
                               outputs=[policy_output_layer, value_output_layer, transition_reward_output_layer])
        return model



    def transfrom_state(self, state: CPPState, for_prediction=False):
        bool_map = state.get_boolean_map()
        scalars = np.array(state.get_scalars(), dtype=np.single)
        state = [bool_map, scalars]
        if for_prediction:
            state = [state_oi[tf.newaxis, ...] for state_oi in state]
        return state


class Network(PartModel):
    def __init__(self, input_states, num_actions, params):
        super().__init__(params)
        self.representation, self.dynamic, self.prediction = self.build_model(bool_map=input_states[0],
                                      scalars=input_states[1],
                                      inputs = input_states,
                                      num_actions=num_actions,
                                      name = 'MuZero')
        self.model = self.build_combined_model(bool_map=input_states[0],
                                               inputs = input_states,
                                               num_actions= num_actions, name='MuZeroCombinedNetwork')


        def save(self, model_name):

            os.mkdir(f'models/{model_name}')
            self.representation_function.save_weights(f'models/{model_name}/representation_function_weights.h5')
            self.dynamics_function.save_weights(f'models/{model_name}/dynamics_function_weights.h5')
            self.prediction_function.save_weights(f'models/{model_name}/prediction_function_weights.h5')

        def load(self, model_name):

            self.representation_function.load_weights(f'models/{model_name}/representation_function_weights.h5')
            self.dynamics_function.load_weights(f'models/{model_name}/dynamics_function_weights.h5')
            self.prediction_function.load_weights(f'models/{model_name}/prediction_function_weights.h5')


