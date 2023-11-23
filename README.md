
## Table of contents
>>>>>>> b5d3836 (Initial Upload)

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
* [Resources](#resources)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the MuZero approach to control a UAV on coverage path planning including global-local map processing. The corresponding paper where DDQN has been implemented ["UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/9659413) is available on IEEEXplore. Basically custom environment from the paper is used and the MuZero algorithm is trained on it. The results obtained from MuZero are not as good as DDQN(implemented in the original paper). This implementation is only to check the performance of the MuZero on a given environment from the original paper from the above-mentioned link.

The repository also consists of data harvesting files but the Muzero algorithm has not been implemented for data harvesting.




## Requirements

```
You can find the requirement.txt file inside the repository.

## How to use

Train a new MuZero model with the parameters of your choice in the specified config file for Coverage Path Planning (CPP):

```
python main.py --cpp --gpu --config config/manhattan32_cpp.json --id manhattan32_cpp

--cpp             Activates CPP
--gpu                       Activates GPU acceleration for DDQN training
--config                    Path to config file in JSON format
--id                        Overrides standard name for logfiles and model
--generate_config           Enable only to write default config from default values in the code
```

Evaluate a model through Monte Carlo analysis over the random parameter space for the performance indicators 'Successful Landing', 'Collection Ratio', 'Collection Ratio and Landed' as defined in the paper (plus 'Boundary Counter' counting safety controller activations), e.g. for 1000 Monte Carlo iterations:

```
python main_mc.py --cpp --weights example/models/manhattan32_cpp --config config/manhattan32_cpp.json --id manhattan32_cpp_mc --samples 1000
>>>>>>> b5d3836 (Initial Upload)

--cpp|--dh                  Activates CPP or DH
--weights                   Path to weights of the trained model
--config                    Path to config file in JSON format
--id                        Name for exported files
--samples                   Number of Monte Carlo  over random scenario parameters
--seed                      Seed for repeatability
--show                      Pass '--show True' for individual plots of scenarios and allow plot saving
```

For an example run of pre-trained agents the following commands can be used:
```
python main_scenario.py --cpp --config config/manhattan32_cpp.json --weights example/models/manhattan32_cpp --scenario example/scenarios/manhattan_cpp.json --video
python main_scenario.py --cpp --config config/urban50_cpp.json --weights example/models/urban50_cpp --scenario example/scenarios/urban_cpp.json --video

```

## Resources

The city environments from the paper 'manhattan32' is included in the 'res' directory. Map information is formatted as PNG files with one pixel representing the on-grid world cell. The pixel colour determines the type of cell according to

* red #ff0000 no-fly zone (NFZ)
* blue #0000ff start and landing zone
* yellow #ffff00 buildings blocking wireless links (also obstacles for flying)

If you would like to create a new map, you can use any tool to design a PNG with the same pixel dimensions as the desired map and the above colour codes.

The shadowing maps, defining for each position and each IoT device whether there is a line-of-sight (LoS) or non-line-of-sight (NLoS) connection, are computed automatically the first time a new map is used for training and then saved to the 'res' directory as an NPY file. The shadowing maps are further used to determine which cells the field of view of the camera in a CPP scenario.


## Reference

If using this code for research purposes, please cite:

[1] M. Theile, H. Bayerlein, R. Nai, D. Gesbert, M. Caccamo, “UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning" 20th International Conference on Advanced Robotics (ICAR), 2021. 

```
@inproceedings{theile2021uav,
  title={UAV path planning using global and local map information with deep reinforcement learning},
  author={Theile, Mirco and Bayerlein, Harald and Nai, Richard and Gesbert, David and Caccamo, Marco},
  booktitle={2021 20th International Conference on Advanced Robotics (ICAR)},
  pages={539--546},
  year={2021},
  organization={IEEE}
}


## License 

This code is under a BSD license.
