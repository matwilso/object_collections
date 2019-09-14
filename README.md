# Manipulating Object Collections Using Grounded State Representations

Code to accompany our paper.

⚠️ Disclaimer
This is a research codebase.
My hope is that it will be useful in the case that a reader of the paper wants 
to dig more into the specific details and implementation.
I have cleaned some things up a bit, but the code is still a bit messy.
This is not designed as a good jumping off point for future work.

Contents:
- [Codebase Outline](#codebase-outline)
- [Notes](#notes)
- [Installation](#installation)
- [Commands executed to run training](#commands-executed-to-run-training)
- [Acknowledgements](#acknowledgements)

## Codebase Outline
- `sl_main.py` - Script used to train SL models
- `rl_main.py` - Script used to train RL models
- `eval_main.py` - Script used to evaluate policies in simulation

### object_collections\/
- `define_flags.py`: Command line arg parsing.  Everything gets shoved into a dictionary called FLAGS.  It's super hacky, it's great.
- envs\/: simulation environment we use to train in, following OpenAI Gym API.
  - `object_env.py`: main file for env logic, with actions, resets, rewards, domain randomization, etc.
  - `base.py`: base class file with minimal stuff
  - utils\/: utils
- rl\/: RL training code, model definitions, algos, losses 
  - `agents.py`: Scripted policy logic and SAC losses
  - `data.py`: Data utils for reading and writing rollouts to file and using TensorFlow data API with TFRecords
  - `encoders.py`: Full encoder definitions that take state/image and encode them (our full method called DYN and an Autoencoder approach called VAE)
  - `models.py`: Models used within the encoders
  - `rollers.py`: Logic to do rollouts in parallel
  - `trainer.py`: Main logic to run training loop for RL
- scripts\/: Misc scripts for doing some things, including real world evaluations.  Require ROS/Baxter/MoveIt knowledge, not likely very useful.
- sl\/: Supervised learning code, losses 
  - `aux_losses.py`: For training MDN and Autoencoder heads
  - `building_blocks.py`: CNN archs, transformer/MHDPA implementation, MDN head (lower-level blocks)
  - `trainer.py` - Supervised learning training logic (main control loop)
  - `viz.py` - Code for visualizing MDNs, rollouts, etc.  Some gems, but overall not that happy with this code.

## Notes
- I call the Autoencoder model VAE in the code, sorry for any confusions this causes
- I parse all CMD line args into a dictionary called FLAGS which gets passed everywhere.  
- Some of the rewards processing is pretty hacky, so sorry about that.

## Installation
This requires installing Mujoco and mujoco_py.  Our python dependencies are
all listed in the requirements.txt file.  Installation can be a pain.

## Commands executed to run training 
(Arguments that are not passed in via command line default to the values in the `object_collections/define_flags.py`.)

### 1. Generate data
```
./rl_main.py --agent=scripted --dump_rollouts=1 --run_rl_optim=0 --goal_conditioned=False --debug=1 --rollout_data_path data/rollouts --num_envs=8 --use_embed=False --horizon=65 --max_episode_steps=65 --use_canonical=False
```

### 2. Train supervised learning model(s)
These reach 75k iterations (what we use in the paper) in about 15 hours (on my single i7 + NVIDIA-1080Ti).
```
# FULL:
 ./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=gn --use_image=False --phi_noise=0.0 

# MLP:
./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=mlp --use_image=False --phi_noise=0.0 --mlp_hidden_size=256 

# CNN:
./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=cnn_gn --phi_noise=0.1 

# CNN w/o MHDPA:
 ./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=cnn --phi_noise=0.1

# Only L STATE (GN, same for CNN):
./sl_main.py lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=gn --use_image=False --phi_noise=0.0 --dyn_weight=0.0

# Only L DYN (GN, same for CNN)
./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=gn --use_image=False --phi_noise=0.0 --mdn_weight=0.0 

# GNN Autoencoder:
./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=gnvae --phi_noise=0.0 --use_canonical=True

# CNN Autoencoder:
./sl_main.py --lr=3e-4 --bs=512 --goal_conditioned=False --cnn_gn=cnn_gn_vae --phi_noise=0.1 --use_canonical=True
```
### 2.b. Manually splice the trained models together
You have to train a state-based and image-based model and then rename some
of the weights and place them in a single checkpoint so that they can be loaded by the RL trainer.

See `rn_vars.py`.  

### 3. Train reinforcement learning policies
These reach 10k iterations (what we use in the paper) in about 10 hours.

```
# FULL:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024  --phi_noise=0.1 --is_training=False --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn --value_goal=True --goal_threshold=0.005

# MLP:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn_mlp --goal_threshold=0.005


# CNN w/o MHDPA:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn --goal_threshold=0.005


# Autoencoder:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn_vae --goal_threshold=0.2

# Only L STATE:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn --goal_threshold=0.04


# Only L DYN:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn --goal_threshold=0.004

# Image-based:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn --goal_threshold=0.005 --value_goal=False 

# No AAC:
./rl_main.py --goal_conditioned=True --lr=1e-3 --bs=1024 --phi_noise=0.1 --is_training=False  --goal_conditioned=True --load_path $PATH_TO_SL_MODEL_CKPT --cnn_gn=cnn_gn --goal_threshold=0.005 --value_goal=False --aac=False
```

### 4. Evaluate in sim 

```
# FULL 
./eval_main.py --goal_conditioned=True --num_envs=1 --eval_n=100 --phi_noise=0.0 --is_training=False --cnn_gn=cnn_gn --goal_conditioned=True --load_path $PATH_TO_RL_CKPT --aac=True --value_goal=True  --play=True  --render=1 --reset_mode=cluster --suffix=full
```

## Acknowledgements
Some code in this repo is borrowed from:
- Parallelizing data collection from a Gym environment: https://github.com/unixpickle/anyrl-py
- Seed implementation of Soft Actor Critic that I then modified: https://github.com/openai/spinningup
- Some RL/TensorFlow utilities https://github.com/openai/baselines
- TensorFlow Probability examples: https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples`
- graph_nets examples: https://github.com/deepmind/graph_nets




