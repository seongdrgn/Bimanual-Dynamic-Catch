# Learning Bimanual Catch Policy

## Simulation Environment (Isaac-Lab)

### Train Catch Policy with Multi-agent PPO (MAPPO)
> Environment version : 

* Training
  
   ```ruby
  python source/standalone/workflows/skrl/train.py --task Bimanual-Catch-mappo-skrl --num_envs=2048 --headless --algorithm MAPPO
   ```

* Evaluation
  
   ```ruby
  python source/standalone/workflows/skrl/play.py --taks Bimanul-Catch-mappo-skrl --num_envs=1 --checkpoint=/your_root/logs/rl_games/dynamic_catch_asym/2024-08-22_10-58-12/nn/dynamic_catch_asym.pth
   ```

## Real Environment

On going project...

## Troubleshooting

For anaconda users

> Users must complete installation the Isaac-Lab
```
cd /your/workspace/IsaacLab/_isaac_sim && source setup_conda_env.sh
```
