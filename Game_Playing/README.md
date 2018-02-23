# Game Playing
*	Using reinforcement learning agent defeat original rule based AI
*	DQN model for defeating breakout-v1 AI
*	Gradient Policy for defeating pong AI
*	Our DQN model using double DQN & duel DQN to improve its performance
##	Usage
training policy gradient:  
* `$ python3 main.py --train_pg`

testing policy gradient:  
* `$ python3 test.py --test_pg`

training DQN:  
* `$ python3 main.py --train_dqn`

testing DQN:  
* `$ python3 test.py --test_dqn`
##	Analysis
###	Pong (Gradient Policy)

![pong](https://github.com/willylulu/ADLxMLDS2017/blob/master/Game_Playing/char/pong.JPG?raw=true)
###	Breakout-v1 (DQN)

![break](https://github.com/willylulu/ADLxMLDS2017/blob/master/Game_Playing/char/breakout.JPG?raw=true)
### Breakout-v1 (Tuning)

![compare](https://github.com/willylulu/ADLxMLDS2017/blob/master/Game_Playing/char/comp.JPG?raw=true)
*	The best DQN model is using double DQN & duel DQN
##	Code
* DQN model is writing in agent_dir/agent_dqn.py
* Gradient Policy model is writing in agent_dir/agent_pg.py
* A2C model is our bouns in this project, written in  agent_dir/agent_a2c.py