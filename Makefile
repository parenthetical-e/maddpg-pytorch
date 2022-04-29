SHELL=/bin/bash -O expand_aliases

# -------------------------------------------------------------------------
# Some basic function tests for two easy tasks.
#
# We set 'cuda:0' to default to a GPU but it will
# get set to 'cpu' if needed
test1:
	python run_maddpg.py "simple_v2" "test1" --buffer_length=2500 --device="cuda:0"

test2:
	python run_maddpg.py "simple_tag_v2" "test2" --buffer_length=2500 --device="cuda:0"

test3:
	python run_infoduel_maddpg.py "simple_v2" "test3" --eta=0.002 --buffer_length=2500 --device="cuda:0"

test4:
	python run_infoduel_maddpg.py "simple_tag_v2" "test4" --eta=0.002 --buffer_length=2500 --device="cuda:0"

test5:
	python run_infoduel_maddpg.py "simple_tag_v2" "test5" --eta=0.002 --kappa=1.0 --buffer_length=2500 --device="cuda:0"


# -------------------------------------------------------------------------

# Experiments - all MPE 
# a) simple_v2
# b) simple_adversary_v2 (Physical deception)
# c) simple_crypto_v2 (Covert communication)
# d) simple_push_v2 (Keep-away)
# e) simple_reference_v2 (other agent location coop communication)
# f) simple_speaker_listener_v3 (Cooperative communication)
# g) simple_spread_v2 (Cooperative navigation)
# h) simple_tag_v2 (Predator-prey)
# i) simple_world_comm_v2 (more complex Predator-prey)
# simple_tag, except (1) there is food (small blue balls) that the good agents are rewarded for being near, (2) we now have ‘forests’ that hide agents inside from being seen from outside; (3) there is a ‘leader adversary” that can see the agents at all times, and can communicate with the other adversaries to help coordinate the chase.

# ----
# Sweep 1
# 1585e11

# ----
# a) simple_v2 (one agent, no parkid)
exp1m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp1m.parallel.log --header : python run_maddpg.py '"simple_v2" "exp1m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

exp1d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp1d.parallel.log --header : python run_infoduel_maddpg.py '"simple_v2" "exp1d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.00275 0.003

# ----
# b) simple_adversary_v2 (Physical deception)
# maddpg
exp2m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp2m.parallel.log --header : python run_maddpg.py '"simple_adversary_v2" "exp2m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp2d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp2d.parallel.log --header : python run_infoduel_maddpg.py '"simple_adversary_v2" "exp2d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp2k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp2k.parallel.log --header : python run_infoduel_maddpg.py '"simple_adversary_v2" "exp2k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 1.5

# ----
# c) simple_crypto_v2 (Covert communication)
# maddpg
exp3m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp3m.parallel.log --header : python run_maddpg.py '"simple_crypto_v2" "exp3m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp3d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp3d.parallel.log --header : python run_infoduel_maddpg.py '"simple_crypto_v2" "exp3d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp3k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp3k.parallel.log --header : python run_infoduel_maddpg.py '"simple_crypto_v2" "exp3k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 


# ----
# d) simple_push_v2 (Keep-away)
# maddpg
exp4m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp4m.parallel.log --header : python run_maddpg.py '"simple_push_v2" "exp4m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp4d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp4d.parallel.log --header : python run_infoduel_maddpg.py '"simple_push_v2" "exp4d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp4k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp4k.parallel.log --header : python run_infoduel_maddpg.py '"simple_push_v2" "exp4k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 

# ----
# e) simple_reference_v2 (other agent location coop communication)
# maddpg
exp5m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp5m.parallel.log --header : python run_maddpg.py '"simple_reference_v2" "exp5m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp5d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp5d.parallel.log --header : python run_infoduel_maddpg.py '"simple_reference_v2" "exp5d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp5k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp5k.parallel.log --header : python run_infoduel_maddpg.py '"simple_reference_v2" "exp5k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 

# ----
# f) simple_speaker_listener_v3 (Cooperative communication)
# maddpg
exp6m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp6m.parallel.log --header : python run_maddpg.py '"simple_speaker_listener_v3" "exp6m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp6d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp6d.parallel.log --header : python run_infoduel_maddpg.py '"simple_speaker_listener_v3" "exp6d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp6k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp6k.parallel.log --header : python run_infoduel_maddpg.py '"simple_speaker_listener_v3" "exp6k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 


# ----
# g) simple_spread_v2 (Cooperative navigation)
# maddpg
exp7m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp7m.parallel.log --header : python run_maddpg.py '"simple_spread_v2" "exp7m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp7d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp7d.parallel.log --header : python run_infoduel_maddpg.py '"simple_spread_v2" "exp7d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp7k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp7k.parallel.log --header : python run_infoduel_maddpg.py '"simple_spread_v2" "exp7k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 

# ----
# h) simple_tag_v2 (Predator-prey)
# maddpg
exp8m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp8m.parallel.log --header : python run_maddpg.py '"simple_tag_v2" "exp8m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp8d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp8d.parallel.log --header : python run_infoduel_maddpg.py '"simple_tag_v2" "exp8d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp8k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp8k.parallel.log --header : python run_infoduel_maddpg.py '"simple_tag_v2" "exp8k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 

# ----
# i) simple_world_comm_v2 (more complex Predator-prey)
# maddpg
exp9m: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp9m.parallel.log --header : python run_maddpg.py '"simple_world_comm_v2" "exp9m_lr{lr}" --buffer_length=2500 --device="cuda:0" --seed={seed} --lr={lr}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001

# infoduel-maddpg
exp9d: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp9d.parallel.log --header : python run_infoduel_maddpg.py '"simple_world_comm_v2" "exp9d_lr{lr}_eta{eta}" --buffer_length=2500 --device="cuda:1" --seed={seed} --lr={lr} --eta={eta}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035

# parkid-maddpg
exp9k: 
	parallel -j 4 --colsep , --delay 10 --joblog=models/exp9k.parallel.log --header : python run_infoduel_maddpg.py '"simple_world_comm_v2" "exp9k_lr{lr}_eta{eta}_kappa{kappa}" --buffer_length=2500 --device="cuda:2" --seed={seed} --lr={lr} --eta={eta} --kappa={kappa}' ::: seed 40 103 59 ::: lr 0.01 0.005 0.0025 0.001 ::: eta 0.0020 0.0025 0.003 0.0035 ::: kappa 1.0 
