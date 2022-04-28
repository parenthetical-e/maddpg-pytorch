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
	python run_infoduel_maddpg.py "simple_tag_v2" "test4" --eta=0.002 --kappa=1.0 --buffer_length=2500 --device="cuda:0"


# -------------------------------------------------------------------------