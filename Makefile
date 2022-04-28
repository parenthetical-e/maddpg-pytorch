SHELL=/bin/bash -O expand_aliases

test1:
	python run_maddpg.py "simple_v2" "test1"

test2:
	python run_maddpg.py "simple_tag_v2" "test2"

test3:
	python run_infoduel_maddpg.py "simple_v2" "test3" --eta=0.002

test4:
	python run_infoduel_maddpg.py "simple_tag_v2" "test4" --eta=0.002

test5:
	python run_infoduel_maddpg.py "simple_tag_v2" "test4" --eta=0.002 --kappa=1.0
