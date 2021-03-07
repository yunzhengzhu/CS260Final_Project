export DATA=comve
python ground_concepts_simple.py $DATA
python find_neighbours.py $DATA
python filter_triple.py $DATA
