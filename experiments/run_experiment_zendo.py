import random
import torch
import csv
import os
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from type_system import BOOL, Arrow, List
from experiments.run_experiment import gather_data, list_algorithms
from DSL import zendo
import grammar.dsl as dsl
from model_loader import __build_generic_zendo_model, __buildintlist_zendo_model
from experiment_helper import task_set2zendodataset

dataset_name = "zendo"
save_folder = "experiment-output"


def load_zendo_dataset(pkl_path="data/test_tasks_filtered.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    return tasks


tasks = load_zendo_dataset()
print("Loaded", len(tasks))

base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical"]
max_objects = 7
zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
bigrams_cfg, bigrams_model = __build_generic_zendo_model(dsl=zendo_dsl, max_program_depth=5, size_max=11, size_hidden=64, embedding_output_dimension=78, number_layers_RNN=1, autoload=True, name="model_weights/bigramsPredictor.weights")
print(len(tasks), "tasks loaded.")
examples = [(task[0], task[1]) for task in tasks]
bigrams_dataset = task_set2zendodataset(examples[:50], bigrams_model, zendo_dsl, bigrams_cfg, use_model=True)

for algo_index in range(len(list_algorithms)):
    print("Running algorithm index:", algo_index)
    algo_name = list_algorithms[algo_index][1]
    if algo_name != "Heap Search":
        print(f"Skipping algorithm {algo_name} as it is not 'heap search'.")
        continue

    print("Starting...")
    for splits in [2]:
        for i, dataset in enumerate([bigrams_dataset]):
            filename = f"{save_folder}/bigramsPredictor_trained_with_red.csv"
            if os.path.exists(filename):
                print("Already exists:", filename)
                continue

            print(f"Running {algo_name} with {splits} CPUs...")
            data = gather_data(dataset, 0, 1, [], 10)
            col_names = ["task_name", "program", "search_time", "evaluation_time",
                         "nb_programs", "cumulative_probability", "accuracy", "probability"]

            processed_data = []
            for task_name, results in data:
                for result in results:
                    program, search_time, evaluation_time, nb_programs, cumulative_probability, accuracy, probability = result

                    processed_data.append([
                        str(task_name),
                        str(program),
                        search_time,
                        evaluation_time,
                        nb_programs,
                        cumulative_probability,
                        accuracy,
                        probability
                    ])

            with open(filename, "w", newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow(col_names)
                writer.writerows(processed_data)

            print("Saved results to", filename)
