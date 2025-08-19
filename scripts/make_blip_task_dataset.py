import os
import shutil
import argparse
import json

parser = argparse.ArgumentParser(description="Parse model argument")
parser.add_argument('--start', type=int, default=0, help='Start from this file #')
parser.add_argument('--end', type=int, default=-1, help='Stop processing at this file, set to -1 for all files from start.')
parser.add_argument('--dir', type=str, default='../training_data/blip_tasks/original_tasks/', help='Directory where all files to be processed are located.')

args = parser.parse_args()

repoDir = args.dir
actionsDir = repoDir + ""

actionFiles = [f for f in os.listdir(actionsDir) if os.path.isfile(os.path.join(actionsDir, f))]

#print(frameFolders) names not dir
#print(actionFiles)

idx = 0

for actionFile in actionFiles:
    print(actionFile[:len(actionFile)-5])
    data = None
    with open(actionsDir + actionFile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    item = data
    tasks = []
    category = item.get("category")
    index = item.get("index")
    curFolder = str(category) + "/" + index + "/"
    for task_block in item.get("tasks", []):
        task_description = task_block.get("task")
        subtasks = task_block.get("subtasks", [])
        tasks.append([task_description] + subtasks)
    #print(tasks[0])

    for t in range(len(tasks)):
        question = "What steps are needed to perform this task: " + tasks[t][0]
        answer = ""
        for _t in range(1, len(tasks[t])):
            answer += str(tasks[t][_t]) + " "
        fOutput = question + "\n" + answer
        outFile = open("../training_data/blip_tasks/bliptask_training_data/" + str(t+1) + ".txt", "w")
        outFile.write(fOutput)
        outFile.close()

