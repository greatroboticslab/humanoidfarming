import os
import shutil
import argparse
import json

parser = argparse.ArgumentParser(description="Parse model argument")
parser.add_argument('--start', type=int, default=0, help='Start from this file #')
parser.add_argument('--end', type=int, default=-1, help='Stop processing at this file, set to -1 for all files from start.')
parser.add_argument('--dir', type=str, default='../../gerges/HumanoidRobotTrainingData/', help='Directory where all files to be proccessed are located.')

args = parser.parse_args()

repoDir = args.dir
framesDir = repoDir + "video_processing/frames/"
actionsDir = repoDir + "s1_baseline/output/tasks/"

frameFolders = [name for name in os.listdir(framesDir) if os.path.isdir(os.path.join(framesDir, name))]
actionFiles = [f for f in os.listdir(actionsDir) if os.path.isfile(os.path.join(actionsDir, f))]

#print(frameFolders) names not dir
#print(actionFiles)

idx = 0

for actionFile in actionFiles:
    for i in range(len(frameFolders)):
        #print("LEN: " + str(len(frameFolders)) + ", I: " + str(i))
        if actionFile[:len(actionFile)-5] == frameFolders[i]:
            print(actionFile[:len(actionFile)-5] + " matches " + frameFolders[i])
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
            frameFolder = framesDir + "/" + frameFolders[i] + "/raw_frames/"
            frames = [name for name in os.listdir(frameFolder) if os.path.isfile(os.path.join(frameFolder, name))]
            #print(len(frames))
            if len(tasks) > 0:
                entryAmount = min(len(tasks[0]), len(frames))
                for j in range(entryAmount):
                    #Copy frames over, write task as matching file.
                    idx += 1
                    idxs = f"{idx:06d}"
                    os.makedirs("../training_data/blip/", exist_ok=True)
                    shutil.copy(frameFolder + "/" + frames[j], "../training_data/blip/" + idxs + ".jpg")
                    taskFile = open("../training_data/blip/" + idxs + ".txt", "w")
                    taskFile.write(tasks[0][j])
                    taskFile.close()
