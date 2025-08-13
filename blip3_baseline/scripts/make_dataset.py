import os
import shutil
import argparse
import json

parser = argparse.ArgumentParser(description="Parse model argument")
parser.add_argument('--start', type=int, default=0, help='Start from this file #')
parser.add_argument('--end', type=int, default=-1, help='Stop processing at this file, set to -1 for all files from start.')
parser.add_argument('--dir', type=str, default='../../../gerges/HumanoidRobotTrainingData/', help='Directory where all files to be proccessed are located.')

args = parser.parse_args()

repoDir = args.dir
framesDir = repoDir + "video_processing/frames/"
actionsDir = repoDir + "s1_baseline/output/tasks/"

frameFolders = [name for name in os.listdir(framesDir) if os.path.isdir(os.path.join(framesDir, name))]
actionFiles = [f for f in os.listdir(actionsDir) if os.path.isfile(os.path.join(actionsDir, f))]

#print(frameFolders) names not dir
#print(actionFiles)

for actionFile in actionFiles:
    for i in range(len(frameFolders)):
        if actionFile[:len(actionFile)-5] == frameFolders[i]:
            print(actionFile[:len(actionFile)-5] + " matches " + frameFolders[i])
            with open(args.dir + actionsDir + actionFile, 'r', encoding='utf-8') as f:
                data = json.load(f)
