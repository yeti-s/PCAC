import sys
import os

from dev import Nuscene
sys.path.append('..')
from util import create_las


def help():
    print('example:')
    print('exec.py [nuscene_directory]                                  (print all scenes in dataset)')
    print('exec.py [nuscene_directory] [target_path]                    (write las with all points in dataset)')
    print('exec.py [nuscene_directory] [target_path] [target_scene]     (write las with points of target scene)')
    exit(1)

def main(argv):
    argc = len(argv)
    if argc < 2:
        help()
    
    root_path = argv[1]
    nuscene = Nuscene(root_path)
    scenes = nuscene.get_scenes()
    
    if argc == 2:
        for token, scene in scenes.items():
            print(f'{token} [{scene.name}] : {scene.description}')
        return 0
    
    target_path = argv[2]
    if argc == 3:
        for token, scene in scenes.items():
            points = scene.get_points()
            create_las(os.path.join(target_path, f'{token}.las'), points.numpy())
        
        return 0
    
    target_scene = argv[3]
    if argc == 4:
        scene = scenes[target_scene]
        points = scene.get_points()
        create_las(os.path.join(target_path, f'{target_scene}.las'), points.numpy())
        
        return 0
    
    else:
        help()
        
if __name__ == '__main__':
    main(sys.argv)