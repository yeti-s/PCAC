import sys
import os

from dev import Nuscene
sys.path.append('..')
from util import create_las

def main(args):
    if len(args) < 3:
        print('./exec [nuscene_directory] [target_path]')
        exit(1)
        
    root_path = args[1]
    target_path = args[2]
    
    nuscene = Nuscene(root_path)
    for token, scene in nuscene.get_scenes().items():
        points = scene.get_points()
        create_las(os.path.join(target_path, f'{token}.las'), points.numpy())

if __name__ == '__main__':
    main(sys.argv)