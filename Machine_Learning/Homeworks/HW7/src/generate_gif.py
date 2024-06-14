#!/usr/bin/env python3
"""! @brief Python program for creating GIF from images."""
##
# @file generate_gif.py
#
# @brief Functions to generate GIF from images.
#
##
#
# @section Libraries/Modules
# - os intern library (https://docs.python.org/3/library/os.html)
# - PIL extern library (https://pypi.org/project/pillow/)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
import os
from PIL import Image

def create_gifs(folder):
    # Searching png
    for root, dirs, files in os.walk(folder):
        # Storing paths
        images = {}
        
        for filename in files:
            if filename.endswith('.png') and not filename.endswith('.gif') and "epochs" in filename:
                # Assuming perplexity is part of the filename before the first underscore
                perplexity = filename.split('_')[0]

                if perplexity not in images:
                    images[perplexity] = []

                images[perplexity].append(os.path.join(root, filename))

        # Create a gif
        for perplexity, image_paths in images.items():
            # Sorting images by path
            image_paths.sort()

            # Load images
            images = [Image.open(image_path) for image_path in image_paths]

            # Creating GIF
            gif_path = os.path.join(root, f'all_epochs.gif')

            # Saving GIF
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)

            print(f'\nGIF created for group {perplexity} in {root}: {gif_path}\n')