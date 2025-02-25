# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 10:36:27 2025

@author: Blitxaac
"""

'''Positional data from simulations'''
import h5py
import re
import os
import imageio
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_frame(points):
    #ignore if not plotting Voronoi
    vor = Voronoi(points) #input points
    #regions, vertices = voronoi_finite_polygons_2d(vor)
    voronoi_plot_2d(vor, show_points=False)
    #add color to this, in terms of area - area calculation, color bar
    #to figure out how to plot the finite one

bet = [0.5,2,5,10]
kap = [0,0.1,0.25,0.5,1,2,5,10]
alp1 = [1,2,4,8,12]
alp2 = [1,2,4,8,12]

for ka in kap:
    for bt in bet:
# for a1 in alp1:
#     for a2 in alp2:
        #script_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        script_directory = "C:/Users/Blitxaac/Desktop/FYP/Simulation/Results/1.1.7/b=2"
        file = f"position_arrays_beta{bt}_kappa{ka}.h5"
        # file = f"position_arrays_{a1}_{a2}.h5"  
        hdf5_file_path = os.path.join(script_directory, file)
        save_directory = os.path.join(script_directory, "positions") 
        # Saves to a folder named "output" in the script's directory
        
        if not os.path.exists(hdf5_file_path):
            continue
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with h5py.File(hdf5_file_path, "r") as f:
            dataset_names = list(f.keys())
            
            # Sort the dataset names numerically
            dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
            
            for name in dataset_names_sorted:
                data = f[name][:]  # Load the dataset as a NumPy array
                
                # Image formatting
                #plot_frame(data)
                plt.scatter(data[:,0],data[:,1],s=0.5)
                plt.title(f'frame_{name}')
                plt.xlim(0,1)
                plt.ylim(0,1)
                
                frame_path = os.path.join(save_directory, f'frame_{name}.png')  # Full path for each frame
                plt.savefig(frame_path)  # Save frame as an image
                plt.close()

        gif_path = os.path.join(save_directory, f'{file}.gif')

        # Open the writer and add frames directly
        with imageio.get_writer(gif_path, mode="I", fps=8) as writer:
            for name in dataset_names_sorted:
                frame_path = os.path.join(save_directory, f'frame_{name}.png')
                if os.path.exists(frame_path):  # Ensure file exists before reading
                    writer.append_data(imageio.imread(frame_path))

        # Clean up: Delete the individual PNG frames
        for name in dataset_names_sorted:
            frame_path = os.path.join(save_directory, f'frame_{name}.png')
            if os.path.exists(frame_path):
                os.remove(frame_path)

#%% Removal
for a1 in alp1:
    for a2 in alp2:
        #script_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        script_directory = "C:/Users/Blitxaac/Desktop/FYP/Simulation/Results/1.1.4/positions"
        # file = f"position_arrays_beta{bt}_kappa{ka}.h5"
        file = f"position_arrays_{a1}_{a2}.h5"
        hdf5_file_path = os.path.join(script_directory, file)
        if os.path.exists(hdf5_file_path):
            os.remove(hdf5_file_path)

#%%
'''Velocity data of the particles based on their positions'''

import h5py
import numpy as np
import re
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

bet = [0,0.1,0.25,0.5,1,2,5,10,20]
kap = [0,0.1,0.25,0.5,1,2,5,10,20]
alp1 = [1,5,10,20]
alp2 = [1,5,10,20]

frameskips = 1

for ka in kap:
    for bt in bet:
        #script_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        script_directory = "C:/Users/Blitxaac/Desktop/FYP/Simulation/Results/1.1.5"
        
        file = f"velocity_arrays_beta{bt}_kappa{ka}.h5" #f"velocity_arrays_beta{bt}_kappa{ka}.h5"
        hdf5_file_path = os.path.join(script_directory, file)
        
        pos = f"position_arrays_beta{bt}_kappa{ka}.h5" #f"position_arrays_beta{bt}_kappa{ka}.h5"
        pos_path = os.path.join(script_directory, pos)
        
        save_directory = os.path.join(script_directory, "velocities") 
        # Saves to a folder named "output" in the script's directory

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with h5py.File(hdf5_file_path, "r") as f1, h5py.File(pos_path, "r") as f2:
            
            dataset_names = list(f1.keys())
            
            # Sort the dataset names numerically
            dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
            
            for i,name in enumerate(dataset_names_sorted):
                    vel = f1[name][:]
                    pos = f2[name][:]
                    
                    X = pos[:,0]
                    Y = pos[:,1]
                    U = vel[:,0]
                    V = vel[:,1]
                    M = np.hypot(U,V)  # Compute vector magnitudes

                    # Normalize Magnitude to [0,1]
                    norm = mcolors.Normalize(vmin=np.min(M), vmax=np.max(M))
                    colors = cm.viridis(norm(M))
                    
                    fig, ax = plt.subplots()
                    quiver = ax.quiver(X, Y, U, V, color=colors, angles="xy", scale_units="xy")
                    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
                    cbar.set_label("Velocity Magnitude")
                    plt.title(f'frame_{name}')

                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    
                    frame_path = os.path.join(save_directory, f'frame_{name}.png')  # Full path for each frame
                    plt.savefig(frame_path,dpi=300)  # Save frame as an image
                    plt.close()
                
            
        gif_path = os.path.join(save_directory, f'{file}.gif')

        # Open the writer and add frames directly
        with imageio.get_writer(gif_path, mode="I", fps=4) as writer:
            for name in dataset_names_sorted:
                frame_path = os.path.join(save_directory, f'frame_{name}.png')
                if os.path.exists(frame_path):  # Ensure file exists before reading
                    writer.append_data(imageio.imread(frame_path))

        # Clean up: Delete the individual PNG frames
        for name in dataset_names_sorted:
            frame_path = os.path.join(save_directory, f'frame_{name}.png')
            if os.path.exists(frame_path):
                os.remove(frame_path)

#%%
import h5py
import numpy as np
import re
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

bet = [0.5,1,2,5,10]
kap = [0.5,1,2,5,10]
alp1 = [1,5,10,20]
alp2 = [1]

frameskips = 4

# for a1 in alp1:
#     for a2 in alp2:
for bt in bet:
    for ka in kap:
        #script_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        script_directory = f"C:/Users/Blitxaac/Desktop/FYP/Simulation/Results/1.1.6/b={bt}"
        
        file = f"polarity_arrays_beta{bt}_kappa{ka}.h5"
        # file = f"polarity_arrays_{a1}_{a2}.h5"
        hdf5_file_path = os.path.join(script_directory, file)
        
        pos = f"position_arrays_beta{bt}_kappa{ka}.h5"
        # pos = f"position_arrays_{a1}_{a2}.h5"
        pos_path = os.path.join(script_directory, pos)
        
        save_directory = os.path.join(script_directory, "polarities") 
        # Saves to a folder named "output" in the script's directory

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with h5py.File(hdf5_file_path, "r") as f1, h5py.File(pos_path, "r") as f2:
            
            dataset_names = list(f1.keys())
            
            # Sort the dataset names numerically
            dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
            
            for i,name in enumerate(dataset_names_sorted):
                if i%frameskips == 0:
                    pol = f1[name][:]
                    pos = f2[name][:]
                    
                    X = pos[:,0]
                    Y = pos[:,1]
                    U = pol[:,0]
                    V = pol[:,1]
                    M = np.hypot(U,V)  # Compute vector magnitudes

                    # Normalize Magnitude to [0,1]
                    norm = mcolors.Normalize(vmin=np.min(M), vmax=np.max(M))
                    colors = cm.viridis(norm(M))
                    
                    fig, ax = plt.subplots()
                    quiver = ax.quiver(X, Y, U, V, color=colors, angles="xy", scale_units="xy")
                    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
                    cbar.set_label("Polarity Magnitude")
                    plt.title(f'frame_{name}')

                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    
                    frame_path = os.path.join(save_directory, f'frame_{name}.png')  # Full path for each frame
                    plt.savefig(frame_path,dpi=250)  # Save frame as an image
                    plt.close()
                
            
        gif_path = os.path.join(save_directory, f'{file}.gif')

        # Open the writer and add frames directly
        with imageio.get_writer(gif_path, mode="I", fps=3) as writer:
            for name in dataset_names_sorted:
                frame_path = os.path.join(save_directory, f'frame_{name}.png')
                if os.path.exists(frame_path):  # Ensure file exists before reading
                    writer.append_data(imageio.imread(frame_path))

        # Clean up: Delete the individual PNG frames
        for name in dataset_names_sorted:
            frame_path = os.path.join(save_directory, f'frame_{name}.png')
            if os.path.exists(frame_path):
                os.remove(frame_path)

#%%
# '''Adding animation for polarity change in the front layer(s)'''
# '''Or just an interactive'''

# import h5py
# import re
# import os
# import imageio
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# def polarity_calculations(pols):
#     mag = np.sqrt(pols[:,0]**2 + pols[:,1]**2)
#     rad = np.arctan2(pols[:,1],pols[:,0])
#     deg = np.degrees(rad)
#     return mag,deg

# def avg_polarity_mag(pols):
#     mag = np.sqrt(pols[:,0]**2 + pols[:,1]**2)
#     avg = np.sum(mag)/len(mag)
#     return avg

# def avg_polarity_deg(pols):
#     rad = np.arctan2(pols[:,1],pols[:,0])
#     avg = np.sum(rad)/len(rad)
#     deg = np.degrees(avg)
#     return deg

# script_directory = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
# hdf5_file_path = os.path.join(script_directory, "polarity_arrays.h5")
# save_directory = os.path.join(script_directory, "output") 

# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)
    
# #s = 1 # Frame skip

# with h5py.File(hdf5_file_path, "r") as f:
#     dataset_names = list(f.keys())
    
#     dataset_names_sorted = sorted(dataset_names, key=lambda x: int(re.search(r"\d+", x).group()))
    
#     frames = []
#     graph = np.zeros((1,2))
#     for i,name in enumerate(dataset_names_sorted):
#         if i==0:
#             continue
# #-----------------------------------------------------
#         data = f[name][:]
#         average = avg_polarity_deg(data)
#         temp = np.array([i*0.1,average])
#         graph = np.vstack((graph,temp))
    
#     x = graph[1:,0]
#     y = graph[1:,1]
    
#     fig, ax = plt.subplots()
#     line, = ax.plot(x, y, label="Magnitude")
#     tracker, = ax.plot([], [], 'ro', markersize=8)  # Red dot tracker
#     annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
#                          bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

#     # Function to update tracker and display coordinates
#     def on_mouse_move(event):
#         if event.xdata is not None and event.ydata is not None:
#             idx = np.abs(x - event.xdata).argmin()  # Find closest x index
#             x_val, y_val = x[idx], y[idx]
    
#             # Move tracker point
#             tracker.set_data([x_val], [y_val])
    
#             # Move annotation to follow tracker
#             annotation.set_text(f"({x_val:.2f}, {y_val:.5f})")
#             annotation.xy = (x_val, y_val)  # Attach directly to point
#             annotation.set_visible(True)
    
#             fig.canvas.draw_idle()

#     # Connect event to function
#     fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
#     plt.xlabel('Time(h)')
#     plt.xticks([i for i in range(round(len(graph)*0.1)+1)])
#     plt.minorticks_on()
#     plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
#     plt.legend()
#     plt.show()
#-----------------------------------------------------
#         if i%s==0:
#             data = f[name][:]
#             mag,deg = polarity_calculations(data)
#             plt.xlabel("Angle")
#             plt.scatter(deg,mag,label=f'{name}')
#             plt.title(f'frame_{name}')
#             #plt.xlim(-10,190) #angle
#             #plt.ylim(0,0.05) #magnitude
#             plt.legend()
        
#             frame_path = os.path.join(save_directory, f'frame_{name}.png')  # Full path for each frame
#             plt.savefig(frame_path)  # Save frame as an image
#             plt.close()
#             frames.append(imageio.imread(frame_path))
#         else:
#             continue
        
# gif_path = os.path.join(save_directory, 'polarity mag vs angle over time.gif')
# imageio.mimsave(gif_path, frames, fps=5)

# for i in range(len(dataset_names)):
#     frame_path = os.path.join(save_directory, f'frame_{i*s}.png')
#     if os.path.exists(frame_path):
#         os.remove(frame_path)